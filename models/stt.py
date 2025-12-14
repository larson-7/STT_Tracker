from typing import Optional
import torch
import torch.nn as nn


# TODO: Incorporate this for variance predictions
def nll_loss(pred_state, pred_variance, target_state):
    """
    Used to guide log varaiance prediction.

    Gaussian NLL: -log p(y|μ,σ²) = 0.5 * (log(σ²) + (y-μ)²/σ²)
    """
    diff = target_state - pred_state
    loss = 0.5 * (torch.log(pred_variance) + (diff**2) / pred_variance)
    return loss.mean()


class DetectionEncoder(nn.Module):
    def __init__(self, input_dim=8, num_sensor_types=3, embed_dim=16, out_dim=256):
        """
        Encodes raw detection measurements into a high-dimensional embedding space.
        Input: Raw detection vector (e.g., x, y, z, vx, vy, vz, ax, ay, az, quality) and Sensor ID
        Output: Detection embedding.
        """
        super().__init__()

        # Dedicated embedding for the sensor type
        self.sensor_embedding = nn.Embedding(num_sensor_types, embed_dim)
        combined_dim = input_dim + embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, features, sensor_ids):
        # features shape:   [..., input_dim]
        # sensor_ids shape: [..., 1]

        # Create the learnable vector for the sensor type
        sensor_vecs = self.sensor_embedding(sensor_ids)  # shape: [..., embed_dim]

        if len(sensor_vecs.shape) > len(features.shape):
            sensor_vecs = sensor_vecs.squeeze(1)
        # Fuse them together, we concatenate along the last dimension (feature dimension)
        combined = torch.cat([features, sensor_vecs], dim=-1)

        return self.mlp(combined)


class TemporalEncoder(nn.Module):
    """
    Fuses the history of track embeddings into a single 'Track Query'.
    Input: Sequence of past track embeddings (history).
    Output: Single vector 'track_query'.
    """

    def __init__(self, hidden_dim=256, nhead=4, num_layers=2):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # TODO: positional offset here

    def forward(self, track_history):
        """
        track_history: [batch_size, history_length, hidden_dim]
        """
        seq_len = track_history.size(1)

        # Create Causal Mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
            track_history.device
        )

        # Apply the mask during the forward pass
        temporal_features = self.transformer_encoder(
            track_history, mask=mask, is_causal=True
        )

        # Take the last valid token as the query for current timestep
        track_query = temporal_features[:, -1, :]

        return track_query


class TrackDetectionInteraction(nn.Module):
    """
    The TDI Module.
    Performs Cross-Attention between the 'track_query' and 'context_detections'.
    Outputs:
      1. Updated track embedding (fused with best detection info).
      2. Association scores (classification logits).
    """

    def __init__(self, hidden_dim=256, nhead=4):
        super().__init__()
        # Query is Track, Key/Value are Detections
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=nhead, batch_first=True
        )

        # MLP for normalization/processing after attention (standard Transformer block)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Association Score Predictor
        # Takes the result of interaction to decide if it was a good match
        self.association_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        track_query,
        context_detections,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        track_query: [batch_size, 1, hidden_dim] (The active track)
        context_detections: [batch_size, num_context_dets, hidden_dim] (The candidates)
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and float masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        """
        # Query = track_query, Key = context_detections, Value = context_detections
        attn_output, attn_weights = self.multihead_attn(
            query=track_query,
            key=context_detections,
            value=context_detections,
            key_padding_mask=key_padding_mask,
        )

        # Residual connection + Norm
        # This 'updated_embedding' contains information from the attended detections
        updated_embedding = self.norm(track_query + attn_output)
        updated_embedding = updated_embedding + self.ffn(updated_embedding)

        # Predict Association Scores (Logits for binary classification/ranking)
        association_scores = self.association_head(updated_embedding)

        # TODO: Need to threshold distance here to help runtime/memory
        # TODO: Should I add an additional MLP decoder here instead of using TrackStateDecoder???
        return updated_embedding, association_scores


class TrackStateDecoder(nn.Module):
    """
    Decodes the abstract embedding back into a physical kinematic state.
    Input: The updated track embedding from the TDI.
    Output: State vector (e.g., x, y, z, vx, vy, ax, ay, track_quality).
    """

    def __init__(self, hidden_dim=256, state_dim=10):
        super().__init__()
        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        # Separate heads for mean and log variance
        self.state_head = nn.Linear(hidden_dim // 2, state_dim - 1)  # kinematics
        self.log_var_head = nn.Linear(hidden_dim // 2, state_dim - 1)  # uncertainties

    def forward(self, updated_track_embedding):
        features = self.decoder_mlp(updated_track_embedding)

        kinematics = self.state_head(features)  # [B, 1, 9]
        log_variance = self.log_var_head(features)  # [B, 1, 9]

        # Ensure positive variance
        variance = torch.exp(log_variance)

        return kinematics, variance


class STTTracker(nn.Module):
    def __init__(
        self,
        num_tracks: int = 20,
        input_dim: int = 9,
        sensor_type_embedding_dim: int = 16,
        num_sensor_modalities: int = 3,
        embedding_dim: int = 256,
        max_history_len: int = 1000,
    ):
        super().__init__()
        self.num_tracks = num_tracks
        self.embedding_dim = embedding_dim
        self.max_history_len = max_history_len

        self.detection_encoder = DetectionEncoder(
            input_dim=input_dim,
            num_sensor_types=num_sensor_modalities,
            embed_dim=sensor_type_embedding_dim,
            out_dim=embedding_dim,
        )

        self.temporal_encoder = TemporalEncoder(embedding_dim)
        self.tdi_module = TrackDetectionInteraction(embedding_dim)
        self.state_decoder = TrackStateDecoder(embedding_dim, input_dim)

        # Learn distinct embeddings for each of the N track slots
        # Shape: [1, Num_Tracks, Embed_Dim]
        self.track_query_embed = nn.Parameter(torch.randn(1, num_tracks, embedding_dim))

    def forward(self, batch):
        features = batch["obs_features"]  # [B, Seq, Max_Dets, Dim]
        sensor_ids = batch["obs_ids"]
        mask = batch["obs_mask"]

        batch_size, seq_len, max_dets, _ = features.shape
        flat_feats = features.view(-1, features.shape[-1])
        flat_ids = sensor_ids.view(-1, 1)
        encoded_dets = self.detection_encoder(flat_feats, flat_ids)

        # Reshape to [Batch, Seq, Max_Detections, Embed_Dim]
        encoded_dets = encoded_dets.view(
            batch_size, seq_len, max_dets, self.embedding_dim
        )
        mask_expanded = mask.unsqueeze(-1).expand_as(encoded_dets)
        encoded_dets = encoded_dets * mask_expanded.float()

        # Expand the detections so each of the N tracks sees the scene.
        # New Shape: [Batch * Num_Tracks, Seq, Max_Detections, Embed_Dim]
        encoded_dets_expanded = encoded_dets.repeat_interleave(self.num_tracks, dim=0)
        mask_expanded = mask.repeat_interleave(self.num_tracks, dim=0)

        # Initialize History for ALL N tracks
        # Queries: [Batch, Num_Tracks, Embed_Dim]
        init_queries = self.track_query_embed.expand(batch_size, -1, -1)

        # Fold into effective batch: [Batch * Num_Tracks, 1, Embed_Dim]
        track_history = init_queries.reshape(
            batch_size * self.num_tracks, 1, self.embedding_dim
        )

        # Storage for outputs
        all_kinematics = []
        all_variances = []
        all_association_scores = []

        for t in range(seq_len):
            # Context: [Batch * Num_Tracks, Max_Detections, Embed_Dim]
            current_context = encoded_dets_expanded[:, t, :, :]
            current_padding_mask = ~mask_expanded[:, t, :]

            track_query = self.temporal_encoder(track_history)  # [Batch * N, Dim]
            track_query = track_query.unsqueeze(1)  # [Batch * N, 1, Dim]

            # Each track slot attends to the same detections, but has a different query vector
            updated_embedding, assoc_score = self.tdi_module(
                track_query, current_context, key_padding_mask=current_padding_mask
            )

            # Decode
            kin, var = self.state_decoder(updated_embedding)

            all_kinematics.append(kin.squeeze(1))
            all_variances.append(var.squeeze(1))
            all_association_scores.append(assoc_score.squeeze(1))

            # Update History
            track_history = torch.cat([track_history, updated_embedding], dim=1)
            if track_history.shape[1] > self.max_history_len:
                track_history = track_history[:, -self.max_history_len :, :]

        # Unfold / Reshape Output
        # Stack time: [Batch * Num_Tracks, Seq_Len, State_Dim]
        kinematics_stacked = torch.stack(all_kinematics, dim=1)
        variance_stacked = torch.stack(all_variances, dim=1)
        scores_stacked = torch.stack(all_association_scores, dim=1)

        # Reshape to separate Batch and Tracks
        # Final Shape: [Batch, Num_Tracks, Seq_Len, State_Dim]
        def unfold(tensor):
            _, s_len, dim = tensor.shape
            return tensor.view(batch_size, self.num_tracks, s_len, dim)

        return {
            "kinematics": unfold(kinematics_stacked),
            "variance": unfold(variance_stacked),
            "association_scores": unfold(scores_stacked),
        }


if __name__ == "__main__":
    import torch

    # Dimensions
    BATCH_SIZE = 2  # e.g., 2 separate snippets of data (Scene A, Scene B)
    SEQ_LEN = 5  # 5 timesteps of history provided
    MAX_DETS = 10  # Max 10 detections per frame (some will be padding)
    FEAT_DIM = 9  # (x, y, z, v...)
    NUM_TRACKS = 20  # The model will output 20 track slots per scene
    NUM_SENSORS = 3  # Radar, Camera, Lidar types
    EMBED_DIM = 256

    print(f"--- Configuration ---")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Sequence Length: {SEQ_LEN}")
    print(f"Num Track Slots: {NUM_TRACKS}")
    print(f"Effective Processing Batch: {BATCH_SIZE * NUM_TRACKS} (Batch * Num_Tracks)")
    print("-" * 30)

    tracker = STTTracker(
        num_tracks=NUM_TRACKS,
        input_dim=FEAT_DIM,
        num_sensor_modalities=NUM_SENSORS,
        embedding_dim=EMBED_DIM,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker.to(device)
    print(f"Model created on device: {device}\n")

    # Random kinematic features: [Batch, Seq, Max_Dets, Feat_Dim]
    obs_features = torch.randn(BATCH_SIZE, SEQ_LEN, MAX_DETS, FEAT_DIM).to(device)

    # Random sensor IDs (0, 1, or 2): [Batch, Seq, Max_Dets, 1]
    obs_ids = torch.randint(0, NUM_SENSORS, (BATCH_SIZE, SEQ_LEN, MAX_DETS, 1)).to(
        device
    )

    # Random Mask: [Batch, Seq, Max_Dets]
    # Let's say ~30% are "False" (padding/empty slots), 70% are "True" (real detections)
    obs_mask = (torch.rand(BATCH_SIZE, SEQ_LEN, MAX_DETS) > 0.3).to(device)

    # Pack into dictionary
    batch = {"obs_features": obs_features, "obs_ids": obs_ids, "obs_mask": obs_mask}

    print("Running Forward Pass...")
    outputs = tracker(batch)

    kinematics = outputs["kinematics"]
    variance = outputs["variance"]
    scores = outputs["association_scores"]

    print("\n--- Output Shapes ---")
    print(f"Kinematics: {kinematics.shape}")
    # Expect: [Batch, Num_Tracks, Seq_Len, Out_Dim]

    print(f"Variance:   {variance.shape}")
    # Expect: [Batch, Num_Tracks, Seq_Len, Out_Dim]

    print(f"Scores:     {scores.shape}")
    # Expect: [Batch, Num_Tracks, Seq_Len, 1]

    assert kinematics.shape[0] == BATCH_SIZE
    assert kinematics.shape[1] == NUM_TRACKS
    assert kinematics.shape[2] == SEQ_LEN

    # Example: Check the score of the 1st track in the 1st batch at the last timestep
    last_frame_score = scores[0, 0, -1, 0].item()
    print(
        f"Sample Association Score (Batch 0, Track 0, Final Frame): {last_frame_score:.4f}"
    )
