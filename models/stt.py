from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

NEAR_NEG_INF = -1e9

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
        self.unknown_idx = num_sensor_types 
        self.sensor_embedding = nn.Embedding(num_sensor_types + 1, embed_dim)
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
        safe_ids = sensor_ids.clone()
        safe_ids[safe_ids == -1] = self.unknown_idx

        # Create the learnable vector for the sensor type
        sensor_vecs = self.sensor_embedding(safe_ids)  # shape: [..., embed_dim]

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
    "White Box" Transformer Attention Block.

    Performs Cross-Attention between 'track_query' and 'context_detections'.
    Unlike standard nn.MultiheadAttention, this exposes the raw attention logits
    (pre-softmax) for Association Loss (L_d) and the normalized probabilities
    for inference-time matching.

    Outputs (Tuple):
      1. Updated track embedding (fused with detection info).
      2. Association logits (raw scores, for Loss calculation).
      3. Association probabilities (Sigmoid scores, for Inference matching).
    """

    def __init__(self, hidden_dim: int = 256, nhead: int = 4):
        super().__init__()
        self.nhead = nhead
        self.head_dim = hidden_dim // nhead
        self.scale = self.head_dim**-0.5

        assert (
            self.head_dim * nhead == hidden_dim
        ), "Hidden dim must be divisible by nhead"

        # Q, K, V Projections (Shared by Loss AND State Update)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection for the state update path
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Standard Transformer Block components
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        track_query: torch.Tensor,
        context_detections: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            track_query: [Batch, Num_Tracks, Dim]
                The active track embeddings.
            context_detections: [Batch, Num_Dets, Dim]
                The candidate detections for this frame.
            key_padding_mask: [Batch, Num_Dets]
                True indicates the index is padding and should be ignored.

        Returns:
            updated_embedding: [Batch, Num_Tracks, Dim]
                The track states updated with information from relevant detections.
            association_logits: [Batch, Num_Tracks, Num_Dets]
                Raw similarity scores (logits) averaged across heads.
                Use this with BCEWithLogitsLoss.
            association_probs: [Batch, Num_Tracks, Num_Dets]
                Sigmoid-normalized probabilities (0.0 to 1.0).
                Use this for Hungarian Matching or Thresholding during inference.
        """
        B, N_tracks, C = track_query.shape
        _, N_dets, _ = context_detections.shape

        # Reshape to [Batch, Num_Elements, Num_Heads, Head_Dim] -> Transpose to [B, Heads, N, Head_Dim]
        q = (
            self.q_proj(track_query)
            .view(B, N_tracks, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(context_detections)
            .view(B, N_dets, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(context_detections)
            .view(B, N_dets, self.nhead, self.head_dim)
            .transpose(1, 2)
        )

        # Matrix Mult: (B, Heads, N_tracks, Head_Dim) @ (B, Heads, Head_Dim, N_dets)
        # Output: [B, Heads, N_tracks, N_dets]
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            # key_padding_mask: [B, N_dets] -> Expand to [B, 1, 1, N_dets]
            mask_expanded = key_padding_mask.view(B, 1, 1, N_dets)
            attn_logits = attn_logits.masked_fill(mask_expanded, NEAR_NEG_INF)

        attn_weights = F.softmax(attn_logits, dim=-1)

        # Weighted sum of Values: [B, Heads, N_tracks, Head_Dim]
        out = attn_weights @ v

        # Reassemble Heads: [B, N_tracks, Hidden_Dim]
        out = out.transpose(1, 2).contiguous().view(B, N_tracks, C)

        # Final Projection, Residual, and FFN
        out = self.out_proj(out)
        updated_embedding = self.norm(track_query + out)
        updated_embedding = updated_embedding + self.ffn(updated_embedding)

        # Sigmoid association scores used during inference
        # We allow the model to predict "0" for everything if no match exists
        mean_association_logits = attn_logits.mean(dim=1)
        association_probs = mean_association_logits.sigmoid()

        return updated_embedding, mean_association_logits, association_probs


class TrackStateDecoder(nn.Module):
    """
    Decodes the abstract embedding back into a physical kinematic state.
    Input: The updated track embedding from the TDI.
    Output: State vector (e.g., x, y, z, vx, vy, ax, ay).
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
        self.state_head = nn.Linear(hidden_dim // 2, state_dim)  # kinematics
        self.log_var_head = nn.Linear(hidden_dim // 2, state_dim)  # uncertainties

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
        self.prior_state_decoder = TrackStateDecoder(embedding_dim, input_dim)
        self.posterior_state_decoder = TrackStateDecoder(embedding_dim, input_dim)

        # Learn distinct embeddings for each of the N track slots
        # Shape: [1, Num_Tracks, Embed_Dim]
        self.track_query_embed = nn.Parameter(torch.randn(1, num_tracks, embedding_dim))

    def forward(self, batch):
        features = batch["obs_features"]  # [B, Seq, Max_Dets, Dim]
        sensor_ids = batch["obs_ids"]
        mask = batch["obs_mask"]
        # TODO: incorporate ownship position, run through encoder, stack and cross attend before TDI module?

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
        all_prior_kinematics = []
        all_prior_variances = []
        all_posterior_kinematics = []
        all_posterior_variances = []
        all_association_scores = []

        for t in range(seq_len):
            # Context: [Batch * Num_Tracks, Max_Detections, Embed_Dim]
            current_context = encoded_dets_expanded[:, t, :, :]
            current_padding_mask = ~mask_expanded[:, t, :]

            track_query = self.temporal_encoder(track_history)  # [Batch * N, Dim]
            track_query = track_query.unsqueeze(1)  # [Batch * N, 1, Dim]

            prior_kin, prior_var = self.prior_state_decoder(track_query)

            # TODO gate queries based on prior_kin (maybe var too via BD distance?) with the tracks and detections

            # Each track slot attends to the same detections, but has a different query vector
            updated_embedding, assoc_score, _ = self.tdi_module(
                track_query, current_context, key_padding_mask=current_padding_mask
            )

            # Decode
            posterior_kin, posterior_var = self.posterior_state_decoder(
                updated_embedding
            )
            all_prior_kinematics.append(prior_kin.squeeze(1))
            all_prior_variances.append(prior_var.squeeze(1))
            all_posterior_kinematics.append(posterior_kin.squeeze(1))
            all_posterior_variances.append(posterior_var.squeeze(1))
            all_association_scores.append(assoc_score.squeeze(1))

            # Update History
            track_history = torch.cat([track_history, updated_embedding], dim=1)
            if track_history.shape[1] > self.max_history_len:
                track_history = track_history[:, -self.max_history_len :, :]

        # Unfold / Reshape Output
        # Stack time: [Batch * Num_Tracks, Seq_Len, State_Dim]
        prior_kinematics_stacked = torch.stack(all_prior_kinematics, dim=1)
        prior_variances_stacked = torch.stack(all_prior_variances, dim=1)
        posterior_kinematics_stacked = torch.stack(all_posterior_kinematics, dim=1)
        posterior_variance_stacked = torch.stack(all_posterior_variances, dim=1)
        scores_stacked = torch.stack(all_association_scores, dim=1)

        # Reshape to separate Batch and Tracks
        # Final Shape: [Batch, Num_Tracks, Seq_Len, State_Dim]
        def unfold(tensor):
            _, s_len, dim = tensor.shape
            return tensor.view(batch_size, self.num_tracks, s_len, dim)

        return {
            "prior_kinematics": unfold(prior_kinematics_stacked),
            "prior_variance": unfold(prior_variances_stacked),
            "posterior_kinematics": unfold(posterior_kinematics_stacked),
            "posterior_variance": unfold(posterior_variance_stacked),
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

    kinematics = outputs["posterior_kinematics"]
    variance = outputs["posterior_variance"]
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
