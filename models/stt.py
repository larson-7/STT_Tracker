import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

NEAR_NEG_INF = -1e4


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
        # Set padding_idx to prevent gradient updates for unknown sensors
        self.sensor_embedding = nn.Embedding(
            num_sensor_types + 1, embed_dim, padding_idx=self.unknown_idx
        )

        nn.init.uniform_(self.sensor_embedding.weight, -0.1, 0.1)
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
    Fuses the history of track embeddings into a single 'Track Query' with also considering the ownship context.
    Input: Sequence of past track embeddings (history).
    Output: Single vector 'track_query'.
    """

    pe: torch.Tensor

    def __init__(
        self, hidden_dim=256, nhead=4, num_layers=2, max_len=5000, ownship_dim=6
    ):
        super().__init__()

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Use double precision for stability
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float64)
            * (-math.log(10000.0) / hidden_dim)
        ).float()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.ownship_mlp = nn.Sequential(
            nn.Linear(ownship_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, track_history, ownship_state):
        """
        track_history: [Batch, History_Length, Hidden_Dim]
        ownship_state: [Batch, Ownship_Dim]
        """
        seq_len = track_history.size(1)
        x = track_history + self.pe[:, :seq_len, :]

        if seq_len == 1:
            temporal_features = x
        else:
            temporal_features = self.transformer_encoder(x)

        # The 'History Query' is the last token
        history_query = temporal_features[:, -1, :]  # [Batch, Hidden_Dim]
        ownship_embed = self.ownship_mlp(ownship_state)  # [Batch, Hidden_Dim]
        combined = torch.cat([history_query, ownship_embed], dim=-1)
        final_track_query = self.fusion_proj(combined)

        final_track_query = self.norm(final_track_query)

        # Explicit NaN/Inf checks after LayerNorm with fallback to zeros
        if torch.isnan(final_track_query).any() or torch.isinf(final_track_query).any():
            final_track_query = torch.nan_to_num(
                final_track_query, nan=0.0, posinf=0.0, neginf=0.0
            )

        return final_track_query


class TrackDetectionInteraction(nn.Module):
    """
    Transformer Attention Block.

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

        # A constant vector representing "No Detection Found"
        self.null_obs_embedding = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.xavier_uniform_(self.null_obs_embedding, gain=0.1)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

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

        B, N_tracks, C = track_query.shape
        _, N_dets, _ = context_detections.shape

        # Expand null token to batch size: [B, 1, C]
        null_obs = self.null_obs_embedding.expand(B, -1, -1)

        # Concatenate null token to the end of detections
        # New shape: [B, N_dets + 1, C]
        context_with_null = torch.cat([context_detections, null_obs], dim=1)

        # Handle Mask: Append 'False' (Not Padding) for the null token so we can always attend to it
        if key_padding_mask is not None:
            # key_padding_mask is [B, N_dets]
            # null_mask is [B, 1] of False (valid)
            null_mask = torch.zeros(
                (B, 1), dtype=torch.bool, device=key_padding_mask.device
            )
            key_padding_mask_with_null = torch.cat([key_padding_mask, null_mask], dim=1)
        else:
            key_padding_mask_with_null = None

        q = (
            self.q_proj(track_query)
            .view(B, N_tracks, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        # Dimensions will operate on N_dets + 1
        k = (
            self.k_proj(context_with_null)
            .view(B, N_dets + 1, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(context_with_null)
            .view(B, N_dets + 1, self.nhead, self.head_dim)
            .transpose(1, 2)
        )

        # Attention Logits: [B, Heads, N_tracks, N_dets + 1]
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        attn_logits = torch.clamp(attn_logits, min=-100.0, max=100.0)

        # Apply Mask (now includes the Null token slot which is always valid)
        if key_padding_mask_with_null is not None:
            mask_expanded = key_padding_mask_with_null.view(B, 1, 1, N_dets + 1)
            attn_logits = attn_logits.masked_fill(mask_expanded, NEAR_NEG_INF)

        attn_weights = F.softmax(attn_logits, dim=-1)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, N_tracks, C)
        out = self.out_proj(out)

        fused = track_query + out
        updated_embedding = self.norm(fused)
        updated_embedding = updated_embedding + self.ffn(updated_embedding)

        # For Association Loss, we only care about the REAL detections.
        # Slice off the Null Token column from the logits.
        # Shape becomes: [B, Heads, N_tracks, N_dets]
        real_det_logits = attn_logits[..., :-1]

        mean_association_logits = real_det_logits.mean(dim=1)
        association_probs = mean_association_logits.sigmoid()

        return updated_embedding, mean_association_logits, association_probs


class TrackStateDecoder(nn.Module):
    def __init__(self, hidden_dim=256, state_dim=10):
        super().__init__()
        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.state_head = nn.Linear(hidden_dim // 2, state_dim)
        self.log_var_head = nn.Linear(hidden_dim // 2, state_dim)
        self.existence_head = nn.Linear(
            hidden_dim // 2, 1
        )  # Output: 1 logit (is this track alive?)

        # Initialize log_var head to predict small variances initially
        torch.nn.init.constant_(self.log_var_head.weight, 0)
        torch.nn.init.constant_(
            self.log_var_head.bias, -1.0
        )  # Start with exp(-1) variance

    def forward(self, updated_track_embedding):
        features = self.decoder_mlp(updated_track_embedding)

        kinematics = self.state_head(features)
        log_variance = self.log_var_head(features)
        log_variance = torch.clamp(log_variance, min=-10.0, max=10.0)
        existence_logits = self.existence_head(features)

        variance = torch.exp(log_variance)

        return kinematics, variance, existence_logits


class STTTracker(nn.Module):
    def __init__(
        self,
        num_tracks: int = 20,
        input_dim: int = 9,
        sensor_type_embedding_dim: int = 16,
        num_sensor_modalities: int = 3,
        embedding_dim: int = 256,
        ownship_dim: int = 6,
        max_history_len: int = 1000,
    ):
        super().__init__()
        self.num_tracks = num_tracks
        self.embedding_dim = embedding_dim
        self.max_history_len = max_history_len
        self.ownship_dim = ownship_dim

        self.detection_encoder = DetectionEncoder(
            input_dim=input_dim,
            num_sensor_types=num_sensor_modalities,
            embed_dim=sensor_type_embedding_dim,
            out_dim=embedding_dim,
        )

        self.temporal_encoder = TemporalEncoder(embedding_dim, ownship_dim=ownship_dim)
        self.tdi_module = TrackDetectionInteraction(embedding_dim)
        self.prior_state_decoder = TrackStateDecoder(embedding_dim, input_dim)
        self.posterior_state_decoder = TrackStateDecoder(embedding_dim, input_dim)

        # Learn distinct embeddings for each of the N track slots
        # Shape: [1, Num_Tracks, Embed_Dim]
        self.track_query_embed = nn.Parameter(torch.empty(1, num_tracks, embedding_dim))
        nn.init.xavier_uniform_(self.track_query_embed, gain=0.1)

        self.bootstrap_norm = nn.LayerNorm(embedding_dim)

    def load_weights(
        self, path: str, device: torch.device | None = None, strict: bool = True
    ):
        """
        Loads weights from a .pth file.

        Args:
            path: Path to the .pth file.
            device: Device to load the tensors onto (default: current model device).
            strict: Whether to strictly enforce that the keys in state_dict
                    match the keys returned by this module's state_dict().
        """
        import os

        if device is None:
            # Detect the device of the current model parameters
            device = next(self.parameters()).device

        if not os.path.exists(path):
            raise FileNotFoundError(f"Weight file not found at: {path}")

        print(f"Loading weights from {path} to {device}...")

        # map_location ensures we don't crash trying to load CUDA weights on CPU
        checkpoint = torch.load(path, map_location=device, weights_only=True)

        # Unwrap if it's a full training checkpoint (e.g., {'model': ..., 'optimizer': ...})
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                print(
                    "  > Detected full checkpoint dictionary. Extracting 'model_state_dict'..."
                )
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # Assume the dict itself is the state dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Load the state dict
        try:
            missing, unexpected = self.load_state_dict(state_dict, strict=strict)

            if len(missing) > 0:
                print(f"  [WARNING] Missing keys: {missing}")
            if len(unexpected) > 0:
                print(f"  [WARNING] Unexpected keys: {unexpected}")

            if len(missing) == 0 and len(unexpected) == 0:
                print("  > Success: All keys matched perfectly.")

        except RuntimeError as e:
            print(f"  [ERROR] Failed to load weights: {e}")
            raise e

        self.to(device)

    def forward(self, batch):
        # TODO gate queries based on prior_kin (maybe var too via BD distance?) with the tracks and detections
        features = batch["obs_features"]
        sensor_ids = batch["obs_ids"]
        mask = batch["obs_mask"]
        ownship = batch["ownship"]

        batch_size, seq_len, max_dets, _ = features.shape
        flat_feats = features.view(-1, features.shape[-1])
        flat_ids = sensor_ids.view(-1, 1)
        encoded_dets = self.detection_encoder(flat_feats, flat_ids)

        encoded_dets = encoded_dets.view(
            batch_size, seq_len, max_dets, self.embedding_dim
        )
        mask_expanded = mask.unsqueeze(-1).expand_as(encoded_dets)
        encoded_dets = encoded_dets * mask_expanded.float()

        encoded_dets_expanded = encoded_dets.repeat_interleave(self.num_tracks, dim=0)
        mask_expanded = mask.repeat_interleave(self.num_tracks, dim=0)

        # on first detection, "snap" to initial position
        init_queries = self.bootstrap_norm(self.track_query_embed)
        init_queries = torch.clamp(init_queries, min=-10.0, max=10.0)

        init_queries = init_queries.expand(batch_size, -1, -1)
        bootstrap_query = init_queries.reshape(
            batch_size * self.num_tracks, 1, self.embedding_dim
        )

        track_history = None

        all_prior_kinematics = []
        all_prior_variances = []
        all_posterior_kinematics = []
        all_posterior_variances = []
        all_association_scores = []
        all_posterior_track_active_logits = []

        for t in range(seq_len):
            current_context = encoded_dets_expanded[:, t, :, :]
            current_padding_mask = ~mask_expanded[:, t, :]
            current_ownship = ownship[:, t, :]  # [Batch, Ownship_Dim]
            current_ownship_expanded = current_ownship.repeat_interleave(
                self.num_tracks, dim=0
            )

            if track_history is None:
                # We skip the TemporalEncoder because there is no history yet.
                track_query = bootstrap_query
            else:
                track_query = self.temporal_encoder(
                    track_history, current_ownship_expanded
                )
                track_query = track_query.unsqueeze(1)

            prior_kin, prior_var, _ = self.prior_state_decoder(track_query)

            updated_embedding, assoc_score, _ = self.tdi_module(
                track_query, current_context, key_padding_mask=current_padding_mask
            )

            posterior_kin, posterior_var, posterior_track_active_logits = (
                self.posterior_state_decoder(updated_embedding)
            )

            all_prior_kinematics.append(prior_kin.squeeze(1))
            all_prior_variances.append(prior_var.squeeze(1))
            all_posterior_kinematics.append(posterior_kin.squeeze(1))
            all_posterior_variances.append(posterior_var.squeeze(1))
            all_posterior_track_active_logits.append(
                posterior_track_active_logits.squeeze(1)
            )
            all_association_scores.append(assoc_score.squeeze(1))

            if track_history is None:
                # Overwrite history with the first embedding.
                # History becomes: [State_0] (NOT [Init, State_0])
                track_history = updated_embedding
            else:
                track_history = torch.cat([track_history, updated_embedding], dim=1)

            if track_history.shape[1] > self.max_history_len:
                track_history = track_history[:, -self.max_history_len :, :]

        prior_kinematics_stacked = torch.stack(all_prior_kinematics, dim=1)
        prior_variances_stacked = torch.stack(all_prior_variances, dim=1)
        posterior_kinematics_stacked = torch.stack(all_posterior_kinematics, dim=1)
        posterior_variance_stacked = torch.stack(all_posterior_variances, dim=1)
        posterior_track_active_logits_stacked = torch.stack(
            all_posterior_track_active_logits, dim=1
        )
        scores_stacked = torch.stack(all_association_scores, dim=1)

        def unfold(tensor):
            _, s_len, dim = tensor.shape
            return tensor.view(batch_size, self.num_tracks, s_len, dim)

        return {
            "prior_kinematics": unfold(prior_kinematics_stacked),
            "prior_variance": unfold(prior_variances_stacked),
            "posterior_kinematics": unfold(posterior_kinematics_stacked),
            "posterior_variance": unfold(posterior_variance_stacked),
            "posterior_existence_logits": unfold(posterior_track_active_logits_stacked),
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
    OWNSHIP_DIM = 6

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
        ownship_dim=OWNSHIP_DIM,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker.to(device)
    print(f"Model created on device: {device}\n")

    # Random kinematic features: [Batch, Seq, Max_Dets, Feat_Dim]
    obs_features = torch.randn(BATCH_SIZE, SEQ_LEN, MAX_DETS, FEAT_DIM).to(device)

    # Random ownship features: [Batch, Seq, Feat_Dim]
    ownship = torch.randn(BATCH_SIZE, SEQ_LEN, OWNSHIP_DIM).to(device)

    # Random sensor IDs (0, 1, or 2): [Batch, Seq, Max_Dets, 1]
    obs_ids = torch.randint(0, NUM_SENSORS, (BATCH_SIZE, SEQ_LEN, MAX_DETS, 1)).to(
        device
    )

    # Random Mask: [Batch, Seq, Max_Dets]
    # Let's say ~30% are "False" (padding/empty slots), 70% are "True" (real detections)
    obs_mask = (torch.rand(BATCH_SIZE, SEQ_LEN, MAX_DETS) > 0.3).to(device)

    # Pack into dictionary
    batch = {
        "obs_features": obs_features,
        "obs_ids": obs_ids,
        "obs_mask": obs_mask,
        "ownship": ownship,
    }

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
