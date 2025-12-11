import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Positional encoding, null-detects, multi-modal, context length, casual mask?
# database of local tracks stored based on their local track id


def nll_loss(pred_state, pred_variance, target_state):
    """
    Used to guide log varaiance prediction.

    Gaussian NLL: -log p(y|μ,σ²) = 0.5 * (log(σ²) + (y-μ)²/σ²)
    """
    diff = target_state - pred_state
    loss = 0.5 * (torch.log(pred_variance) + (diff**2) / pred_variance)
    return loss.mean()


class DetectionEncoder(nn.Module):
    """
    Encodes raw detection measurements into a high-dimensional embedding space.
    Input: Raw detection vector (e.g., x, y, z, vx, vy, vz, ax, ay, az, sensor_type, quality).
    Output: Detection embedding.
    """

    def __init__(self, input_dim=8, embedding_dim=256):
        super().__init__()
        # Simple MLP as described: Linear -> ReLU -> Linear
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, raw_detections):
        # raw_detections shape: [batch_size, num_detections, input_dim]
        return self.mlp(raw_detections)


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
        # Cross-Attention: Query is Track, Key/Value are Detections
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

    def forward(self, track_query, context_detections):
        """
        track_query: [batch_size, 1, hidden_dim] (The active track)
        context_detections: [batch_size, num_context_dets, hidden_dim] (The candidates)
        """
        # Query = track_query, Key = context_detections, Value = context_detections
        attn_output, attn_weights = self.multihead_attn(
            query=track_query, key=context_detections, value=context_detections
        )

        # Residual connection + Norm
        # This 'updated_embedding' contains information from the attended detections
        updated_embedding = self.norm(track_query + attn_output)
        updated_embedding = updated_embedding + self.ffn(updated_embedding)

        # Predict Association Scores (Logits for binary classification/ranking)
        association_scores = self.association_head(updated_embedding)

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


# TODO Make STTTRacker here
# TODO Consider padding data with nulltoken somehow for missed detects

if __name__ == "__main__":
    # Hyperparameters
    HIDDEN_DIM = 256
    INPUT_DIM = 9  # (x, y, z, x_dot, y_dot, z_dot, type)
    STATE_DIM = 10  # (x, y, z, vx, vy, vz, ax, ay, az)

    # Instantiate Modules
    detection_encoder = DetectionEncoder(INPUT_DIM, HIDDEN_DIM)
    temporal_encoder = TemporalEncoder(HIDDEN_DIM)
    tdi_module = TrackDetectionInteraction(HIDDEN_DIM)
    state_decoder = TrackStateDecoder(HIDDEN_DIM, STATE_DIM)

    # Dummy Data
    # Batch of 1 track, with 9 frames of history
    track_history_raw = torch.randn(1, 9, INPUT_DIM)
    # Batch of 1 track, looking at 5 nearby context detections
    raw_context_detections = torch.randn(1, 5, INPUT_DIM)

    # Encode History
    # (In a real loop, these would already be embeddings, but we encode here for demo)
    history_embeddings = detection_encoder(track_history_raw)

    # Temporal Fusion -> Get Track Query
    track_query = temporal_encoder(history_embeddings)  # Shape: [1, 256]
    track_query = track_query.unsqueeze(1)  # Shape: [1, 1, 256] for Attention

    # Encode Current Context Detections
    context_detections = detection_encoder(raw_context_detections)  # Shape: [1, 5, 256]

    # TDI Interaction
    updated_embedding, association_scores = tdi_module(track_query, context_detections)

    # Decode State
    current_state, current_variance = state_decoder(updated_embedding)

    print(f"Track Query Shape: {track_query.shape}")
    print(f"Association Scores: {association_scores.item():.4f}")
    print(f"Predicted State (x, y, z, v...): {current_state.detach().numpy()}")
    print(f"Predicted Variance (x, y, z, v...): {current_variance.detach().numpy()}")
