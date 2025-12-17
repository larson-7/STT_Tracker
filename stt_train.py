import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from models.stt import STTTracker
from stt_data_loader import TrackingDataset


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, batch_data in enumerate(loader):
        # Forward Pass
        # outputs["kinematics"]: [batch, num_tracks, seq_len, output_dim]
        # outputs["association_scores"]: [batch, num_tracks, seq_len, 1]
        outputs = model(batch_data)

        pred_states = outputs["kinematics"]
        pred_logits = outputs["association_scores"]

        # Ground Truth Data
        # gt_states is [batch, seq_len, max_gt_objects, output_dim]
        # Assuming truth_mask is [batch, seq_len, max_gt_objects] (1 if valid object, 0 if padding)
        gt_states = batch_data["truth_states"].to(device)
        gt_mask = batch_data["truth_mask"].to(device)

        batch_loss_cls = 0
        batch_loss_box = 0

        batch_size, num_queries, seq_len, _ = pred_logits.shape

        for b in range(batch_size):
            for t in range(seq_len):

                # Get predictions for this specific frame
                # p_state: [num_queries, obs_dim]
                # p_score: [num_queries, 1]
                p_state = pred_states[b, :, t]
                p_score = pred_logits[b, :, t]

                # Get valid Ground Truths for this frame
                # Filter out padding using the mask
                valid_indices = gt_mask[b, t] > 0
                g_box = gt_states[b, t][valid_indices]  # [num_actual_objs, obs_dim]

                if len(g_box) == 0:
                    # If no GT objects exist in this frame, all preds should be background (0)
                    target_cls = torch.zeros_like(p_score)
                    loss_cls = F.binary_cross_entropy_with_logits(p_score, target_cls)
                    batch_loss_cls += loss_cls
                    continue

                # Association Loss (L_d):
                # l_d = -(y * log(AS_i) + (1 - y) * log(1 - AS_i))

                # Compute Cost Matrix (on CPU for Scipy)
                # Cost = Classification Cost + Position Cost
                # We want to match preds that are confident AND close to the GT

                # L1 Distance Cost
                cost_dist = torch.cdist(p_state, g_box, p=1).detach().cpu()

                # Prob Cost:
                # Penalize if the model is confident but wrong
                # specific implementation depends on if p_score is sigmoid or raw logits
                out_prob = p_score.sigmoid().detach().cpu()
                cost_class = -out_prob.matmul(torch.ones(len(g_box), 1).T.cpu())

                # Final Cost Matrix
                C = cost_dist + cost_class

                # 4. Hungarian Matching
                # row_idx -> indices of predicted tracks
                # col_idx -> indices of ground truth objects
                row_idx, col_idx = linear_sum_assignment(C.numpy())

                # Build Classification Targets
                # Initialize all targets to 0 (Background)
                target_cls = torch.zeros_like(p_score)
                # Set matched tracks to 1 (Object)
                target_cls[row_idx] = 1.0

                # Calculate Losses
                # We want the matched indices to predict 1, everyone else 0
                loss_cls = F.binary_cross_entropy_with_logits(p_score, target_cls)

                # State Prediction Loss (L1) - Only for matched pairs
                matched_p_box = p_state[row_idx]
                matched_g_box = g_box[col_idx]
                loss_box = F.l1_loss(matched_p_box, matched_g_box)

                # Accumulate
                batch_loss_cls += loss_cls
                batch_loss_box += loss_box

        # Average loss over the batch/sequence
        total_steps = batch_size * seq_len
        final_loss = (batch_loss_cls + batch_loss_box * 5.0) / total_steps

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        total_loss += final_loss.item()

    return total_loss / len(loader)


# --- Configuration ---
if __name__ == "__main__":
    # Create dummy files for testing
    # In reality, point these to your CSVs
    TRACK_FILE = "data/train_tracks.csv"
    TRUTH_FILE = "data/train_truth.csv"
    OWN_FILE = "data/train_ownship.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init
    dataset = TrackingDataset(TRACK_FILE, TRUTH_FILE, OWN_FILE, seq_len=5)
    loader = DataLoader(dataset, batch_size=5)

    model = STTTracker(
        num_tracks=10,
        input_dim=9,
        sensor_type_embedding_dim=8,
        num_sensor_modalities=3,
        embedding_dim=256,
        max_history_len=100,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("Starting Training...")
    for epoch in range(10):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch}: Loss {loss:.4f}")
