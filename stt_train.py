import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from stt_data_loader import TrackingDataset
from models.stt import (
    STTTracker,
)
import torch.optim as optim
from scipy.optimize import linear_sum_assignment


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, batch_data in enumerate(loader):
        # A batch is a list of sequences (batch_size items)
        # Each item is a dict with 'observations', 'ground_truth', 'ownship' lists
        batch_size, seq_size, num_sensor_types = batch_data["truth_ids"].shape
        # Process sequence frame by frame
        outputs = model(batch_data)
        kinematics = outputs["kinematics"]
        variance = outputs["variance"]
        pred_logits = outputs["association_scores"]

        # Loss Calculation
        # Association Loss (L_d) = -(ylog(AS_i) + (1 - y)log(1 - AS_i))
        # where y is the ground truth association, 1 if associated or 0 if not
        target_cls = torch.zeros(batch_size, model.num_tracks, seq_size, 1).to(device)
        # TODO: truth_ids doesn't seem right, should be a tensor that is batch, seq, truth_id ... oh we can have mutliple measurements per timestep so maybe this is right
        target_cls[p_indices] = 1.0

        loss_cls = F.binary_cross_entropy_with_logits(
            pred_logits.squeeze(0), target_cls
        )
        # Need to match Predicted Tracks (M) to Ground Truth (N)
        truth_ids = batch_data["truth_ids"]
        gt_states = batch_data["truth_states"]

        # Cost Matrix: L1 distance between Pred State and GT State
        # pred_states: [batch_size, M, obs_dim], gt_states: [N, obs_dim]
        p_s = kinematics.squeeze(0)  # [M, obs_dim]
        cost_dist = torch.cdist(p_s, gt_states, p=1)  # [M, N]

        # Cost Class: Probability of "Not Object"
        # If we want tracks to persist, we reward matching specific IDs

        # Hungarian Match
        p_indices, g_indices = linear_sum_assignment(cost_dist.cpu().detach().numpy())

        # Association Loss (L_d)
        # Matched tracks should have score 1, others 0
        target_cls = torch.zeros(model.num_tracks, 1).to(device)
        target_cls[p_indices] = 1.0

        loss_cls = F.binary_cross_entropy_with_logits(
            pred_logits.squeeze(0), target_cls
        )

        # Calculate Losses
        loss_state = 0
        loss_cls = 0

        # Matches
        matched_preds = p_s[p_indices]
        matched_gts = gt_states[g_indices]
        loss_state = F.l1_loss(matched_preds, matched_gts)

        # Combined Loss
        step_loss = loss_state * 5.0 + loss_cls

        # Backprop (Truncated BPTT usually, but here we do per step or per seq)
        optimizer.zero_grad()
        step_loss.backward(retain_graph=True)  # Retain graph for sequence BPTT
        optimizer.step()

        total_loss += step_loss.item()

        # Update states/memory for next frame
        # Detach to prevent gradients exploding over long sequences unless using TBPTT
        curr_states = pred_states.detach()
        curr_embeds = output_embeds.detach()

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
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print("Starting Training...")
    for epoch in range(10):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch}: Loss {loss:.4f}")
