import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from models.stt import STTTracker
from stt_data_loader import TrackingDataset


def train_one_epoch(model, loader, optimizer, device, max_num_truth_entities):
    model.train()
    total_loss = 0

    for batch_idx, batch_data in enumerate(loader):
        # Forward Pass
        # outputs["kinematics"]: [batch, num_tracks, seq_len, output_dim]
        # outputs["association_scores"]: [batch, num_tracks, seq_len, 1]
        outputs = model(batch_data)

        pred_states = outputs["kinematics"]
        pred_assoc_logits = outputs["association_scores"]

        # Ground Truth Data
        # gt_states is [batch, seq_len, max_gt_objects, output_dim]
        # Assuming truth_mask is [batch, seq_len, max_gt_objects] (1 if valid object, 0 if padding)
        gt_prior_states = batch_data["prior_truth_states"].to(device)
        gt_postieror_states = batch_data["posterior_truth_states"].to(device)
        gt_mask = batch_data["truth_mask"].to(device)
        truth_ids = batch_data["truth_ids"].to(device)
        total_association_loss = 0.0

        batch_size, num_track_slots, seq_len, _ = pred_assoc_logits.shape
        # Initialize persistent assignments for the batch
        # track_assignments[b, slot_i] = entity truth id (e.g., 1)
        track_assignments = torch.full((batch_size, max_num_truth_entities), -1).to(
            device
        )

        for t in range(seq_len):
            # Get the truth IDs present in this frame's detections
            # truth_ids: [Batch, Seq, 9] -> current frame: [Batch, 9]
            frame_truth_ids = truth_ids[:, t]

            for b in range(batch_size):
                for det_idx, truth_id in enumerate(frame_truth_ids[b]):
                    if truth_id == -1:
                        continue  # Skip padding
                    # Check if we already have a slot tracking this Real ID
                    assigned_slot = (track_assignments[b] == truth_id).nonzero()

                    if len(assigned_slot) > 0:
                        # We found the slot (e.g., Slot 3) that owns this ID.
                        # Target: Slot 3 should attend to det_idx
                        pass
                    else:
                        # This ID has never been seen before. Assign it a free slot.
                        # (e.g., Slot 5 was empty, now Slot 5 owns this ID)
                        free_slots = (track_assignments[b] == -1).nonzero()
                        if len(free_slots) > 0:
                            first_free = free_slots[0]
                            track_assignments[b, first_free] = truth_id

            # Association Loss (L_d):
            # l_d = -(y * log(AS_i) + (1 - y) * log(1 - AS_i))
            # y = 1.0 if Slot i matched Det j, 0.0 otherwise.
            y = torch.zeros((num_track_slots, max_num_detects_per_step), device=device)

            # Mask for slots that are currently active (tracking something)
            # We only calculate loss for slots that have been assigned an ID.
            active_track_mask = torch.zeros((num_track_slots), device=device)

            for slot_idx, det_idx in track_assignments:
                y[slot_idx, det_idx] = 1.0
                active_track_mask[slot_idx] = 1.0

            # Preds: [Num_Tracks, Max_Dets]
            preds = pred_assoc_logits[b, :, t, :]

            loss = F.binary_cross_entropy_with_logits(preds, y, reduction="none")
            total_association_loss += loss.mean()

            # TODO: loss objective for variance, either mahalanobis or KL-Div to minimize distribution differences?

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
    num_sensor_modalities = 3
    max_num_detects_per_step = (
        num_sensor_modalities**2
    )  # n^2 sensor modalities each sensor type can detect all n objects,
    dataset = TrackingDataset(
        TRACK_FILE,
        TRUTH_FILE,
        OWN_FILE,
        seq_len=5,
        max_num_detects_per_step=max_num_detects_per_step,
    )
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
        loss = train_one_epoch(
            model, loader, optimizer, device, dataset.max_num_truth_entities
        )
        print(f"Epoch {epoch}: Loss {loss:.4f}")
