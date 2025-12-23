import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from models.stt import STTTracker
from stt_data_loader import TrackingDataset
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, device, max_num_truth_entities):
    model.train()
    total_epoch_loss = 0
    
    # Gaussian NLL Loss to handle state pred + variance simultaneously
    nll_loss_fn = nn.GaussianNLLLoss(reduction='none')

    for batch_idx, batch_data in enumerate(loader):
        gt_prior_states = batch_data["prior_truth_states"].to(device)       # [B, T, MaxGT, Dim]
        gt_posterior_states = batch_data["posterior_truth_states"].to(device) # [B, T, MaxGT, Dim]
        truth_ids = batch_data["truth_ids"].to(device)                      # [B, T, MaxGT]
        
        outputs = model(batch_data)
        # Unpack outputs
        # Shapes: [Batch, Num_Tracks, Seq_Len, Dim]
        prior_pred_states = outputs["prior_kinematics"]
        prior_pred_var = outputs["prior_variance"]
        posterior_pred_states = outputs["posterior_kinematics"]
        posterior_pred_var = outputs["posterior_variance"]
        pred_assoc_logits = outputs["association_scores"] # [Batch, Num_Tracks, Seq_Len, Max_Dets]

        batch_size, num_track_slots, seq_len, _ = pred_assoc_logits.shape
        max_detects = pred_assoc_logits.shape[-1]

        # Initialize persistent assignments for this batch sequence
        # track_assignments[b, slot_i] = Real Entity ID (e.g., 101)
        # Initialize with -1 (no assignment)
        track_assignments = torch.full((batch_size, num_track_slots), -1, dtype=torch.long).to(device)

        batch_loss_assoc = 0.0
        batch_loss_state = 0.0
        valid_steps = 0 # Counter for normalization

        for t in range(seq_len):
            # Get the truth IDs present in this specific frame
            current_frame_truth_ids = truth_ids[:, t] # [Batch, MaxGT]

            # build the target for association and mask for state loss
            target_assoc_map = torch.zeros_like(pred_assoc_logits[:, :, t, :]) # [B, Slots, Dets]
            
            # Masks to ensure we only learn from active slots
            active_slot_mask = torch.zeros((batch_size, num_track_slots), device=device)
            
            # Temporary storage for gathering state targets
            # We want to align GT states to the Slots
            aligned_gt_prior = torch.zeros_like(prior_pred_states[:, :, t, :])
            aligned_gt_posterior = torch.zeros_like(posterior_pred_states[:, :, t, :])
            
            for b in range(batch_size):
                # Valid GT indices for this frame (ignore -1 padding in truth_ids)
                # Assume truth_ids and detections are 1:1 in index for this logic 
                # (i.e., truth_id[k] corresponds to detection[k])
                valid_gt_indices = (current_frame_truth_ids[b] != -1).nonzero().squeeze(-1)
                
                for gt_idx in valid_gt_indices:
                    tid = current_frame_truth_ids[b, gt_idx].item()
                    
                    # Check if a slot already owns this TID
                    slot_idx = (track_assignments[b] == tid).nonzero()
                    
                    if len(slot_idx) > 0:
                        slot_idx = slot_idx[0].item()
                    else:
                        # Assign to new slot if available
                        free_slots = (track_assignments[b] == -1).nonzero()
                        if len(free_slots) > 0:
                            slot_idx = free_slots[0].item()
                            track_assignments[b, slot_idx] = tid
                        else:
                            # No slots left, skip tracking this object
                            continue
                    
                    if gt_idx < max_detects:
                        target_assoc_map[b, slot_idx, gt_idx] = 1.0
                    
                    # Grab the GT state from the specific index where the ID lives
                    aligned_gt_prior[b, slot_idx] = gt_prior_states[b, t, gt_idx]
                    aligned_gt_posterior[b, slot_idx] = gt_posterior_states[b, t, gt_idx]
                    
                    # Mark this slot as active for loss calculation
                    active_slot_mask[b, slot_idx] = 1.0


            # Association Loss (Binary Cross Entropy)
            # compute over all slots/dets, but targets are 0 unless set above
            # pred_assoc_logits: [B, Slots, Dets]
            loss_assoc = F.binary_cross_entropy_with_logits(
                pred_assoc_logits[:, :, t, :], 
                target_assoc_map, 
                reduction='none'
            )
            batch_loss_assoc += loss_assoc.mean()

            # State Estimation Loss (Gaussian NLL)
            # We only calculate state loss for ACTIVE slots
            if active_slot_mask.sum() > 0:
                # Prior Loss (t-1)
                # NLL inputs: (input, target, var)
                l_prior = nll_loss_fn(
                    prior_pred_states[:, :, t, :], 
                    aligned_gt_prior, 
                    prior_pred_var[:, :, t, :]
                )
                # Mask out inactive slots (sum over dim, then mask)
                l_prior = (l_prior.mean(dim=-1) * active_slot_mask).sum() / active_slot_mask.sum()
                
                # Posterior Loss (t)
                l_posterior = nll_loss_fn(
                    posterior_pred_states[:, :, t, :], 
                    aligned_gt_posterior, 
                    posterior_pred_var[:, :, t, :]
                )
                l_posterior = (l_posterior.mean(dim=-1) * active_slot_mask).sum() / active_slot_mask.sum()

                batch_loss_state += (l_prior + l_posterior)
                valid_steps += 1

        # Normalize over sequence length
        if valid_steps > 0:
            batch_loss_state /= valid_steps
        
        batch_loss_assoc /= seq_len
        
        # Weights from the paper (Equation 6)
        # Gamma (assoc), Lambda (post), Alpha (prior)
        gamma, lam, alpha = 1.0, 5.0, 2.0 
        
        final_loss = (gamma * batch_loss_assoc) + (lam * batch_loss_state)

        optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_epoch_loss += final_loss.item()

    return total_epoch_loss / len(loader)


if __name__ == "__main__":
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
        device=device
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
