import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # Tensorboard
from tqdm import tqdm # Progress Bar
import matplotlib.pyplot as plt # Plotting

# Your custom imports
from scipy.optimize import linear_sum_assignment
from models.stt import STTTracker
from stt_data_loader import TrackingDataset

# --- Helper Function for XY Plotting ---
def log_xy_plot(writer, epoch, gt_states, pred_states, batch_idx=0):
    """
    Generates an X-Y plot of Truth vs Prediction for the first element in the batch
    and logs it to TensorBoard.
    
    Assumes:
    - gt_states: [Batch, Seq, MaxGT, Dim]
    - pred_states: [Batch, Slots, Seq, Dim]
    - Dim 0 = X, Dim 1 = Y (Standard 2D tracking assumption)
    """
    # Take the first sample in the batch
    b = 0 
    
    # Detach and move to CPU for plotting
    gt = gt_states[b].detach().cpu().numpy()      # [Seq, MaxGT, Dim]
    pred = pred_states[b].detach().cpu().numpy()  # [Slots, Seq, Dim]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Epoch {epoch} - Batch {batch_idx} Tracking (X vs Y)")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True, alpha=0.3)

    # Plot Ground Truth (Green circles)
    # Iterate over time to plot the path of each GT entity
    seq_len, max_gt, _ = gt.shape
    for gt_idx in range(max_gt):
        # Filter out 0.0 or padding if necessary, 
        # distinct GTs often drift. Just plotting all non-zero paths roughly.
        path_x = gt[:, gt_idx, 0]
        path_y = gt[:, gt_idx, 1]
        
        # Simple check to avoid plotting empty padding rows if they are pure zeros
        if (path_x == 0).all() and (path_y == 0).all():
            continue
            
        ax.plot(path_x, path_y, 'g-o', markersize=4, alpha=0.5, label='Truth' if gt_idx == 0 else "")
        # Mark Start
        ax.text(path_x[0], path_y[0], 'S', color='green', fontsize=8)

    # Plot Predictions (Red crosses)
    num_slots, seq_len, _ = pred.shape
    for slot_idx in range(num_slots):
        path_x = pred[slot_idx, :, 0]
        path_y = pred[slot_idx, :, 1]
        
        # Heuristic: If track variance is massive or position is 0, model might consider it inactive.
        # For now, we plot everything to see what the model is "thinking".
        ax.plot(path_x, path_y, 'r-x', markersize=4, alpha=0.6, label='Pred' if slot_idx == 0 else "")

    # Deduplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Log to TensorBoard
    writer.add_figure("Tracking/XY_Plot", fig, global_step=epoch)
    plt.close(fig)


def train_one_epoch(model, loader, optimizer, device, max_num_truth_entities, writer, epoch_index):
    model.train()
    total_epoch_loss = 0
    
    # Gaussian NLL Loss
    nll_loss_fn = nn.GaussianNLLLoss(reduction='none')

    # Wrap loader with tqdm for progress bar
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch_index}")

    for batch_idx, batch_data in pbar:
        gt_prior_states = batch_data["prior_truth_states"].to(device)       
        gt_posterior_states = batch_data["posterior_truth_states"].to(device) 
        truth_ids = batch_data["truth_ids"].to(device)                      
        
        outputs = model(batch_data)
        
        prior_pred_states = outputs["prior_kinematics"]
        prior_pred_var = outputs["prior_variance"]
        posterior_pred_states = outputs["posterior_kinematics"]
        posterior_pred_var = outputs["posterior_variance"]
        pred_assoc_logits = outputs["association_scores"] 

        batch_size, num_track_slots, seq_len, _ = pred_assoc_logits.shape
        max_detects = pred_assoc_logits.shape[-1]

        track_assignments = torch.full((batch_size, num_track_slots), -1, dtype=torch.long).to(device)

        batch_loss_assoc = 0.0
        batch_loss_prior_state = 0.0
        batch_loss_posterior_state = 0.0
        valid_steps = 0 

        for t in range(seq_len):
            current_frame_truth_ids = truth_ids[:, t] 
            target_assoc_map = torch.zeros_like(pred_assoc_logits[:, :, t, :]) 
            active_slot_mask = torch.zeros((batch_size, num_track_slots), device=device)
            
            aligned_gt_prior = torch.zeros_like(prior_pred_states[:, :, t, :])
            aligned_gt_posterior = torch.zeros_like(posterior_pred_states[:, :, t, :])
            
            for b in range(batch_size):
                valid_gt_indices = (current_frame_truth_ids[b] != -1).nonzero().squeeze(-1)
                
                for gt_idx in valid_gt_indices:
                    tid = current_frame_truth_ids[b, gt_idx].item()
                    
                    # Logic to find or assign slot
                    slot_idx = (track_assignments[b] == tid).nonzero()
                    if len(slot_idx) > 0:
                        slot_idx = slot_idx[0].item()
                    else:
                        free_slots = (track_assignments[b] == -1).nonzero()
                        if len(free_slots) > 0:
                            slot_idx = free_slots[0].item()
                            track_assignments[b, slot_idx] = tid
                        else:
                            continue
                    
                    if gt_idx < max_detects:
                        target_assoc_map[b, slot_idx, gt_idx] = 1.0
                    
                    aligned_gt_prior[b, slot_idx] = gt_prior_states[b, t, gt_idx]
                    aligned_gt_posterior[b, slot_idx] = gt_posterior_states[b, t, gt_idx]
                    active_slot_mask[b, slot_idx] = 1.0

            # --- Association Loss ---
            loss_assoc = F.binary_cross_entropy_with_logits(
                pred_assoc_logits[:, :, t, :], 
                target_assoc_map, 
                reduction='none'
            )
            batch_loss_assoc += loss_assoc.mean()

            # --- State Loss ---
            if active_slot_mask.sum() > 0:
                l_prior = nll_loss_fn(prior_pred_states[:, :, t, :], aligned_gt_prior, prior_pred_var[:, :, t, :])
                l_prior = (l_prior.mean(dim=-1) * active_slot_mask).sum() / active_slot_mask.sum()
                
                l_posterior = nll_loss_fn(posterior_pred_states[:, :, t, :], aligned_gt_posterior, posterior_pred_var[:, :, t, :])
                l_posterior = (l_posterior.mean(dim=-1) * active_slot_mask).sum() / active_slot_mask.sum()

                batch_loss_prior_state += l_prior
                batch_loss_posterior_state += l_posterior
                valid_steps += 1

        if valid_steps > 0:
            batch_loss_prior_state /= valid_steps
            batch_loss_posterior_state /= valid_steps
        
        batch_loss_assoc /= seq_len
        
        gamma, lam, alpha = 1.0, 5.0, 2.0 
        final_loss = (gamma * batch_loss_assoc) + (lam * batch_loss_prior_state) + (alpha * batch_loss_posterior_state)

        optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_epoch_loss += final_loss.item()
        
        # --- Update Tensorboard & Tqdm ---
        # Calculate global step for continuous plotting across epochs
        global_step = epoch_index * len(loader) + batch_idx
        
        writer.add_scalar("Loss/Total", final_loss.item(), global_step)
        writer.add_scalar("Loss/Association", batch_loss_assoc.item(), global_step)
        writer.add_scalar("Loss/State", batch_loss_prior_state.item() if isinstance(batch_loss_prior_state, torch.Tensor) else 0, global_step)

        # Update tqdm bar text
        pbar.set_postfix({
            "Loss": f"{final_loss.item():.4f}", 
            "Assoc": f"{batch_loss_assoc.item():.4f}"
        })

        # --- XY Plotting (First batch only) ---
        if batch_idx == 0:
            log_xy_plot(writer, epoch_index, gt_posterior_states, posterior_pred_states)

    return total_epoch_loss / len(loader)


if __name__ == "__main__":
    TRACK_FILE = "data/train_tracks.csv"
    TRUTH_FILE = "data/train_truth.csv"
    OWN_FILE = "data/train_ownship.csv"

    # Define TensorBoard writer
    # Run in terminal: tensorboard --logdir=runs
    writer = SummaryWriter("runs/stt_experiment_1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_sensor_modalities = 3
    max_num_detects_per_step = num_sensor_modalities**2
    
    dataset = TrackingDataset(
        TRACK_FILE, TRUTH_FILE, OWN_FILE,
        seq_len=5,
        max_num_detects_per_step=max_num_detects_per_step,
        device=device
    )
    loader = DataLoader(dataset, batch_size=5, shuffle=True) # Added shuffle=True

    model = STTTracker(
        num_tracks=10,
        input_dim=9,
        sensor_type_embedding_dim=8,
        num_sensor_modalities=3,
        embedding_dim=256,
        max_history_len=100,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"Starting Training on {device}...")
    
    try:
        for epoch in range(10):
            loss = train_one_epoch(
                model, loader, optimizer, device, dataset.max_num_truth_entities, writer, epoch
            )
            # Tqdm handles inner loop prints, we just print summary here
            print(f"Epoch {epoch} Complete. Avg Loss: {loss:.4f}")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        writer.close()
        print("TensorBoard writer closed.")