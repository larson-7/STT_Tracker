import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # Tensorboard
from tqdm import tqdm  # Progress Bar
import matplotlib.pyplot as plt  # Plotting
import matplotlib.cm as cm  # Colormaps
import numpy as np
import os

# Your custom imports
from scipy.optimize import linear_sum_assignment
from models.stt import STTTracker
from stt_data_loader import TrackingDataset


def log_xy_plot(
    writer, epoch, batch_data, pred_states, batch_idx=0, tag="Tracking/XY_Plot"
):
    b = batch_idx
    gt = batch_data["posterior_truth_states"][b].detach().cpu().numpy()
    detections = batch_data["obs_features"][b].detach().cpu().numpy()
    obs_ids = batch_data["obs_ids"][b].detach().cpu().numpy()
    obs_mask = batch_data["obs_mask"][b].detach().cpu().numpy()
    pred = pred_states[b].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Epoch {epoch} - {tag}")
    ax.grid(True, alpha=0.3)

    COLOR_TRUTH = "dimgray"
    COLOR_PRED = "tab:orange"
    cmap = matplotlib.colormaps["tab10"]

    # --- Plot Ground Truth ---
    _, max_gt, _ = gt.shape
    for gt_idx in range(max_gt):
        path_x, path_y = gt[:, gt_idx, 0], gt[:, gt_idx, 1]
        if (path_x == 0).all() and (path_y == 0).all():
            continue

        ax.plot(
            path_x,
            path_y,
            color=COLOR_TRUTH,
            marker="o",
            markersize=3,
            alpha=0.4,
            label="Truth" if gt_idx == 0 else "",
        )

    # --- Plot Predictions ---
    num_slots, _, _ = pred.shape
    for slot_idx in range(num_slots):
        path_x, path_y = pred[slot_idx, :, 0], pred[slot_idx, :, 1]
        if (abs(path_x) < 1e-3).all() and (abs(path_y) < 1e-3).all():
            continue

        ax.plot(
            path_x,
            path_y,
            color=COLOR_PRED,
            marker=">",
            markersize=4,
            alpha=0.8,
            linestyle="None",
            label="Prediction" if slot_idx == 0 else "",
        )

    # --- Plot Detections (Conditional Markers) ---
    seq_len, max_det, _ = detections.shape
    plotted_labels = set()

    for t in range(seq_len):
        for d in range(max_det):
            if not obs_mask[t, d]:
                continue

            x, y = detections[t, d, 0], detections[t, d, 1]
            oid = int(obs_ids[t, d])

            # Star if > 0, 'x' if <= 0
            marker_type = "*" if oid >= 0 else "x"
            label = "det" if oid >= 0 else "clutter"
            if x != 0 or y != 0:
                color = cmap(oid % 10)
                label = f"ObsID:{oid}-{label}"

                ax.plot(
                    x,
                    y,
                    marker=marker_type,
                    markeredgecolor=color,
                    markerfacecolor="none" if marker_type == "*" else color,
                    markersize=7 if marker_type == "*" else 5,
                    alpha=0.7,
                    linestyle="None",
                    label=label if label not in plotted_labels else "",
                )
                plotted_labels.add(label)

    # Clean up Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    sorted_keys = sorted(by_label.keys())
    ax.legend(
        [by_label[k] for k in sorted_keys],
        sorted_keys,
        loc="upper right",
        bbox_to_anchor=(1.15, 1),
    )

    writer.add_figure(tag, fig, global_step=epoch)
    plt.close(fig)


def train_one_epoch(
    model, loader, optimizer, device, max_num_truth_entities, writer, epoch_index
):
    model.train()
    total_epoch_loss = 0
    l1_loss_fn = nn.L1Loss()

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train Ep {epoch_index}")

    for batch_idx, batch_data in pbar:

        gt_prior_states = batch_data["prior_truth_states"].to(device)
        gt_posterior_states = batch_data["posterior_truth_states"].to(device)
        truth_ids = batch_data["truth_ids"].to(device)

        # forward pass
        outputs = model(batch_data)

        prior_pred_states = outputs["prior_kinematics"]
        posterior_pred_states = outputs["posterior_kinematics"]
        pred_assoc_logits = outputs["association_scores"]

        batch_size, num_track_slots, seq_len, _ = pred_assoc_logits.shape
        max_detects = pred_assoc_logits.shape[-1]

        track_assignments = torch.full(
            (batch_size, num_track_slots), -1, dtype=torch.long
        ).to(device)

        batch_loss_assoc = 0.0
        batch_loss_prior_state = 0.0
        batch_loss_posterior_state = 0.0
        metric_post_mae = 0.0
        valid_steps = 0

        for t in range(seq_len):
            current_frame_truth_ids = truth_ids[:, t]
            target_assoc_map = torch.zeros_like(pred_assoc_logits[:, :, t, :])
            active_slot_mask = torch.zeros((batch_size, num_track_slots), device=device)

            aligned_gt_prior = torch.zeros_like(prior_pred_states[:, :, t, :])
            aligned_gt_posterior = torch.zeros_like(posterior_pred_states[:, :, t, :])

            for b in range(batch_size):
                # only evaluate loss on indicies that are valid, create gt_id to track slot map
                valid_gt_indices = (
                    (current_frame_truth_ids[b] != -1).nonzero().squeeze(-1)
                )
                for gt_idx in valid_gt_indices:
                    tid = current_frame_truth_ids[b, gt_idx].item()
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
                    aligned_gt_posterior[b, slot_idx] = gt_posterior_states[
                        b, t, gt_idx
                    ]
                    active_slot_mask[b, slot_idx] = 1.0

            loss_assoc = F.binary_cross_entropy_with_logits(
                pred_assoc_logits[:, :, t, :], target_assoc_map, reduction="none"
            )
            batch_loss_assoc += loss_assoc.mean()

            if active_slot_mask.sum() > 0:
                l_prior = l1_loss_fn(prior_pred_states[:, :, t, :], aligned_gt_prior)
                l_prior = (
                    l_prior.mean(dim=-1) * active_slot_mask
                ).sum() / active_slot_mask.sum()

                l_posterior = l1_loss_fn(
                    posterior_pred_states[:, :, t, :], aligned_gt_posterior
                )
                l_posterior = (
                    l_posterior.mean(dim=-1) * active_slot_mask
                ).sum() / active_slot_mask.sum()

                batch_loss_prior_state += l_prior
                batch_loss_posterior_state += l_posterior

                mae_post = F.l1_loss(
                    posterior_pred_states[:, :, t, :],
                    aligned_gt_posterior,
                    reduction="none",
                ).mean(dim=-1)
                metric_post_mae += (
                    mae_post * active_slot_mask
                ).sum() / active_slot_mask.sum()

                valid_steps += 1

        if valid_steps > 0:
            batch_loss_prior_state /= valid_steps
            batch_loss_posterior_state /= valid_steps
            metric_post_mae /= valid_steps

        batch_loss_assoc /= seq_len

        gamma, lam, alpha = 1.0, 5.0, 2.0
        final_loss = (
            (gamma * batch_loss_assoc)
            + (lam * batch_loss_prior_state)
            + (alpha * batch_loss_posterior_state)
        )

        optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_epoch_loss += final_loss.item()

        global_step = epoch_index * len(loader) + batch_idx
        writer.add_scalar("Train/Loss_Total", final_loss.item(), global_step)
        writer.add_scalar(
            "Train/MAE_Posterior",
            metric_post_mae.item() if isinstance(metric_post_mae, torch.Tensor) else 0,
            global_step,
        )
        pbar.set_postfix(
            {"Loss": f"{final_loss.item():.2f}", "MAE": f"{metric_post_mae:.2f}"}
        )

        if batch_idx == 0:
            log_xy_plot(
                writer,
                epoch_index,
                batch_data,
                posterior_pred_states,
                tag="Train/XY_Plot",
            )

    return total_epoch_loss / len(loader)


def validate_one_epoch(model, loader, device, writer, epoch_index):
    model.eval()
    total_epoch_loss = 0
    l1_loss_fn = nn.L1Loss()
    total_mae_post = 0.0

    with torch.no_grad():
        pbar = tqdm(
            enumerate(loader), total=len(loader), desc=f"Valid Ep {epoch_index}"
        )
        for batch_idx, batch_data in pbar:
            gt_prior_states = batch_data["prior_truth_states"].to(device)
            gt_posterior_states = batch_data["posterior_truth_states"].to(device)
            truth_ids = batch_data["truth_ids"].to(device)

            outputs = model(batch_data)

            prior_pred_states = outputs["prior_kinematics"]
            posterior_pred_states = outputs["posterior_kinematics"]
            pred_assoc_logits = outputs["association_scores"]

            batch_size, num_track_slots, seq_len, _ = pred_assoc_logits.shape
            max_detects = pred_assoc_logits.shape[-1]

            track_assignments = torch.full(
                (batch_size, num_track_slots), -1, dtype=torch.long
            ).to(device)

            batch_loss_assoc = 0.0
            batch_loss_prior_state = 0.0
            batch_loss_posterior_state = 0.0
            metric_post_mae = 0.0
            valid_steps = 0

            for t in range(seq_len):
                current_frame_truth_ids = truth_ids[:, t]
                target_assoc_map = torch.zeros_like(pred_assoc_logits[:, :, t, :])
                active_slot_mask = torch.zeros(
                    (batch_size, num_track_slots), device=device
                )

                aligned_gt_prior = torch.zeros_like(prior_pred_states[:, :, t, :])
                aligned_gt_posterior = torch.zeros_like(
                    posterior_pred_states[:, :, t, :]
                )

                for b in range(batch_size):
                    valid_gt_indices = (
                        (current_frame_truth_ids[b] != -1).nonzero().squeeze(-1)
                    )
                    for gt_idx in valid_gt_indices:
                        tid = current_frame_truth_ids[b, gt_idx].item()
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
                        aligned_gt_posterior[b, slot_idx] = gt_posterior_states[
                            b, t, gt_idx
                        ]
                        active_slot_mask[b, slot_idx] = 1.0

                loss_assoc = F.binary_cross_entropy_with_logits(
                    pred_assoc_logits[:, :, t, :], target_assoc_map, reduction="none"
                )
                batch_loss_assoc += loss_assoc.mean()

                if active_slot_mask.sum() > 0:
                    l_prior = l1_loss_fn(
                        prior_pred_states[:, :, t, :], aligned_gt_prior
                    )
                    l_prior = (
                        l_prior.mean(dim=-1) * active_slot_mask
                    ).sum() / active_slot_mask.sum()

                    l_posterior = l1_loss_fn(
                        posterior_pred_states[:, :, t, :], aligned_gt_posterior
                    )
                    l_posterior = (
                        l_posterior.mean(dim=-1) * active_slot_mask
                    ).sum() / active_slot_mask.sum()

                    batch_loss_prior_state += l_prior
                    batch_loss_posterior_state += l_posterior

                    mae_post = F.l1_loss(
                        posterior_pred_states[:, :, t, :],
                        aligned_gt_posterior,
                        reduction="none",
                    ).mean(dim=-1)
                    metric_post_mae += (
                        mae_post * active_slot_mask
                    ).sum() / active_slot_mask.sum()
                    valid_steps += 1

            if valid_steps > 0:
                batch_loss_prior_state /= valid_steps
                batch_loss_posterior_state /= valid_steps
                metric_post_mae /= valid_steps

            batch_loss_assoc /= seq_len
            gamma, lam, alpha = 1.0, 5.0, 2.0
            final_loss = (
                (gamma * batch_loss_assoc)
                + (lam * batch_loss_prior_state)
                + (alpha * batch_loss_posterior_state)
            )
            total_epoch_loss += final_loss.item()
            total_mae_post += (
                metric_post_mae.item()
                if isinstance(metric_post_mae, torch.Tensor)
                else 0
            )

            if batch_idx == 0:
                log_xy_plot(
                    writer,
                    epoch_index,
                    batch_data,
                    posterior_pred_states,
                    tag="Validation/XY_Plot",
                )

    avg_loss = total_epoch_loss / len(loader)
    avg_mae = total_mae_post / len(loader)
    writer.add_scalar("Validation/Loss_Total", avg_loss, epoch_index)
    writer.add_scalar("Validation/MAE_Posterior", avg_mae, epoch_index)

    return avg_loss


if __name__ == "__main__":
    TRAIN_TRACKS = "data/train_tracks.csv"
    TRAIN_TRUTH = "data/train_truth.csv"
    TRAIN_OWN = "data/train_ownship.csv"
    VAL_TRACKS = "data/val_tracks.csv"
    VAL_TRUTH = "data/val_truth.csv"
    VAL_OWN = "data/val_ownship.csv"
    BEST_MODEL_PATH = "best_stt_model.pth"
    LOG_DIR = "runs/stt_experiment_colored_detections"

    BATCH_SIZE = 20
    MAX_SEQUENCE_LENGTH = 60
    NUM_SENSOR_MODALITIES = 1
    MAX_NUM_DETECTS_PER_STEP = NUM_SENSOR_MODALITIES**2
    MAX_NUM_TRACK_SLOTS = 1
    DETECT_NUM_DIMS = 9
    SENSOR_TYPE_EMBEDDING_DIM = 8
    EMBEDDING_DIM = 256

    writer = SummaryWriter(LOG_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TrackingDataset(
        TRAIN_TRACKS,
        TRAIN_TRUTH,
        TRAIN_OWN,
        seq_len=MAX_SEQUENCE_LENGTH,
        max_num_detects_per_step=MAX_NUM_DETECTS_PER_STEP,
        device=device,
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TrackingDataset(
        VAL_TRACKS,
        VAL_TRUTH,
        VAL_OWN,
        seq_len=MAX_SEQUENCE_LENGTH,
        max_num_detects_per_step=MAX_NUM_DETECTS_PER_STEP,
        device=device,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    model = STTTracker(
        num_tracks=MAX_NUM_TRACK_SLOTS,
        input_dim=DETECT_NUM_DIMS,
        sensor_type_embedding_dim=SENSOR_TYPE_EMBEDDING_DIM,
        num_sensor_modalities=NUM_SENSOR_MODALITIES,
        embedding_dim=EMBEDDING_DIM,
        max_history_len=MAX_SEQUENCE_LENGTH,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"Starting Training on {device}...")
    best_val_loss = float("inf")

    try:
        for epoch in range(200):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                train_dataset.max_num_truth_entities,
                writer,
                epoch,
            )
            val_loss = validate_one_epoch(model, val_loader, device, writer, epoch)
            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"  >>> New Best Model Saved (Val Loss: {val_loss:.4f})")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        writer.close()
        print("TensorBoard writer closed.")
