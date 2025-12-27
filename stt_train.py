import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Your custom imports
from scipy.optimize import linear_sum_assignment
from models.stt import STTTracker
from stt_data_loader import TrackingDataset


def get_curriculum_weights(epoch, device):
    """
    Returns a weight tensor [9] for (x,y,z, vx,vy,vz, ax,ay,az).

    Curriculum Strategy:
    - Epoch 0-50:   Focus ONLY on Position (Vel/Acc weights are negligible).
    - Epoch 50-100: Ramp up Velocity from 0.0 to 1.0.
    - Epoch 100+:   Ramp up Acceleration from 0.0 to 0.5.
    """
    # Base weights
    pos_w = 1.0

    vel_start = 50
    vel_ramp_end = 100

    if epoch < vel_start:
        # Use a tiny epsilon (1e-4) instead of 0.0 to keep the gradient heads alive
        # without affecting the loss significantly.
        vel_w = 1e-4
    else:
        # Ramp from 0.0 to 1.0
        progress = (epoch - vel_start) / (vel_ramp_end - vel_start)
        vel_w = np.clip(progress, 0.0, 1.0)

    acc_start = 100
    acc_ramp_end = 150

    if epoch < acc_start:
        acc_w = 0.0
    else:
        # Ramp from 0.0 to 0.5
        progress = (epoch - acc_start) / (acc_ramp_end - acc_start)
        acc_w = np.clip(progress * 0.5, 0.0, 0.5)

    weights = torch.tensor(
        [
            pos_w,
            pos_w,
            pos_w,  # X, Y, Z
            vel_w,
            vel_w,
            vel_w,  # Vx, Vy, Vz
            acc_w,
            acc_w,
            acc_w,  # Ax, Ay, Az
        ],
        device=device,
        dtype=torch.float32,
    )
    return weights


def log_xy_plot(
    writer,
    epoch,
    batch_data,
    pred_states,
    dataset_stats,
    batch_idx=0,
    tag="Tracking/XY_Plot",
):
    """
    Logs an XY plot of truth, predictions, and detections.
    Un-normalizes data using dataset_stats before plotting so axes are in Meters.
    """
    b = batch_idx
    valid_seq_mask = batch_data["valid_seq_mask"][b].detach().cpu().numpy()

    # Get Normalization Stats
    mu_x, std_x = dataset_stats["x"]["mean"], dataset_stats["x"]["std"]
    mu_y, std_y = dataset_stats["y"]["mean"], dataset_stats["y"]["std"]

    def unnorm(arr_x, arr_y):
        return (arr_x * std_x + mu_x), (arr_y * std_y + mu_y)

    # Ground Truth
    gt_raw = (
        batch_data["posterior_truth_states"][b].detach().cpu().numpy()[valid_seq_mask]
    )
    gt_x, gt_y = unnorm(gt_raw[:, :, 0], gt_raw[:, :, 1])

    # Detections
    det_raw = batch_data["obs_features"][b].detach().cpu().numpy()[valid_seq_mask]
    det_x, det_y = unnorm(det_raw[:, :, 0], det_raw[:, :, 1])

    obs_ids = batch_data["obs_ids"][b].detach().cpu().numpy()[valid_seq_mask]
    obs_mask = batch_data["obs_mask"][b].detach().cpu().numpy()[valid_seq_mask]
    truth_ids = batch_data["truth_ids"][b].detach().cpu().numpy()[valid_seq_mask]

    # Predictions
    pred_raw = pred_states[b].detach().cpu().numpy()[:, valid_seq_mask, :]
    pred_x, pred_y = unnorm(pred_raw[:, :, 0], pred_raw[:, :, 1])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Epoch {epoch} - {tag} (Units: Meters)")
    ax.grid(True, alpha=0.3)
    cmap = matplotlib.colormaps["tab10"]

    # Plot Ground Truth
    _, max_gt, _ = gt_raw.shape
    for gt_idx in range(max_gt):
        px, py = gt_x[:, gt_idx], gt_y[:, gt_idx]
        if (gt_raw[:, gt_idx, 0] == 0).all() and (gt_raw[:, gt_idx, 1] == 0).all():
            continue
        ax.plot(
            px,
            py,
            color=cmap(0),
            marker="o",
            markersize=3,
            alpha=0.4,
            label="Truth" if gt_idx == 0 else "",
        )

    # Plot Predictions
    num_slots, seq_len, _ = pred_raw.shape
    prediction_color = cmap(1)
    for slot_idx in range(num_slots):
        px, py = pred_x[slot_idx, :], pred_y[slot_idx, :]
        if (np.abs(pred_raw[slot_idx, :, 0]) < 1e-2).all():
            continue

        ax.plot(
            px,
            py,
            color=prediction_color,
            marker=">",
            markersize=4,
            alpha=0.8,
            linestyle="None",
            label="Prediction" if slot_idx == 0 else "",
        )
        for t in range(seq_len):
            ax.text(
                px[t],
                py[t],
                str(t),
                fontsize=8,
                color="black",
                ha="left",
                va="bottom",
                alpha=0.8,
            )

    # Plot Detections
    seq_len, max_det, _ = det_raw.shape
    plotted_labels = set()
    for t in range(seq_len):
        for d in range(max_det):
            if not obs_mask[t, d]:
                continue
            x, y = det_x[t, d], det_y[t, d]
            oid = int(obs_ids[t, d])
            truth_id = int(truth_ids[t, d])
            marker_type = "*" if truth_id >= 0 else "x"
            label_suffix = "det" if truth_id >= 0 else "clutter"

            if det_raw[t, d, 0] != 0 or det_raw[t, d, 1] != 0:
                color = cmap(oid % 10 + 2)
                label = f"Obs{oid}:{label_suffix}"
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
    model,
    loader,
    optimizer,
    device,
    max_num_truth_entities,
    writer,
    epoch_index,
    dataset_stats,
):
    model.train()
    total_epoch_loss = 0
    l1_loss_fn = nn.L1Loss(reduction="none")
    loss_weights = get_curriculum_weights(epoch_index, device)

    if epoch_index % 5 == 0:
        writer.add_scalar("Weights/Position", loss_weights[0], epoch_index)
        writer.add_scalar("Weights/Velocity", loss_weights[3], epoch_index)
        writer.add_scalar("Weights/Accel", loss_weights[6], epoch_index)

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train Ep {epoch_index}")

    for batch_idx, batch_data in pbar:
        gt_prior_states = batch_data["prior_truth_states"].to(device)
        gt_posterior_states = batch_data["posterior_truth_states"].to(device)
        truth_ids = batch_data["truth_ids"].to(device)
        valid_mask = batch_data["valid_seq_mask"].to(device)

        outputs = model(batch_data)

        prior_pred_states = outputs["prior_kinematics"]
        posterior_pred_states = outputs["posterior_kinematics"]
        pred_assoc_logits = outputs["association_scores"]

        batch_size, num_track_slots, seq_len, _ = pred_assoc_logits.shape
        max_detects = pred_assoc_logits.shape[-1]
        track_assignments = torch.full(
            (batch_size, num_track_slots), -1, dtype=torch.long
        ).to(device)

        batch_loss_assoc_sum = 0.0
        batch_loss_prior_sum = 0.0
        batch_loss_posterior_sum = 0.0
        batch_metric_post_mae_sum = 0.0
        batch_metric_prior_mae_sum = 0.0
        batch_metric_assoc_acc_sum = 0.0

        total_valid_assoc_steps = 1e-6
        total_active_slot_steps = 1e-6

        for t in range(seq_len):
            current_frame_truth_ids = truth_ids[:, t]
            is_step_valid = valid_mask[:, t] > 0
            target_assoc_map = torch.zeros_like(pred_assoc_logits[:, :, t, :])
            active_slot_mask = torch.zeros((batch_size, num_track_slots), device=device)
            aligned_gt_prior = torch.zeros_like(prior_pred_states[:, :, t, :])
            aligned_gt_posterior = torch.zeros_like(posterior_pred_states[:, :, t, :])

            for b in range(batch_size):
                if not is_step_valid[b]:
                    continue
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

            active_slot_mask = active_slot_mask * is_step_valid.unsqueeze(-1).float()

            loss_assoc = F.binary_cross_entropy_with_logits(
                pred_assoc_logits[:, :, t, :], target_assoc_map, reduction="none"
            )
            batch_loss_assoc_sum += (
                loss_assoc.mean(dim=(1, 2)) * is_step_valid.float()
            ).sum()

            pred_assoc_binary = (pred_assoc_logits[:, :, t, :] > 0.0).float()
            acc_assoc = (pred_assoc_binary == target_assoc_map).float().mean(dim=(1, 2))
            batch_metric_assoc_acc_sum += (acc_assoc * is_step_valid.float()).sum()

            total_valid_assoc_steps += is_step_valid.sum()

            num_active = active_slot_mask.sum()
            if num_active > 0:
                raw_prior = l1_loss_fn(prior_pred_states[:, :, t, :], aligned_gt_prior)
                raw_post = l1_loss_fn(
                    posterior_pred_states[:, :, t, :], aligned_gt_posterior
                )

                weighted_prior = raw_prior * loss_weights.view(1, 1, -1)
                weighted_post = raw_post * loss_weights.view(1, 1, -1)

                batch_loss_prior_sum += (
                    weighted_prior.sum(dim=-1) * active_slot_mask
                ).sum()
                batch_loss_posterior_sum += (
                    weighted_post.sum(dim=-1) * active_slot_mask
                ).sum()

                batch_metric_prior_mae_sum += (
                    raw_prior.mean(dim=-1) * active_slot_mask
                ).sum()
                batch_metric_post_mae_sum += (
                    raw_post.mean(dim=-1) * active_slot_mask
                ).sum()

                total_active_slot_steps += num_active

        final_assoc_loss = batch_loss_assoc_sum / total_valid_assoc_steps
        final_assoc_acc = batch_metric_assoc_acc_sum / total_valid_assoc_steps

        final_prior_loss = batch_loss_prior_sum / total_active_slot_steps
        final_prior_mae = batch_metric_prior_mae_sum / total_active_slot_steps

        final_posterior_loss = batch_loss_posterior_sum / total_active_slot_steps
        final_mae = batch_metric_post_mae_sum / total_active_slot_steps

        gamma, lam, alpha = 1.0, 5.0, 2.0
        final_loss = (
            (gamma * final_assoc_loss)
            + (lam * final_prior_loss)
            + (alpha * final_posterior_loss)
        )

        optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_epoch_loss += final_loss.item()
        global_step = epoch_index * len(loader) + batch_idx

        writer.add_scalar("Train/Loss_Total", final_loss.item(), global_step)
        writer.add_scalar("Train/Loss_Assoc", final_assoc_loss.item(), global_step)
        writer.add_scalar("Train/Acc_Assoc", final_assoc_acc.item(), global_step)
        writer.add_scalar("Train/Loss_Prior", final_prior_loss.item(), global_step)
        writer.add_scalar("Train/MAE_Prior", final_prior_mae.item(), global_step)
        writer.add_scalar(
            "Train/Loss_Posterior", final_posterior_loss.item(), global_step
        )
        writer.add_scalar("Train/MAE_Posterior", final_mae.item(), global_step)

        pbar.set_postfix(
            {"Loss": f"{final_loss.item():.2f}", "MAE": f"{final_mae.item():.2f}"}
        )

        if batch_idx == 0:
            log_xy_plot(
                writer,
                epoch_index,
                batch_data,
                posterior_pred_states,
                dataset_stats,
                tag="Train/XY_Plot",
            )

    return total_epoch_loss / len(loader)


def validate_one_epoch(model, loader, device, writer, epoch_index, dataset_stats):
    model.eval()
    total_epoch_loss = 0
    l1_loss_fn = nn.L1Loss(reduction="none")
    loss_weights = get_curriculum_weights(epoch_index, device)

    # Accumulators for Epoch Averages
    avg_assoc_loss = 0.0
    avg_assoc_acc = 0.0
    avg_prior_loss = 0.0
    avg_prior_mae = 0.0
    avg_post_loss = 0.0
    avg_post_mae = 0.0

    with torch.no_grad():
        pbar = tqdm(
            enumerate(loader), total=len(loader), desc=f"Valid Ep {epoch_index}"
        )
        for batch_idx, batch_data in pbar:
            gt_prior_states = batch_data["prior_truth_states"].to(device)
            gt_posterior_states = batch_data["posterior_truth_states"].to(device)
            truth_ids = batch_data["truth_ids"].to(device)
            valid_mask = batch_data["valid_seq_mask"].to(device)

            outputs = model(batch_data)

            prior_pred_states = outputs["prior_kinematics"]
            posterior_pred_states = outputs["posterior_kinematics"]
            pred_assoc_logits = outputs["association_scores"]

            batch_size, num_track_slots, seq_len, _ = pred_assoc_logits.shape
            max_detects = pred_assoc_logits.shape[-1]
            track_assignments = torch.full(
                (batch_size, num_track_slots), -1, dtype=torch.long
            ).to(device)

            batch_loss_assoc_sum = 0.0
            batch_metric_assoc_acc_sum = 0.0
            batch_loss_prior_sum = 0.0
            batch_metric_prior_mae_sum = 0.0
            batch_loss_posterior_sum = 0.0
            batch_metric_post_mae_sum = 0.0

            total_valid_assoc_steps = 1e-6
            total_active_slot_steps = 1e-6

            for t in range(seq_len):
                current_frame_truth_ids = truth_ids[:, t]
                is_step_valid = valid_mask[:, t] > 0
                target_assoc_map = torch.zeros_like(pred_assoc_logits[:, :, t, :])
                active_slot_mask = torch.zeros(
                    (batch_size, num_track_slots), device=device
                )
                aligned_gt_prior = torch.zeros_like(prior_pred_states[:, :, t, :])
                aligned_gt_posterior = torch.zeros_like(
                    posterior_pred_states[:, :, t, :]
                )

                for b in range(batch_size):
                    if not is_step_valid[b]:
                        continue
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

                active_slot_mask = (
                    active_slot_mask * is_step_valid.unsqueeze(-1).float()
                )

                loss_assoc = F.binary_cross_entropy_with_logits(
                    pred_assoc_logits[:, :, t, :], target_assoc_map, reduction="none"
                )
                batch_loss_assoc_sum += (
                    loss_assoc.mean(dim=(1, 2)) * is_step_valid.float()
                ).sum()

                pred_assoc_binary = (pred_assoc_logits[:, :, t, :] > 0.0).float()
                acc_assoc = (
                    (pred_assoc_binary == target_assoc_map).float().mean(dim=(1, 2))
                )
                batch_metric_assoc_acc_sum += (acc_assoc * is_step_valid.float()).sum()

                total_valid_assoc_steps += is_step_valid.sum()

                num_active = active_slot_mask.sum()
                if num_active > 0:
                    raw_prior = l1_loss_fn(
                        prior_pred_states[:, :, t, :], aligned_gt_prior
                    )
                    raw_post = l1_loss_fn(
                        posterior_pred_states[:, :, t, :], aligned_gt_posterior
                    )

                    weighted_prior = raw_prior * loss_weights.view(1, 1, -1)
                    weighted_post = raw_post * loss_weights.view(1, 1, -1)

                    batch_loss_prior_sum += (
                        weighted_prior.sum(dim=-1) * active_slot_mask
                    ).sum()
                    batch_loss_posterior_sum += (
                        weighted_post.sum(dim=-1) * active_slot_mask
                    ).sum()

                    batch_metric_prior_mae_sum += (
                        raw_prior.mean(dim=-1) * active_slot_mask
                    ).sum()
                    batch_metric_post_mae_sum += (
                        raw_post.mean(dim=-1) * active_slot_mask
                    ).sum()

                    total_active_slot_steps += num_active

            final_assoc_loss = batch_loss_assoc_sum / total_valid_assoc_steps
            final_assoc_acc = batch_metric_assoc_acc_sum / total_valid_assoc_steps
            final_prior_loss = batch_loss_prior_sum / total_active_slot_steps
            final_prior_mae = batch_metric_prior_mae_sum / total_active_slot_steps
            final_posterior_loss = batch_loss_posterior_sum / total_active_slot_steps
            final_posterior_mae = batch_metric_post_mae_sum / total_active_slot_steps

            gamma, lam, alpha = 1.0, 5.0, 2.0
            final_loss = (
                (gamma * final_assoc_loss)
                + (lam * final_prior_loss)
                + (alpha * final_posterior_loss)
            )

            total_epoch_loss += final_loss.item()

            # Add to Epoch Averages
            avg_assoc_loss += final_assoc_loss.item()
            avg_assoc_acc += final_assoc_acc.item()
            avg_prior_loss += final_prior_loss.item()
            avg_prior_mae += final_prior_mae.item()
            avg_post_loss += final_posterior_loss.item()
            avg_post_mae += final_posterior_mae.item()

            if batch_idx == 0:
                log_xy_plot(
                    writer,
                    epoch_index,
                    batch_data,
                    posterior_pred_states,
                    dataset_stats,
                    tag="Validation/XY_Plot",
                )

    loader_len = len(loader)
    writer.add_scalar(
        "Validation/Loss_Total", total_epoch_loss / loader_len, epoch_index
    )
    writer.add_scalar("Validation/Loss_Assoc", avg_assoc_loss / loader_len, epoch_index)
    writer.add_scalar("Validation/Acc_Assoc", avg_assoc_acc / loader_len, epoch_index)
    writer.add_scalar("Validation/Loss_Prior", avg_prior_loss / loader_len, epoch_index)
    writer.add_scalar("Validation/MAE_Prior", avg_prior_mae / loader_len, epoch_index)
    writer.add_scalar(
        "Validation/Loss_Posterior", avg_post_loss / loader_len, epoch_index
    )
    writer.add_scalar(
        "Validation/MAE_Posterior", avg_post_mae / loader_len, epoch_index
    )

    return total_epoch_loss / len(loader)


if __name__ == "__main__":
    TRAIN_TRACKS = "data/train_tracks.csv"
    TRAIN_TRUTH = "data/train_truth.csv"
    TRAIN_OWN = "data/train_ownship.csv"
    VAL_TRACKS = "data/val_tracks.csv"
    VAL_TRUTH = "data/val_truth.csv"
    VAL_OWN = "data/val_ownship.csv"
    MODEL_CHECKPOINT_DIR = "checkpoint"
    LOG_DIR = "runs/stt"
    MODEL_INIT_WEIGHTS_PATH = ""

    BATCH_SIZE = 20
    MAX_SEQUENCE_LENGTH = 60
    NUM_SENSOR_MODALITIES = 3
    MAX_NUM_DETECTS_PER_STEP = NUM_SENSOR_MODALITIES**2
    MAX_NUM_TRACK_SLOTS = 1
    DETECT_NUM_DIMS = 9
    SENSOR_TYPE_EMBEDDING_DIM = 8
    EMBEDDING_DIM = 256
    NUM_EPOCHS = 250

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
        stats=train_dataset.get_stats(),
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

    if MODEL_INIT_WEIGHTS_PATH and os.path.exists(MODEL_INIT_WEIGHTS_PATH):
        model.load_weights(MODEL_INIT_WEIGHTS_PATH)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"Starting Training on {device}...")
    best_val_loss = float("inf")
    start_time = datetime.now().strftime("%d_%H_%M")
    checkpoint_file = f"checkpoint_{start_time}.pth"
    checkpoint_filepath = os.path.join(MODEL_CHECKPOINT_DIR, checkpoint_file)
    try:
        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                train_dataset.max_num_truth_objs,
                writer,
                epoch,
                train_dataset.get_stats(),
            )
            val_loss = validate_one_epoch(
                model, val_loader, device, writer, epoch, val_dataset.get_stats()
            )
            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_filepath)
                print(f"  >>> New Best Model Saved (Val Loss: {val_loss:.4f})")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        writer.close()
        print("TensorBoard writer closed.")
