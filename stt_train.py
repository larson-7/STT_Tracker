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

from models.stt import STTTracker
from stt_data_loader import TrackingDataset

# Hyperparams
GAMMA = 2.0  # Association
LAMBDA = 1.0  # Prior Kinematics
ALPHA = 1.0  # Posterior Kinematics
BETA = 5.0  # Existence (High to remove not used track slots)

# Directories
TRAIN_TRACKS = "data/train_tracks.csv"
TRAIN_TRUTH = "data/train_truth.csv"
TRAIN_OWN = "data/train_ownship.csv"
VAL_TRACKS = "data/val_tracks.csv"
VAL_TRUTH = "data/val_truth.csv"
VAL_OWN = "data/val_ownship.csv"
MODEL_CHECKPOINT_DIR = "checkpoint"
LOG_DIR = "runs/stt"
MODEL_INIT_WEIGHTS_PATH = ""

# Model Params
BATCH_SIZE = 20
MAX_SEQUENCE_LENGTH = 60
NUM_SENSOR_MODALITIES = 3
MAX_NUM_DETECTS_PER_STEP = NUM_SENSOR_MODALITIES**2
MAX_NUM_TRACK_SLOTS = 10
DETECT_NUM_DIMS = 9
SENSOR_TYPE_EMBEDDING_DIM = 8
EMBEDDING_DIM = 256
NUM_EPOCHS = 1000


def sigmoid_focal_loss(
    inputs, targets, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "none"
):
    """
    Focal Loss for binary classification.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def get_curriculum_weights(epoch, device):
    weights = torch.tensor(
        [
            1.0,
            1.0,
            1.0,  # Position
            2.0,
            2.0,
            2.0,  # Velocity
            0.1,
            0.1,
            0.1,  # Acceleration
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
    pred_exist_logits,
    dataset_stats,
    batch_idx=0,
    tag="Tracking/XY_Plot",
    match_threshold=5.0,
):
    b = batch_idx
    valid_seq_mask = batch_data["valid_seq_mask"][b].detach().cpu().numpy()

    mu_x, std_x = dataset_stats["x"]["mean"], dataset_stats["x"]["std"]
    mu_y, std_y = dataset_stats["y"]["mean"], dataset_stats["y"]["std"]

    def unnorm_traj(arr):
        x = arr[..., 0] * std_x + mu_x
        y = arr[..., 1] * std_y + mu_y
        return x, y

    gt_raw = (
        batch_data["posterior_truth_states"][b].detach().cpu().numpy()[valid_seq_mask]
    )
    gt_x, gt_y = unnorm_traj(gt_raw)

    own_raw = batch_data["ownship"][b].detach().cpu().numpy()[valid_seq_mask]
    own_x, own_y = unnorm_traj(own_raw)

    det_features = batch_data["obs_features"][b].detach().cpu().numpy()[valid_seq_mask]
    det_x, det_y = unnorm_traj(det_features)

    obs_mask = batch_data["obs_mask"][b].detach().cpu().numpy()[valid_seq_mask]
    truth_ids = batch_data["truth_ids"][b].detach().cpu().numpy()[valid_seq_mask]

    pred_raw = pred_states[b].detach().cpu().numpy()[:, valid_seq_mask, :]
    pred_x, pred_y = unnorm_traj(pred_raw)

    exist_probs = (
        torch.sigmoid(pred_exist_logits[b]).detach().cpu().numpy()[:, valid_seq_mask, 0]
    )

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title(f"Epoch {epoch} - {tag} (Units: Meters)")
    ax.grid(True, alpha=0.3)
    cmap = matplotlib.colormaps["tab10"]

    # Ownship
    ax.plot(
        own_x,
        own_y,
        color="black",
        linewidth=3,
        linestyle="--",
        alpha=0.7,
        label="Ownship",
    )
    ax.scatter(own_x[0], own_y[0], color="black", marker="o", s=50)
    ax.scatter(own_x[-1], own_y[-1], color="black", marker="X", s=80)

    # Ground Truth
    gt_colors = {}
    _, max_gt, _ = gt_raw.shape
    for gt_idx in range(max_gt):
        if (np.abs(gt_raw[:, gt_idx, :2]) < 1e-3).all():
            continue
        c = cmap(gt_idx % 10)
        gt_colors[gt_idx] = c
        px, py = gt_x[:, gt_idx], gt_y[:, gt_idx]
        ax.plot(
            px,
            py,
            color=c,
            marker=".",
            markersize=4,
            alpha=0.5,
            linewidth=2,
            label=f"GT {gt_idx}",
        )
        ax.scatter(
            px[0], py[0], color=c, marker="*", s=120, edgecolors="black", zorder=5
        )
        ax.text(px[-1], py[-1], f"GT{gt_idx}", color=c, fontsize=9, fontweight="bold")

    # Predictions
    num_slots, _, _ = pred_raw.shape
    for slot_idx in range(num_slots):
        mean_prob = np.mean(exist_probs[slot_idx])
        if mean_prob < 0.5:
            continue

        px, py = pred_x[slot_idx, :], pred_y[slot_idx, :]
        best_dist, best_gt = float("inf"), -1
        for gt_idx in range(max_gt):
            if gt_idx not in gt_colors:
                continue
            gx, gy = gt_x[:, gt_idx], gt_y[:, gt_idx]
            dist = np.mean(np.sqrt((px - gx) ** 2 + (py - gy) ** 2))
            if dist < best_dist:
                best_dist, best_gt = dist, gt_idx

        if best_dist < match_threshold:
            color, line_style, alpha_val = gt_colors[best_gt], "-", 0.9
        else:
            color, line_style, alpha_val = "gray", ":", 0.4

        ax.plot(
            px,
            py,
            color=color,
            marker=">",
            markersize=4,
            linestyle=line_style,
            alpha=alpha_val,
            label="Predictions" if slot_idx == 0 else "",
        )
        ax.scatter(
            px[0], py[0], color=color, marker="*", s=120, edgecolors="black", zorder=5
        )

    # Detections
    plotted_labels = set()
    for t in range(valid_seq_mask.sum()):
        for d in range(det_features.shape[1]):
            if not obs_mask[t, d]:
                continue
            tid = int(truth_ids[t, d])
            marker, sz, det_color, lbl = (
                (".", 4, gt_colors.get(tid, "black"), "True Det")
                if tid >= 0
                else ("x", 4, "red", "Clutter")
            )
            ax.plot(
                det_x[t, d],
                det_y[t, d],
                marker=marker,
                color=det_color,
                markersize=sz,
                linestyle="None",
                alpha=0.3,
                label=lbl if lbl not in plotted_labels else "",
            )
            plotted_labels.add(lbl)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    sorted_keys = sorted(
        by_label.keys(), key=lambda x: (0 if "Own" in x else 1 if "GT" in x else 2)
    )
    ax.legend(
        [by_label[k] for k in sorted_keys],
        sorted_keys,
        loc="upper right",
        bbox_to_anchor=(1.15, 1),
        fontsize="small",
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

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train Ep {epoch_index}")

    for batch_idx, batch_data in pbar:
        gt_prior_states = batch_data["prior_truth_states"].to(device)
        gt_posterior_states = batch_data["posterior_truth_states"].to(device)
        det_truth_ids = batch_data["truth_ids"].to(device)
        gt_object_ids = batch_data["gt_ids"].to(device)
        valid_mask = batch_data["valid_seq_mask"].to(device)

        outputs = model(batch_data)

        prior_pred_states = outputs["prior_kinematics"]
        posterior_pred_states = outputs["posterior_kinematics"]
        pred_assoc_logits = outputs["association_scores"]
        pred_exist_logits = outputs["posterior_existence_logits"]

        batch_size, num_track_slots, seq_len, _ = pred_assoc_logits.shape
        max_detects = pred_assoc_logits.shape[-1]
        track_assignments = torch.full(
            (batch_size, num_track_slots), -1, dtype=torch.long
        ).to(device)

        batch_loss_assoc = 0.0
        batch_loss_exist = 0.0
        batch_loss_prior = 0.0
        batch_loss_post = 0.0
        batch_acc_assoc = 0.0
        batch_acc_exist = 0.0
        batch_mae_post = 0.0

        total_valid_steps = 1e-6
        total_active_slot_steps = 1e-6

        for t in range(seq_len):
            is_step_valid = valid_mask[:, t] > 0
            current_det_ids = det_truth_ids[:, t]
            current_gt_ids = gt_object_ids[:, t]

            target_assoc_map = torch.zeros_like(pred_assoc_logits[:, :, t, :])
            target_exist_map = torch.zeros_like(pred_exist_logits[:, :, t, :])
            active_slot_mask = torch.zeros((batch_size, num_track_slots), device=device)
            aligned_gt_prior = torch.zeros_like(prior_pred_states[:, :, t, :])
            aligned_gt_posterior = torch.zeros_like(posterior_pred_states[:, :, t, :])

            for b in range(batch_size):
                if not is_step_valid[b]:
                    continue

                valid_gt_indices = (current_gt_ids[b] != -1).nonzero().squeeze(-1)
                for gt_idx in valid_gt_indices:
                    tid = current_gt_ids[b, gt_idx].item()
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

                    target_exist_map[b, slot_idx] = 1.0
                    aligned_gt_prior[b, slot_idx] = gt_prior_states[b, t, gt_idx]
                    aligned_gt_posterior[b, slot_idx] = gt_posterior_states[
                        b, t, gt_idx
                    ]
                    active_slot_mask[b, slot_idx] = 1.0

                valid_det_indices = (current_det_ids[b] != -1).nonzero().squeeze(-1)
                for det_idx in valid_det_indices:
                    tid = current_det_ids[b, det_idx].item()
                    slot_idx = (track_assignments[b] == tid).nonzero()
                    if len(slot_idx) > 0:
                        slot_idx = slot_idx[0].item()
                        if det_idx < max_detects:
                            target_assoc_map[b, slot_idx, det_idx] = 1.0

            # Losses
            loss_assoc = F.binary_cross_entropy_with_logits(
                pred_assoc_logits[:, :, t, :], target_assoc_map, reduction="none"
            )
            batch_loss_assoc += (
                loss_assoc.mean(dim=(1, 2)) * is_step_valid.float()
            ).sum()

            loss_exist = sigmoid_focal_loss(
                pred_exist_logits[:, :, t, :], target_exist_map
            )
            batch_loss_exist += (
                loss_exist.mean(dim=(1, 2)) * is_step_valid.float()
            ).sum()

            pred_assoc_bin = (pred_assoc_logits[:, :, t, :] > 0.0).float()
            batch_acc_assoc += (
                (pred_assoc_bin == target_assoc_map).float().mean(dim=(1, 2))
                * is_step_valid.float()
            ).sum()

            pred_exist_bin = (pred_exist_logits[:, :, t, :] > 0.0).float()
            batch_acc_exist += (
                (pred_exist_bin == target_exist_map).float().mean(dim=(1, 2))
                * is_step_valid.float()
            ).sum()

            total_valid_steps += is_step_valid.sum()

            num_active = active_slot_mask.sum()
            if num_active > 0:
                raw_prior = l1_loss_fn(prior_pred_states[:, :, t, :], aligned_gt_prior)
                raw_post = l1_loss_fn(
                    posterior_pred_states[:, :, t, :], aligned_gt_posterior
                )
                w_prior = raw_prior * loss_weights.view(1, 1, -1)
                w_post = raw_post * loss_weights.view(1, 1, -1)
                batch_loss_prior += (w_prior.sum(dim=-1) * active_slot_mask).sum()
                batch_loss_post += (w_post.sum(dim=-1) * active_slot_mask).sum()
                batch_mae_post += (raw_post.mean(dim=-1) * active_slot_mask).sum()
                total_active_slot_steps += num_active

        # Averages
        avg_assoc_loss = batch_loss_assoc / total_valid_steps
        avg_exist_loss = batch_loss_exist / total_valid_steps
        avg_assoc_acc = batch_acc_assoc / total_valid_steps
        avg_exist_acc = batch_acc_exist / total_valid_steps

        avg_prior_loss = (
            batch_loss_prior / total_active_slot_steps
            if total_active_slot_steps > 0
            else 0.0
        )
        avg_post_loss = (
            batch_loss_post / total_active_slot_steps
            if total_active_slot_steps > 0
            else 0.0
        )
        avg_mae = (
            batch_mae_post / total_active_slot_steps
            if total_active_slot_steps > 0
            else 0.0
        )

        final_loss = (
            (GAMMA * avg_assoc_loss)
            + (BETA * avg_exist_loss)
            + (LAMBDA * avg_prior_loss)
            + (ALPHA * avg_post_loss)
        )

        optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_epoch_loss += final_loss.item()
        global_step = epoch_index * len(loader) + batch_idx

        writer.add_scalar("Train/Loss_Total", final_loss.item(), global_step)
        writer.add_scalar("Train/Loss_Exist", avg_exist_loss.item(), global_step)
        writer.add_scalar("Train/Acc_Exist", avg_exist_acc.item(), global_step)
        writer.add_scalar("Train/Loss_Assoc", avg_assoc_loss.item(), global_step)
        writer.add_scalar("Train/Acc_Assoc", avg_assoc_acc.item(), global_step)
        writer.add_scalar("Train/MAE_Posterior", avg_mae.item(), global_step)

        pbar.set_postfix(
            {
                "Loss": f"{final_loss.item():.2f}",
                "Exist": f"{avg_exist_acc.item():.2f}",
                "MAE": f"{avg_mae.item():.2f}",
            }
        )

        if batch_idx == 0:
            log_xy_plot(
                writer,
                epoch_index,
                batch_data,
                posterior_pred_states,
                pred_exist_logits,
                dataset_stats,
                tag="Train/XY_Plot",
            )

    return total_epoch_loss / len(loader)


def validate_one_epoch(model, loader, device, writer, epoch_index, dataset_stats):
    model.eval()
    total_epoch_loss = 0
    l1_loss_fn = nn.L1Loss(reduction="none")
    loss_weights = get_curriculum_weights(epoch_index, device)

    avg_assoc_loss = 0.0
    avg_exist_loss = 0.0
    avg_prior_loss = 0.0
    avg_post_loss = 0.0
    avg_assoc_acc = 0.0
    avg_exist_acc = 0.0
    avg_mae = 0.0

    with torch.no_grad():
        pbar = tqdm(
            enumerate(loader), total=len(loader), desc=f"Valid Ep {epoch_index}"
        )
        for batch_idx, batch_data in pbar:
            gt_prior_states = batch_data["prior_truth_states"].to(device)
            gt_posterior_states = batch_data["posterior_truth_states"].to(device)
            det_truth_ids = batch_data["truth_ids"].to(device)
            gt_object_ids = batch_data["gt_ids"].to(device)
            valid_mask = batch_data["valid_seq_mask"].to(device)

            outputs = model(batch_data)
            prior_pred_states = outputs["prior_kinematics"]
            posterior_pred_states = outputs["posterior_kinematics"]
            pred_assoc_logits = outputs["association_scores"]
            pred_exist_logits = outputs["posterior_existence_logits"]

            batch_size, num_track_slots, seq_len, _ = pred_assoc_logits.shape
            max_detects = pred_assoc_logits.shape[-1]
            track_assignments = torch.full(
                (batch_size, num_track_slots), -1, dtype=torch.long
            ).to(device)

            b_loss_assoc = 0.0
            b_loss_exist = 0.0
            b_loss_prior = 0.0
            b_loss_post = 0.0
            b_acc_assoc = 0.0
            b_acc_exist = 0.0
            b_mae = 0.0
            tot_valid = 1e-6
            tot_active = 1e-6

            for t in range(seq_len):
                is_step_valid = valid_mask[:, t] > 0
                current_det_ids, current_gt_ids = (
                    det_truth_ids[:, t],
                    gt_object_ids[:, t],
                )
                target_assoc = torch.zeros_like(pred_assoc_logits[:, :, t, :])
                target_exist = torch.zeros_like(pred_exist_logits[:, :, t, :])
                active_mask = torch.zeros((batch_size, num_track_slots), device=device)
                aligned_prior = torch.zeros_like(prior_pred_states[:, :, t, :])
                aligned_post = torch.zeros_like(posterior_pred_states[:, :, t, :])

                for b in range(batch_size):
                    if not is_step_valid[b]:
                        continue
                    valid_gt = (current_gt_ids[b] != -1).nonzero().squeeze(-1)
                    for gt_idx in valid_gt:
                        tid = current_gt_ids[b, gt_idx].item()
                        slot_idx = (track_assignments[b] == tid).nonzero()
                        if len(slot_idx) > 0:
                            slot_idx = slot_idx[0].item()
                        else:
                            free = (track_assignments[b] == -1).nonzero()
                            if len(free) > 0:
                                slot_idx = free[0].item()
                                track_assignments[b, slot_idx] = tid
                            else:
                                continue
                        target_exist[b, slot_idx] = 1.0
                        aligned_prior[b, slot_idx] = gt_prior_states[b, t, gt_idx]
                        aligned_post[b, slot_idx] = gt_posterior_states[b, t, gt_idx]
                        active_mask[b, slot_idx] = 1.0

                    valid_det = (current_det_ids[b] != -1).nonzero().squeeze(-1)
                    for det_idx in valid_det:
                        tid = current_det_ids[b, det_idx].item()
                        slot_idx = (track_assignments[b] == tid).nonzero()
                        if len(slot_idx) > 0:
                            slot_idx = slot_idx[0].item()
                            if det_idx < max_detects:
                                target_assoc[b, slot_idx, det_idx] = 1.0

                loss_assoc = F.binary_cross_entropy_with_logits(
                    pred_assoc_logits[:, :, t, :], target_assoc, reduction="none"
                )
                b_loss_assoc += (
                    loss_assoc.mean(dim=(1, 2)) * is_step_valid.float()
                ).sum()

                loss_exist = sigmoid_focal_loss(
                    pred_exist_logits[:, :, t, :], target_exist
                )
                b_loss_exist += (
                    loss_exist.mean(dim=(1, 2)) * is_step_valid.float()
                ).sum()

                pred_assoc_bin = (pred_assoc_logits[:, :, t, :] > 0.0).float()
                b_acc_assoc += (
                    (pred_assoc_bin == target_assoc).float().mean(dim=(1, 2))
                    * is_step_valid.float()
                ).sum()
                pred_exist_bin = (pred_exist_logits[:, :, t, :] > 0.0).float()
                b_acc_exist += (
                    (pred_exist_bin == target_exist).float().mean(dim=(1, 2))
                    * is_step_valid.float()
                ).sum()
                tot_valid += is_step_valid.sum()

                num_act = active_mask.sum()
                if num_act > 0:
                    raw_p = l1_loss_fn(prior_pred_states[:, :, t, :], aligned_prior)
                    raw_pst = l1_loss_fn(
                        posterior_pred_states[:, :, t, :], aligned_post
                    )
                    w_p, w_pst = raw_p * loss_weights.view(
                        1, 1, -1
                    ), raw_pst * loss_weights.view(1, 1, -1)
                    b_loss_prior += (w_p.sum(dim=-1) * active_mask).sum()
                    b_loss_post += (w_pst.sum(dim=-1) * active_mask).sum()
                    b_mae += (raw_pst.mean(dim=-1) * active_mask).sum()
                    tot_active += num_act

            final_loss = (GAMMA * (b_loss_assoc / tot_valid)) + (
                BETA * (b_loss_exist / tot_valid)
            )
            if tot_active > 0:
                final_loss += (LAMBDA * (b_loss_prior / tot_active)) + (
                    ALPHA * (b_loss_post / tot_active)
                )

            total_epoch_loss += final_loss.item()
            avg_assoc_loss += (b_loss_assoc / tot_valid).item()
            avg_exist_loss += (b_loss_exist / tot_valid).item()
            avg_assoc_acc += (b_acc_assoc / tot_valid).item()
            avg_exist_acc += (b_acc_exist / tot_valid).item()
            if tot_active > 0:
                avg_mae += (b_mae / tot_active).item()

            if batch_idx == 0:
                log_xy_plot(
                    writer,
                    epoch_index,
                    batch_data,
                    posterior_pred_states,
                    pred_exist_logits,
                    dataset_stats,
                    tag="Validation/XY_Plot",
                )

    loader_len = len(loader)
    writer.add_scalar(
        "Validation/Loss_Total", total_epoch_loss / loader_len, epoch_index
    )
    writer.add_scalar("Validation/Loss_Exist", avg_exist_loss / loader_len, epoch_index)
    writer.add_scalar("Validation/Acc_Exist", avg_exist_acc / loader_len, epoch_index)
    writer.add_scalar("Validation/Loss_Assoc", avg_assoc_loss / loader_len, epoch_index)
    writer.add_scalar("Validation/Acc_Assoc", avg_assoc_acc / loader_len, epoch_index)
    writer.add_scalar("Validation/MAE_Posterior", avg_mae / loader_len, epoch_index)

    return total_epoch_loss / len(loader)


if __name__ == "__main__":
    writer = SummaryWriter(LOG_DIR)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

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

    # Bias Initialization for Existence Head
    if hasattr(model.posterior_state_decoder, "existence_head"):
        print(">>> Applying Bias Initialization to Existence Head (Bias = -2.0)")
        nn.init.constant_(model.posterior_state_decoder.existence_head.bias, -2.0)
    else:
        print(">>> WARNING: Existence head not found, skipping bias init.")

    if MODEL_INIT_WEIGHTS_PATH and os.path.exists(MODEL_INIT_WEIGHTS_PATH):
        model.load_weights(MODEL_INIT_WEIGHTS_PATH)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"Starting Training on {device}...")
    best_val_loss = float("inf")
    start_time = datetime.now().strftime("%d_%H_%M")
    checkpoint_filepath = os.path.join(
        MODEL_CHECKPOINT_DIR, f"checkpoint_{start_time}.pth"
    )

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
