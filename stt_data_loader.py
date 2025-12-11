import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TrackingDataset(Dataset):
    def __init__(self, track_file, truth_file, ownship_file, seq_len=5):
        """
        Args:
            seq_len: Number of consecutive frames to return per batch item (for temporal training)
        """
        self.seq_len = seq_len

        # Load Data
        self.tracks_df = pd.read_csv(track_file)
        self.truth_df = pd.read_csv(truth_file)
        self.own_df = pd.read_csv(ownship_file)

        # Normalize continuous values (simple max-min or standard scaling recommended in prod)
        # Here we just scale down by 100.0 to keep nums manageable for the Transformer
        scale_factor = 100.0
        cols_to_scale = ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"]

        for c in cols_to_scale:
            if c in self.tracks_df.columns:
                self.tracks_df[c] /= scale_factor
            if c in self.truth_df.columns:
                self.truth_df[c] /= scale_factor
            if c in self.own_df.columns:
                self.own_df[c] /= scale_factor

        # Grouping for fast access
        self.episodes = self.tracks_df["episode_id"].unique()
        self.data_indices = []

        # Create indices for valid sequences
        for ep in self.episodes:
            frames = self.tracks_df[self.tracks_df["episode_id"] == ep][
                "frame_idx"
            ].unique()
            frames = sorted(frames)
            # We need sequences of length seq_len
            if len(frames) >= seq_len:
                for i in range(len(frames) - seq_len + 1):
                    self.data_indices.append((ep, frames[i : i + seq_len]))

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        ep_id, frame_seq = self.data_indices[idx]

        batched_obs = []
        batched_gt = []
        batched_own = []
        batched_labels = []

        for f_idx in frame_seq:
            # 1. Get Observations (Detections) for this frame
            curr_tracks = self.tracks_df[
                (self.tracks_df["episode_id"] == ep_id)
                & (self.tracks_df["frame_idx"] == f_idx)
            ]

            # Input Features: [x, y, z, vx, vy, vz, ax, ay, az, var_x, var_y, var_z]
            # We explicitly exclude truth_id from input, but keep it for label matching
            obs_feats = curr_tracks[
                [
                    "x",
                    "y",
                    "z",
                    "vx",
                    "vy",
                    "vz",
                    "ax",
                    "ay",
                    "az",
                    "var_x",
                    "var_y",
                    "var_z",
                ]
            ].values.astype(np.float32)

            sensor_ids = curr_tracks["sensor_id"].values.astype(np.int64)

            # Labels for training association (which detection belongs to which truth)
            # In a real scenario, you might not have this and would use Hungarian matching on coords
            truth_ids_in_obs = curr_tracks["truth_id"].values.astype(np.int64)

            # 2. Get Ground Truth States (Targets)
            curr_truth = self.truth_df[
                (self.truth_df["episode_id"] == ep_id)
                & (self.truth_df["frame_idx"] == f_idx)
            ]
            gt_states = curr_truth[
                ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"]
            ].values.astype(np.float32)
            gt_ids = curr_truth["object_id"].values.astype(np.int64)

            # 3. Get Ownship (Context)
            curr_own = self.own_df[
                (self.own_df["episode_id"] == ep_id)
                & (self.own_df["frame_idx"] == f_idx)
            ]
            # If ownship is missing, pad with zeros
            if len(curr_own) > 0:
                own_feats = (
                    curr_own[["x", "y", "z", "vx", "vy", "vz"]]
                    .values[0]
                    .astype(np.float32)
                )
            else:
                own_feats = np.zeros(6, dtype=np.float32)

            batched_obs.append(
                {
                    "features": torch.tensor(obs_feats),
                    "sensor_id": torch.tensor(sensor_ids),
                    "truth_id": torch.tensor(truth_ids_in_obs),
                }
            )

            batched_gt.append(
                {"states": torch.tensor(gt_states), "ids": torch.tensor(gt_ids)}
            )

            batched_own.append(torch.tensor(own_feats))

        return {
            "observations": batched_obs,  # List of dicts (len=seq_len)
            "ground_truth": batched_gt,  # List of dicts (len=seq_len)
            "ownship": torch.stack(batched_own),  # [seq_len, 6]
        }


def collate_fn(batch):
    # Custom collate because observations are variable length per frame
    return batch
