import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TrackingDataset(Dataset):
    def __init__(self, track_file, truth_file, ownship_file, seq_len=5, max_objs=50):
        """
        Args:
            max_objs: Fixed size to pad observations to (required for learnable tokens).
        """
        self.seq_len = seq_len
        self.max_objs = max_objs
        self.feature_dim = 12  # x, y, z, vx, vy, vz, ax, ay, az, var_x, var_y, var_z

        # Load Data
        self.tracks_df = pd.read_csv(track_file)
        self.truth_df = pd.read_csv(truth_file)
        self.own_df = pd.read_csv(ownship_file)

        # Normalize
        scale_factor = 100.0
        cols_to_scale = ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"]
        for df in [self.tracks_df, self.truth_df, self.own_df]:
            for c in cols_to_scale:
                if c in df.columns:
                    df[c] /= scale_factor

        self.episodes = self.tracks_df["episode_id"].unique()
        self.data_indices = []

        # Create indices
        for ep in self.episodes:
            frames = sorted(
                self.tracks_df[self.tracks_df["episode_id"] == ep]["frame_idx"].unique()
            )
            if len(frames) >= seq_len:
                for i in range(len(frames) - seq_len + 1):
                    self.data_indices.append((ep, frames[i : i + seq_len]))

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        ep_id, frame_seq = self.data_indices[idx]

        # Pre-allocate fixed-size tensors [seq_len, max_objs, features]
        # We fill these with 0.0 (padding) initially
        obs_tensor = torch.zeros(
            (self.seq_len, self.max_objs, self.feature_dim), dtype=torch.float32
        )

        # Mask: False (0) = Padding, True (1) = Real Data
        mask_tensor = torch.zeros((self.seq_len, self.max_objs), dtype=torch.bool)

        # We also pad IDs to keep alignment (-1 usually denotes no-object in ID lists)
        truth_id_tensor = torch.full(
            (self.seq_len, self.max_objs), -1, dtype=torch.long
        )
        sensor_id_tensor = torch.zeros((self.seq_len, self.max_objs), dtype=torch.long)

        batched_own = []

        for t, f_idx in enumerate(frame_seq):
            # 1. Get Observations
            curr_tracks = self.tracks_df[
                (self.tracks_df["episode_id"] == ep_id)
                & (self.tracks_df["frame_idx"] == f_idx)
            ]

            # Features
            feats = curr_tracks[
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
            ].values

            # Truncate if we have more tracks than max_objs
            num_objs = min(len(feats), self.max_objs)

            if num_objs > 0:
                # Fill the fixed tensor slots
                obs_tensor[t, :num_objs, :] = torch.from_numpy(
                    feats[:num_objs].astype(np.float32)
                )

                # Mark these slots as VALID
                mask_tensor[t, :num_objs] = True

                # Fill IDs
                truth_ids = curr_tracks["truth_id"].values.astype(np.int64)
                truth_id_tensor[t, :num_objs] = torch.from_numpy(truth_ids[:num_objs])

                sensor_ids = curr_tracks["sensor_id"].values.astype(np.int64)
                sensor_id_tensor[t, :num_objs] = torch.from_numpy(sensor_ids[:num_objs])

            # 2. Ownship (Standard handling)
            curr_own = self.own_df[
                (self.own_df["episode_id"] == ep_id)
                & (self.own_df["frame_idx"] == f_idx)
            ]
            if len(curr_own) > 0:
                own_feats = (
                    curr_own[["x", "y", "z", "vx", "vy", "vz"]]
                    .values[0]
                    .astype(np.float32)
                )
            else:
                own_feats = np.zeros(6, dtype=np.float32)
            batched_own.append(torch.tensor(own_feats))

        return {
            "obs_features": obs_tensor,  # Shape: [seq_len, max_objs, 12]
            "obs_mask": mask_tensor,  # Shape: [seq_len, max_objs]
            "truth_ids": truth_id_tensor,  # Shape: [seq_len, max_objs]
            "ownship": torch.stack(batched_own),
        }


def collate_fn(batch):
    # Custom collate because observations are variable length per frame
    return batch
