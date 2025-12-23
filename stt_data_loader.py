import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class TrackingDataset(Dataset):
    def __init__(
        self,
        track_file,
        truth_file,
        ownship_file,
        seq_len=5,
        max_num_detects_per_step=3,
        device="cpu",
    ):
        """
        Args:
            max_num_detects_per_step: Fixed size to pad observations to (required for learnable tokens).
        """
        self.seq_len = seq_len
        self.max_num_detects_per_step = max_num_detects_per_step
        self.feature_dim = 9  # x, y, z, vx, vy, vz, ax, ay, az

        # Load Data
        self.tracks_df = pd.read_csv(track_file)
        self.truth_df = pd.read_csv(truth_file)
        self.max_num_truth_entities = len(self.truth_df["object_id"].unique())
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
        self.device = device
        
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
        obs_tensor = torch.zeros(
            (self.seq_len, self.max_num_detects_per_step, self.feature_dim),
            dtype=torch.float32,
        ).to(self.device)
        mask_tensor = torch.zeros(
            (self.seq_len, self.max_num_detects_per_step), dtype=torch.bool
        ).to(self.device)

        # Pad IDs to keep alignment (-1 denotes no-object in ID lists)
        truth_id_tensor = torch.full(
            (self.seq_len, self.max_num_detects_per_step), -1, dtype=torch.long
        ).to(self.device)
        prior_truth_states_tensor = torch.zeros(
            (self.seq_len, self.max_num_detects_per_step, self.feature_dim),
            dtype=torch.float32,
        ).to(self.device)
        posterior_truth_states_tensor = torch.zeros(
            (self.seq_len, self.max_num_detects_per_step, self.feature_dim),
            dtype=torch.float32,
        ).to(self.device)
        # Mask: False (0) = Padding, True (1) = Real Ground Truth Data
        truth_mask_tensor = torch.zeros(
            (self.seq_len, self.max_num_detects_per_step), dtype=torch.bool
        ).to(self.device)

        sensor_id_tensor = torch.full(
            (self.seq_len, self.max_num_detects_per_step), -1, dtype=torch.long
        ).to(self.device)

        batched_own = []

        for t, f_idx in enumerate(frame_seq):
            curr_tracks = self.tracks_df[
                (self.tracks_df["episode_id"] == ep_id)
                & (self.tracks_df["frame_idx"] == f_idx)
            ]
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
                ]
            ].values

            truth_states = self.truth_df[
                (self.truth_df["episode_id"] == ep_id)
                & (self.truth_df["frame_idx"] == f_idx)
            ]

            # Truth Features
            truth_feats = truth_states[
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
                ]
            ].values

            # Use the number of objects from the truth data
            num_truth_objs = min(len(truth_feats), self.max_num_detects_per_step)

            # Fill Truth States
            if num_truth_objs > 0:
                if t > 0:
                    prior_truth_states_tensor[t - 1, :num_truth_objs, :] = (
                        torch.from_numpy(
                            truth_feats[:num_truth_objs].astype(np.float32)
                        )
                    )
                else:
                    prior_truth_states_tensor[t, :num_truth_objs, :] = torch.from_numpy(
                        truth_feats[:num_truth_objs].astype(np.float32)
                    )

                posterior_truth_states_tensor[t, :num_truth_objs, :] = torch.from_numpy(
                    truth_feats[:num_truth_objs].astype(np.float32)
                )
                # Set the Truth Mask
                truth_mask_tensor[t, :num_truth_objs] = True

            # Truncate if we have more tracks than max_objs
            num_obs = min(len(feats), self.max_num_detects_per_step)

            if num_obs > 0:
                # Fill the fixed tensor slots
                obs_tensor[t, :num_obs, :] = torch.from_numpy(
                    feats[:num_obs].astype(np.float32)
                )

                # Mark these slots as VALID observations
                mask_tensor[t, :num_obs] = True

                # Fill IDs
                truth_ids = curr_tracks["truth_id"].values.astype(np.int64)
                truth_id_tensor[t, :num_obs] = torch.from_numpy(truth_ids[:num_obs])

                sensor_ids = curr_tracks["sensor_id"].values.astype(np.int64)
                sensor_id_tensor[t, :num_obs] = torch.from_numpy(sensor_ids[:num_obs])

            # Ownship (Standard handling)
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
            "obs_features": obs_tensor,  # Shape: [seq_len, max_objs, features]
            "obs_ids": sensor_id_tensor,  # Shape: [seq_len, max_obs]
            "obs_mask": mask_tensor,  # Shape: [seq_len, max_objs]
            "truth_ids": truth_id_tensor,  # Shape: [seq_len, max_objs] (IDs of observed tracks)
            "prior_truth_states": prior_truth_states_tensor,  # Shape [seq_len, max_objs, features]
            "posterior_truth_states": posterior_truth_states_tensor,  # Shape [seq_len, max_objs, features]
            "truth_mask": truth_mask_tensor,  # Shape: [seq_len, max_objs] (The new mask)
            "ownship": torch.stack(batched_own),
        }
