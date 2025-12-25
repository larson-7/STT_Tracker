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
                # sliding window for long episodes
                for i in range(len(frames) - seq_len + 1):
                    self.data_indices.append((ep, frames[i : i + seq_len]))
            else:
                self.data_indices.append((ep, frames))

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        ep_id, frame_seq = self.data_indices[idx]
        actual_len = len(frame_seq)

        # Pre-allocate fixed-size tensors [seq_len, max_objs, features]
        obs_tensor = torch.zeros(
            (self.seq_len, self.max_num_detects_per_step, self.feature_dim),
            dtype=torch.float32,
        ).to(self.device)
        mask_tensor = torch.zeros(
            (self.seq_len, self.max_num_detects_per_step), dtype=torch.bool
        ).to(self.device)
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
        truth_mask_tensor = torch.zeros(
            (self.seq_len, self.max_num_detects_per_step), dtype=torch.bool
        ).to(self.device)
        sensor_id_tensor = torch.full(
            (self.seq_len, self.max_num_detects_per_step), -1, dtype=torch.long
        ).to(self.device)

        # Track which time-steps in the sequence are real data vs padding
        valid_seq_mask = torch.zeros(self.seq_len, dtype=torch.bool).to(self.device)

        # Pre-allocate ownship to ensure it's always seq_len long
        batched_own = torch.zeros((self.seq_len, 6), dtype=torch.float32).to(
            self.device
        )

        for t, f_idx in enumerate(frame_seq):
            valid_seq_mask[t] = True  # Mark this timestep as real data

            # --- Load Detections (Tracks) ---
            curr_tracks = self.tracks_df[
                (self.tracks_df["episode_id"] == ep_id)
                & (self.tracks_df["frame_idx"] == f_idx)
            ]
            num_obs = min(len(curr_tracks), self.max_num_detects_per_step)

            if num_obs > 0:
                feats = curr_tracks[
                    ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"]
                ].values[:num_obs]
                obs_tensor[t, :num_obs, :] = torch.from_numpy(feats.astype(np.float32))
                mask_tensor[t, :num_obs] = True

                truth_ids = curr_tracks["truth_id"].values.astype(np.int64)[:num_obs]
                truth_id_tensor[t, :num_obs] = torch.from_numpy(truth_ids)

                sensor_ids = curr_tracks["sensor_id"].values.astype(np.int64)[:num_obs]
                sensor_id_tensor[t, :num_obs] = torch.from_numpy(sensor_ids)

            # --- Load Truth States ---
            truth_states = self.truth_df[
                (self.truth_df["episode_id"] == ep_id)
                & (self.truth_df["frame_idx"] == f_idx)
            ]
            num_truth_objs = min(len(truth_states), self.max_num_detects_per_step)

            if num_truth_objs > 0:
                truth_feats = truth_states[
                    ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"]
                ].values[:num_truth_objs]

                if t > 0:
                    prior_truth_states_tensor[t, :num_truth_objs, :] = (
                        posterior_truth_states_tensor[t - 1, :num_truth_objs, :]
                    )
                else:
                    prior_truth_states_tensor[t, :num_truth_objs, :] = torch.from_numpy(
                        truth_feats.astype(np.float32)
                    )

                posterior_truth_states_tensor[t, :num_truth_objs, :] = torch.from_numpy(
                    truth_feats.astype(np.float32)
                )
                truth_mask_tensor[t, :num_truth_objs] = True

            # --- Load Ownship ---
            curr_own = self.own_df[
                (self.own_df["episode_id"] == ep_id)
                & (self.own_df["frame_idx"] == f_idx)
            ]
            if len(curr_own) > 0:
                own_feats = curr_own[["x", "y", "z", "vx", "vy", "vz"]].values[0]
                batched_own[t] = torch.tensor(own_feats.astype(np.float32))

        return {
            "obs_features": obs_tensor,
            "obs_ids": sensor_id_tensor,
            "obs_mask": mask_tensor,
            "truth_ids": truth_id_tensor,
            "prior_truth_states": prior_truth_states_tensor,
            "posterior_truth_states": posterior_truth_states_tensor,
            "truth_mask": truth_mask_tensor,
            "ownship": batched_own,
            "valid_seq_mask": valid_seq_mask,  # used for loss masking
        }
