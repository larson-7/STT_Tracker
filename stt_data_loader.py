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
        device: torch.device | str = "cpu",
        # Allow passing pre-computed stats so Train and Val use SAME scaling
        stats: dict | None = None,
    ):
        self.seq_len = seq_len
        self.max_num_detects_per_step = max_num_detects_per_step
        self.feature_dim = 9
        self.device = device
        self.max_num_truth_objs = 0

        # Load Data
        self.tracks_df = pd.read_csv(track_file)
        self.truth_df = pd.read_csv(truth_file)
        self.own_df = pd.read_csv(ownship_file)

        cols_to_scale = ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"]

        if stats is None:
            # Calculate Mean and Std from Truth (most reliable source)
            # You could also concatenate tracks_df + truth_df to get global stats
            print("Computing normalization statistics from training data...")
            self.stats = {}
            for c in cols_to_scale:
                mean = self.truth_df[c].mean()
                std = self.truth_df[c].std()
                # Avoid division by zero for constant columns (like Z often is)
                if std < 1e-6:
                    std = 1.0
                self.stats[c] = {"mean": mean, "std": std}
        else:
            self.stats = stats

        # Apply Standardization: (Value - Mean) / Std
        for df in [self.tracks_df, self.truth_df, self.own_df]:
            for c in cols_to_scale:
                if c in df.columns:
                    mu = self.stats[c]["mean"]
                    sigma = self.stats[c]["std"]
                    df[c] = (df[c] - mu) / sigma

        self.episodes = self.tracks_df["episode_id"].unique()
        self.data_indices = []
        for ep in self.episodes:
            frames = sorted(
                self.tracks_df[self.tracks_df["episode_id"] == ep]["frame_idx"].unique()
            )
            if len(frames) >= seq_len:
                for i in range(len(frames) - seq_len + 1):
                    self.data_indices.append((ep, frames[i : i + seq_len]))
            else:
                self.data_indices.append((ep, frames))

    def get_stats(self):
        """Return stats to be passed to Validation set"""
        return self.stats

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        ep_id, frame_seq = self.data_indices[idx]

        # Pre-allocate tensors
        obs_tensor = torch.zeros(
            (self.seq_len, self.max_num_detects_per_step, self.feature_dim),
            dtype=torch.float32,
        ).to(self.device)
        mask_tensor = torch.zeros(
            (self.seq_len, self.max_num_detects_per_step), dtype=torch.bool
        ).to(self.device)

        # This tensor holds IDs associated with DETECTIONS (can have gaps/missing IDs)
        truth_id_tensor = torch.full(
            (self.seq_len, self.max_num_detects_per_step), -1, dtype=torch.long
        ).to(self.device)

        # This tensor holds IDs associated with GROUND TRUTH (no gaps if object exists)
        gt_ids_tensor = torch.full(
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

        batched_own = torch.zeros((self.seq_len, 6), dtype=torch.float32).to(
            self.device
        )
        valid_seq_mask = torch.zeros(self.seq_len, dtype=torch.bool).to(self.device)

        # Dictionary to store the LAST known state for every Truth ID encountered
        # Format: { truth_id: torch.Tensor(state) }
        last_known_truth_states = {}

        for t, f_idx in enumerate(frame_seq):
            valid_seq_mask[t] = True
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

            truth_subset = self.truth_df[
                (self.truth_df["episode_id"] == ep_id)
                & (self.truth_df["frame_idx"] == f_idx)
            ]

            num_truth_objs = min(len(truth_subset), self.max_num_detects_per_step)

            if num_truth_objs > 0:
                # Get the IDs and Features for this specific frame
                current_ids = truth_subset["object_id"].values[:num_truth_objs]
                current_feats = truth_subset[
                    ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"]
                ].values[:num_truth_objs]
                gt_ids_tensor[t, :num_truth_objs] = torch.from_numpy(
                    current_ids.astype(np.int64)
                )

                for i in range(num_truth_objs):
                    tid = current_ids[i]
                    feat_vec = torch.from_numpy(current_feats[i].astype(np.float32))

                    # Fill Posterior (Current State)
                    posterior_truth_states_tensor[t, i] = feat_vec
                    truth_mask_tensor[t, i] = True

                    # Fill Prior (Previous State) using ID cache
                    if tid in last_known_truth_states:
                        prior_truth_states_tensor[t, i] = last_known_truth_states[tid]
                    else:
                        prior_truth_states_tensor[t, i] = feat_vec

                    # Update the cache for the NEXT timestep
                    last_known_truth_states[tid] = feat_vec

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
            "gt_ids": gt_ids_tensor,
            "prior_truth_states": prior_truth_states_tensor,
            "posterior_truth_states": posterior_truth_states_tensor,
            "truth_mask": truth_mask_tensor,
            "ownship": batched_own,
            "valid_seq_mask": valid_seq_mask,
        }
