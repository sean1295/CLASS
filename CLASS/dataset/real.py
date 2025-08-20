import numpy as np
import torch
from torch.utils.data import Dataset
import roma
import zarr

from CLASS.util.normalization import (
    SafeLimitsNormalizer,
    DebugNormalizer,
)
from CLASS.dataset.robomimic import BCRobomimicDataset

class BCRealWorldDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        normalizers_path=None,
        num_demo: int = 100,
        obs_keys: list = [],
        action_space: str = "joint_pos",
        device: str = "cpu",
        obs_horizon: int = 1,
        pred_horizon: int = 1,
        normalize_action_per_timestep: bool = False
    ):
        self.action_space = action_space  # store the mode (joint_pos or pose)
        self.dataset_path = dataset_path
        self.normalizers_path = normalizers_path
        self.num_demo = num_demo
        self.obs_keys = obs_keys
        self.device = device
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.num_gpus = torch.cuda.device_count()
        self.normalize_action_per_timestep = normalize_action_per_timestep

        self._load_data()

    def _load_data(self):
        import zarr

        dataset_root = zarr.open(self.dataset_path)
        self.episode_starts = []
        current_episode = -1

        if self.num_demo == -1:
            self.num_demo = np.inf
        for idx, episode in enumerate(dataset_root["data"]["episode"]):
            if len(self.episode_starts) > self.num_demo:
                break
            if episode != current_episode:
                self.episode_starts.append(idx)
                current_episode = episode
        else:
            self.episode_starts.append(idx + 1)

        # load observations
        self.train_data = {
            key: torch.from_numpy(dataset_root["data"][key][: self.episode_starts[-1]])
            for key in self.obs_keys
        }

        # load correct action branch
        if "force" in self.action_space:
            action_key = "action_force"
        else:
            action_key = "action_pos"
        self.train_data["action"] = torch.from_numpy(
            dataset_root["data"][action_key][: self.episode_starts[-1]]
        )

        self.make_indices()

        # normalizer
        if self.normalizers_path is not None:
            self.normalizers = torch.load(self.normalizers_path, weights_only=False)
            for k in self.normalizers:
                if k in self.train_data:
                    self.normalizers[k].to_device(self.train_data[k].device)
        else:
            self.normalizers = {}

        action_data = self.generate_action_sequence()
        if "action" not in self.normalizers:
            if self.normalize_action_per_timestep:
                self.normalizers["action"] = SafeLimitsNormalizer(
                action_data.flatten(end_dim=1)
            )
            else:
                self.normalizers["action"] = SafeLimitsNormalizer(
                action_data[:,0]
            )
            
            
        self.train_data["action"] = self.normalizers["action"].normalize(action_data)
        self.normalizers["action"].to_device(self.device)

        for k in self.obs_keys:
            if "image" not in k:
                self.train_data[k] = self.train_data[k].float()
            if k not in self.normalizers:
                if "image" in k:
                    self.normalizers[k] = DebugNormalizer(torch.ones(1))
                else:
                    self.normalizers[k] = SafeLimitsNormalizer(self.train_data[k])
            if "tactile" in k:
                self.normalizers[k] = SafeLimitsNormalizer(self.train_data[k].flatten(start_dim=1))
            else:
                self.train_data[k] = self.normalizers[k].normalize(self.train_data[k])
            self.normalizers[k].to_device(self.device)
            self.train_data[k] = self.train_data[k][self.obs_indices]

        self.transform_data_structure()

    def generate_action_sequence(self):
        """Real-world action is already final, just slice over pred_horizon."""
        return self.train_data["action"][self.action_indices]

    def make_indices(self):
        obs_indices, action_indices = [], []

        for i, episode_start in enumerate(self.episode_starts[:-1]):
            next_start = self.episode_starts[i + 1]
            episode_len = next_start - episode_start

            for start in range(-self.obs_horizon + 1, episode_len):
                obs_idx = torch.tensor([
                    max(episode_start, min(episode_start + j, next_start - 1))
                    for j in range(start, start + self.obs_horizon)
                ])
                act_idx = torch.tensor([
                    max(episode_start, min(episode_start + j, next_start - 1))
                    for j in range(start, start + self.pred_horizon)
                ])
                obs_indices.append(obs_idx)
                action_indices.append(act_idx)

        self.obs_indices = torch.stack(obs_indices)
        self.action_indices = torch.stack(action_indices)

    def transform_data_structure(self):
        n_samples = len(self.obs_indices)
        self.train_data = [
            {key: value[i] for key, value in self.train_data.items()}
            for i in range(n_samples)
        ]

    def __len__(self):
        return len(self.obs_indices)

    def __getitem__(self, idx):
        return self.train_data[idx]