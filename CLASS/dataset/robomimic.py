import numpy as np
import torch
from torch.utils.data import Dataset
import roma
import zarr

from CLASS.util.rotations import rotation_6d_to_matrix, matrix_to_rotation_6d
from CLASS.util.normalization import SafeLimitsNormalizer, DebugNormalizer

class BCRobomimicDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        normalizers_path=None,
        num_demo: int = -1,
        obs_keys: list = [],
        action_space: str = "joint_pos",
        device: str = "cpu",
        obs_horizon: int = 1,
        pred_horizon: int = 1,
        normalize_action_per_timestep: bool = False
    ) -> None:
        self.obs_keys = obs_keys
        self.action_space = action_space
        self.device = device
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        assert num_demo != 0, "num_demo must be a positive integer. To use all episodes, set it as -1."

        dataset_root = zarr.open(dataset_path)
        self.episode_starts = []
        current_episode = -1

        if num_demo == -1:
            num_demo = np.inf

        for idx, episode in enumerate(dataset_root["data"]["episode"]):
            if len(self.episode_starts) > num_demo:
                break
            if episode != current_episode:
                self.episode_starts.append(idx)
                current_episode = episode
        else:
            self.episode_starts.append(idx + 1)  # Sentinel for final episode range

        self.train_data = {
            key: torch.from_numpy(dataset_root["data"][key][: self.episode_starts[-1]])
            for key in self.obs_keys + ["action"]
        }
        self.make_indices()

        if normalizers_path is not None:
            self.normalizers = torch.load(normalizers_path, weights_only=False)
            for k in self.normalizers:
                if k in self.train_data:
                    self.normalizers[k].to_device(self.train_data[k].device)
        else:
            self.normalizers = {}        

        for k in ["action"] + self.obs_keys:        
            if k == "action": 
                action_data = self.generate_action_sequence()
                if k not in self.normalizers:
                    if normalize_action_per_timestep:
                        self.normalizers[k] = SafeLimitsNormalizer(action_data)
                    else:
                        self.normalizers[k] = SafeLimitsNormalizer(action_data[:,0])
                self.train_data[k] = action_data 
            else:
                self.train_data[k] = self.train_data[k].to(self.device)
                if "image" in k:
                    self.normalizers[k] = DebugNormalizer(torch.ones(1))
                    self.train_data[k] = self.train_data[k][self.obs_indices]
                else:
                    self.train_data[k] = self.train_data[k].float()
                    if self.train_data[k].ndim > 2:
                        self.train_data[k] = self.train_data[k].flatten(start_dim = 1)
                    if k not in self.normalizers:
                        self.normalizers[k] = SafeLimitsNormalizer(self.train_data[k])
                    self.train_data[k] = self.train_data[k][self.obs_indices]
            self.normalizers[k].to_device(self.device)

        self.transform_data_structure()

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

    def generate_action_sequence(self):
        assert self.train_data["action"].shape[-1] % 17 == 0, (
            "Expected action dimension to be divisible by 17 "
            "(3-dim pos, 6-dim ori, 7-dim joint, 1-dim gripper)."
        )
        num_robots = self.train_data["action"].shape[-1] // 17
        robot_actions = []

        for n in range(num_robots):
            gripper = self.train_data["action"][..., 17 * n + 16 : 17 * n + 17][self.action_indices]

            if "pose" in self.action_space:
                pos = self.train_data["action"][..., 17 * n : 17 * n + 3][self.action_indices]
                ori = self.train_data["action"][..., 17 * n + 3 : 17 * n + 9][self.action_indices]

                if "abs" in self.action_space:
                    pass

                elif "rel" in self.action_space:
                    pos -= self.train_data[f"robot{n}_eef_pos"].unsqueeze(1)
                    ori = roma.quat_composition([
                        roma.quat_conjugation(
                            self.train_data[f"robot{n}_eef_quat"]
                            .unsqueeze(1).repeat(1, self.pred_horizon, 1)
                        ),
                        roma.rotmat_to_unitquat(rotation_6d_to_matrix(ori)),
                    ])
                    ori = matrix_to_rotation_6d(roma.unitquat_to_rotmat(ori))

                elif "del" in self.action_space:
                    pos -= self.train_data[f"robot{n}_eef_pos"][self.action_indices]
                    ori = roma.quat_composition([
                        roma.quat_conjugation(
                            self.train_data[f"robot{n}_eef_quat"][self.action_indices]
                        ),
                        roma.rotmat_to_unitquat(rotation_6d_to_matrix(ori)),
                    ])
                    ori = matrix_to_rotation_6d(roma.unitquat_to_rotmat(ori))
                else:
                    raise ValueError(f"{self.action_space} is not recognized. Recognized action spaces are abs_ee_pose, rel_ee_pose, del_ee_pose, abs_joint_pos, rel_joint_pos, or del_joint_pos.")

                robot_actions.append(torch.cat((pos, ori, gripper), dim=-1))

            else:
                joint = self.train_data["action"][..., 17 * n + 9 : 17 * n + 16][self.action_indices]
                if "abs" in self.action_space:
                    pass

                elif "rel" in self.action_space:                
                    joint -= self.train_data[f"robot{n}_joint_pos"].unsqueeze(1)

                elif "del" in self.action_space:
                    joint -= self.train_data[f"robot{n}_joint_pos"][self.action_indices]
                else:
                    raise ValueError(f"{self.action_space} is not recognized. Recognized action spaces are abs_ee_pose, rel_ee_pose, del_ee_pose, abs_joint_pos, rel_joint_pos, or del_joint_pos.")

                robot_actions.append(torch.cat((joint, gripper), dim=-1))

        return torch.cat(robot_actions, dim=-1).to(self.device).float()

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