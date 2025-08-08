import collections
import json
import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import h5py
import numpy as np
import robomimic.utils.obs_utils as ObsUtils
import robosuite.utils.transform_utils as T
import roma
import torch
import torch.nn.functional as F
from gymnasium import Wrapper
from scipy.spatial.transform import Rotation as R

from CLASS.util.rotations import rotation_6d_to_matrix
from CLASS.util.search_utils import retrieve_nearest_neighbors
from CLASS.util.visualization import create_image_grid


def get_env_meta(cfg):
    f = h5py.File(cfg.ori_dataset_path, "r")
    obs_keys = [
        (
            key.split("dynamic_")[-1]
            if "dynamic_" in key
            else (
                key.split("static_")[-1]
                if "static_" in key
                else "object-state"
                if key == "object"
                else key
            )
        )
        for key in cfg.obs_keys
    ]

    camera_names = [
        obs_key.split("_image")[0] for obs_key in obs_keys if "_image" in obs_key
    ]

    dummy_spec = dict(
        obs=dict(
            low_dim=["robot0_eef_pos", "object"],
            rgb=[f"{camera_name}_image" for camera_name in camera_names]
            + [f"{camera_name}" for camera_name in camera_names],
        ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
    env_meta = json.loads(f["data"].attrs["env_args"])
    env_meta["env_kwargs"]["camera_names"] = camera_names
    env_meta["env_kwargs"]["camera_heights"] = 256
    env_meta["env_kwargs"]["camera_widths"] = 256
    env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False

    if "joint_pos" in cfg.action_space:
        env_meta["env_kwargs"]["controller_configs"]["type"] = "JOINT_POSITION"
        env_meta["env_kwargs"]["controller_configs"]["input_max"] = np.pi
        env_meta["env_kwargs"]["controller_configs"]["input_min"] = -np.pi
        env_meta["env_kwargs"]["controller_configs"]["output_max"] = np.pi
        env_meta["env_kwargs"]["controller_configs"]["output_min"] = -np.pi

    return env_meta

    
class RobomimicTorchWrapper(Wrapper):
    def __init__(self, env, action_space="abs_ee_pose", device="cuda", num_robots=1):
        super().__init__(env)
        self.action_space = action_space
        self.num_robots = num_robots
        self.device = device
        self.last_obs = None

    def reset(self, *args, **kwargs):
        obs_np, info = self.env.reset(*args, **kwargs)
        self.update_last_obs(obs_np)
        return self.last_obs, info

    def reset_to(self, state):
        self.env.sim.set_state_from_flattened(state)
        self.env.sim.forward()
        obs_np = self.env._get_observations(force_update=True)
        self.update_last_obs(obs_np)
        return self.last_obs

    def update_last_obs(self, obs):
        obs_torch = self.numpy_to_torch(obs)
        if "object-state" in obs_torch:
            obs_torch["object"] = obs_torch["object-state"]
        self.last_obs = obs_torch

    def numpy_to_torch(self, np_input):
        if isinstance(np_input, dict):
            out = {k: torch.from_numpy(v).to(self.device) for k, v in np_input.items()}
            for k in out:
                if "image" in k:
                    img = out[k].flip(dims=(-3,))
                    if img.ndim == 4:
                        img = img.permute(0, 3, 1, 2)
                    elif img.ndim == 3:
                        img = img.permute(2, 0, 1)
                    out[k] = img
        else:
            out = torch.as_tensor(np_input).to(self.device)
        return out

    def step(self, action):
        processed_action = self.process_action(action)
        obs_np, rew, done, _, info = self.env.step(processed_action.to("cpu").numpy())
        self.update_last_obs(obs_np)
        return self.last_obs, self.numpy_to_torch(rew), self.numpy_to_torch(done), False, info

    def process_action_sequence(self, action_seq):
        action_seq = action_seq.clone().to(self.device)
        processed_action_sequence = []
        is_batched = len(action_seq.shape) == 3

        for n in range(self.num_robots):
            if "pose" in self.action_space:
                pos = action_seq[..., 10 * n + 0 : 10 * n + 3]
                quat = roma.rotmat_to_unitquat(rotation_6d_to_matrix(action_seq[..., 10 * n + 3 : 10 * n + 9]))
                grip = action_seq[..., 10 * n + 9 : 10 * n + 10]

                if "rel" in self.action_space:
                    pos_obs = self.last_obs[f"robot{n}_eef_pos"]
                    quat_obs = self.last_obs[f"robot{n}_eef_quat"]

                    if is_batched:
                        if len(pos_obs.shape) == 2:
                            pos_obs = pos_obs.unsqueeze(1)
                            quat_obs = quat_obs.unsqueeze(1)
                        pos_obs = pos_obs.expand(-1, action_seq.shape[1], -1)
                        quat_obs = quat_obs.expand(-1, action_seq.shape[1], -1)
                    else:
                        if len(pos_obs.shape) == 1:
                            pos_obs = pos_obs.unsqueeze(0)
                            quat_obs = quat_obs.unsqueeze(0)

                    pos += pos_obs
                    quat = roma.quat_composition([quat_obs.float(), quat.float()])

                processed_action_per_robot = torch.cat((pos, quat, grip), dim=-1)
            else:
                joint = action_seq[..., 8 * n + 0 : 8 * n + 7]
                grip = action_seq[..., 8 * n + 7 : 8 * n + 8]

                if "rel" in self.action_space:
                    joint_obs = self.last_obs[f"robot{n}_joint_pos"]
                    if is_batched:
                        if len(joint_obs.shape) == 2:
                            joint_obs = joint_obs.unsqueeze(1)
                        joint_obs = joint_obs.expand(-1, action_seq.shape[1], -1)
                    else:
                        if len(joint_obs.shape) == 1:
                            joint_obs = joint_obs.unsqueeze(0)
                    joint += joint_obs

                processed_action_per_robot = torch.cat((joint, grip), dim=-1)

            processed_action_sequence.append(processed_action_per_robot)

        return torch.cat(processed_action_sequence, dim=-1)

    def process_action(self, action):
        processed_action = []
        is_batched = len(action.shape) > 1

        for n in range(self.num_robots):
            if "pose" in self.action_space:
                shape = (*action.shape[:-1], 7) if is_batched else (7,)
                a_robot = torch.empty(shape, device=action.device)

                if "del" in self.action_space:
                    pos = action[..., 8 * n : 8 * n + 3]
                    ori = action[..., 8 * n + 3 : 8 * n + 7]
                    pos_obs = self.last_obs[f"robot{n}_eef_pos"]
                    quat_obs = self.last_obs[f"robot{n}_eef_quat"]

                    a_robot[..., :3] = pos + pos_obs
                    rvec1 = roma.unitquat_to_rotvec(quat_obs.float())
                    rvec2 = roma.unitquat_to_rotvec(ori.float())
                    a_robot[..., 3:6] = -roma.rotvec_composition([rvec1, rvec2])
                    a_robot[..., 6:7] = action[..., 8 * n + 7 : 8 * n + 8]
                else:
                    a_robot[..., :3] = action[..., 8 * n : 8 * n + 3]
                    a_robot[..., 3:6] = roma.unitquat_to_rotvec(action[..., 8 * n + 3 : 8 * n + 7].float())
                    a_robot[..., 6:7] = action[..., 8 * n + 7 : 8 * n + 8]
            else:
                shape = (*action.shape[:-1], 8) if is_batched else (8,)
                a_robot = torch.empty(shape, device=action.device)

                if "del" in self.action_space:
                    joint_obs = self.last_obs[f"robot{n}_joint_pos"]
                    a_robot[..., :7] = action[..., 8 * n : 8 * n + 7] + joint_obs
                else:
                    a_robot[..., :7] = action[..., 8 * n : 8 * n + 7]

                a_robot[..., -1:] = action[..., 8 * n + 7 : 8 * n + 8]

            processed_action.append(a_robot)

        return torch.cat(processed_action, dim=-1)

    def param_evaluate(
            self, policy, num_ep, action_horizon, max_episode_steps, latency=0, action_repeat=0, render=False, seed_base=4444, pbar=None, *args, **kwargs
        ):
        # multiprocessing.set_start_method("spawn", force=True)
        comp_times = []
        num_env = self.env.num_envs
        if render:
            imgs = []
        num_iter = num_ep // num_env
        num_ep = num_iter * num_env
        img_key = None
        if policy.image_keys:
            img_key = policy.image_keys[0]

        for iter_idx in range(num_iter):
            policy.reset()
            obs, info = self.reset(
                seed=[seed_base + ep_idx + num_env * iter_idx for ep_idx in range(num_env)]
            )
            dones, t = [False] * num_env, 0
            obs_deque = collections.deque(
                [obs.copy()] * (policy.obs_horizon + latency),
                maxlen=policy.obs_horizon + latency
            )
            
            if render:
                last_complete_imgs = {}  # {env_idx: last_image}

            while t < max_episode_steps and not all(dones):
                if pbar:
                    pbar.set_description(
                        f"Evaluation loop {iter_idx}/{num_iter}, t = {t}: {len(comp_times)}/{(iter_idx + 1) * num_env}"
                    )
                obs_seq = {
                    k: torch.stack([obs_deque[i][k] for i in range(policy.obs_horizon)]).swapaxes(0, 1).float()
                    for k in obs
                }
                action_seq = policy.get_action(obs_seq, *args, **kwargs)
                actions = self.process_action_sequence(action_seq)
                actions = actions[:, policy.obs_horizon - 1 :, :].swapaxes(0, 1)

                for k, action in enumerate(actions[:action_horizon]):
                    obs, _, _, _, info = self.step(action.float())
                    for _ in range(action_repeat):
                        obs, _, _, _, _ = self.step(action.float())                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        done = np.stack(self.unwrapped.call("check_success"))
                    
                    # Store completion images when envs complete
                    if render:
                        img = obs[img_key].cpu().numpy() if img_key else np.stack(self.unwrapped.call("render"))[:, ::-1].transpose(2, 0, 1)
                        
                    for i, d in enumerate(done):
                        if d and not dones[i]:
                            dones[i] = True
                            comp_times.append(t)
                            if render:
                                last_complete_imgs[i] = img[i].copy()  # Store the completion image
                    
                    obs_deque.append(obs.copy())
                    t += 1
                    
                    if render:
                        # Override completed env images with their last complete image
                        for env_idx, complete_img in last_complete_imgs.items():
                            img[env_idx] = complete_img
                        
                        # Create completion status array for border colors
                        completion_status = [dones[i] for i in range(len(img))]
                        
                        # Create grid with colored borders
                        imgs.append(create_image_grid(
                            img, 
                            n=int(len(img) ** 0.5),
                            completion_status=completion_status,
                        ))
        
        if render:
            imgs = np.stack(imgs)
        else:
            imgs = None

        return {"Success Rate": len(comp_times) / num_ep, 
                "Completion Times": comp_times, 
                "imgs": imgs}

    def nonparam_evaluate(
        self, policy, dataset, num_ep, action_horizon, max_episode_steps, option=0, nnn=64, temperature=0.01, latency=0, action_repeat=0, use_cossim=True, batch_size=256,
        render=False, seed_base=4444, pbar=None, *args, **kwargs
    ):
        retrieval_latent = policy.get_latents(dataset, batch_size)[:, -1]
        actions = torch.stack(
            [dataset.train_data[i]["action"] for i in range(len(dataset))]
        )[:, dataset.obs_horizon - 1:].to(self.device).float()
        assert action_horizon <= actions.shape[1], f"action_horizon {action_horizon} exceeds available action sequence length"

        comp_times = []
        if render:
            imgs = []
        img_key = None
        if policy.image_keys:
            img_key = policy.image_keys[0]

        for ep_idx in range(num_ep):
            policy.reset()
            obs, _ = self.reset(seed=seed_base + ep_idx)
            done, truncated, t = False, False, 0
            obs_deque = collections.deque(
                [obs.copy()] * (policy.obs_horizon + latency),
                maxlen=policy.obs_horizon + latency
            )
            prev_indices = torch.tensor([-1] * nnn).to(self.device)
            tolerance = 3

            while not done and t < max_episode_steps:
                if pbar:
                    pbar.set_description(f"Evaluation: t = {t}, {len(comp_times)}/{ep_idx}")

                obs_seq = {
                    k: torch.stack([obs_deque[i][k] for i in range(policy.obs_horizon)]).float()
                    for k in obs
                }
                obs_latent = policy.encode_obs(obs_seq, mode="eval")[-1]
                indices, scores = retrieve_nearest_neighbors(
                    obs_latent, retrieval_latent, n=nnn, use_cossim=use_cossim
                )
                dists = scores[indices]

                if all(prev_indices == indices):
                    tolerance -= 1
                else:
                    tolerance = 3
                if tolerance == 0:
                    break
                
                weights = F.softmax(-dists / temperature, dim=0)
                
                if option == 0:
                    weighted_actions = torch.einsum("i,i...->...", weights.to(actions), actions[indices])
                else:
                    weighted_actions = actions[indices][torch.multinomial(weights, num_samples=1)[0]]
                
                weighted_actions = self.process_action_sequence(weighted_actions)

                for i in range(action_horizon):
                    obs, _, _, truncated, _ = self.step(weighted_actions[i])
                    for _ in range(action_repeat):
                        obs, _, _, truncated, _ = self.step(weighted_actions[i])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        done = self.check_success()
                    obs_deque.append(obs.copy())
                    t += 1
                    if render:
                        img = obs[img_key].cpu().numpy() if img_key else self.render()[::-1].transpose(2, 0, 1)
                        imgs.append(img)
                    if done or truncated:
                        break
                if done:
                    comp_times.append(t)                    
                prev_indices = indices.clone()

        if render:
            imgs = np.stack(imgs, axis = 0)
        else:
            imgs = None

        return {"Success Rate": len(comp_times) / num_ep, 
                "Completion Times": comp_times, 
                "imgs": imgs}


class DynamicCameraWrapper(Wrapper):
    def __init__(self, env, dynamic=False):
        super().__init__(env)
        self.mover_body_name = "agentview_cameramover"
        xml = self.env.sim.model.get_xml()
        self.xml = self.modify_xml_for_camera_movement(xml=xml, camera_name="agentview")
        self.env.reset_from_xml_string(self.xml)
        self.env.sim.reset()
        self.target = np.array([0.0, 0, 1.0])  # Target point to look at
        self.distance = 0.9 # Distance from camera to target
        self.dynamic = dynamic
        if self.dynamic:
            self.set_direction()

    def reset(self, *args, **kwargs):
        if self.dynamic:
            self.set_direction()
        _, info = super().reset(*args, **kwargs)
        state = self.env.sim.get_state()
        self.env.reset_from_xml_string(self.xml)
        self.env.sim.set_state(state)
        self.init_camera()
        obs = self.env._get_observations(force_update=True)
        post_processed_obs = self.post_process_obs(obs)
        return post_processed_obs, info

    def set_direction(self):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-0.01, 0.3)
        self.direction = np.array([x, y])

    def init_camera(self):
        self.init_camera_pos, self.init_camera_quat = self.initial_camera_pose()
        while self.init_camera_pos[0] < self.target[0] + 0.4 or self.init_camera_pos[2] < self.target[2] + 0.1:
            self.init_camera_pos, self.init_camera_quat = self.initial_camera_pose()
        self.set_camera(self.init_camera_pos, self.init_camera_quat)

    def post_process_obs(self, obs):
        if self.dynamic and "agentview_image" in obs:
            obs["dynamic_agentview_image"] = obs["agentview_image"]
            
        return obs

    def step(self, action):
        if self.dynamic:
            current_pos = self.env.sim.data.get_mocap_pos(self.mover_body_name)
            current_quat = T.convert_quat(
                self.env.sim.data.get_mocap_quat(self.mover_body_name), to="xyzw"
            )
            new_pos, new_quat = self.update_camera_pose(
                current_pos, current_quat, self.direction, move_amount=0.0015
            )
            self.set_camera(new_pos, new_quat)

        obs, reward, terminated, truncated, info = super().step(action)
        post_processed_obs = self.post_process_obs(obs)
        return post_processed_obs, reward, terminated, truncated, info

    def set_camera(self, pos, quat):
        self.env.sim.data.set_mocap_pos(self.mover_body_name, pos)
        self.env.sim.data.set_mocap_quat(
            self.mover_body_name, T.convert_quat(quat, to="wxyz")
        )
        self.env.sim.forward()

    def modify_xml_for_camera_movement(self, xml, camera_name):
        tree = ET.fromstring(xml)
        wb = tree.find("worldbody")

        camera_elem = None
        cameras = wb.findall("camera")
        for camera in cameras:
            if camera.get("name") == camera_name:
                camera_elem = camera
                break
        assert camera_elem is not None

        mocap = ET.SubElement(wb, "body")
        mocap.set("name", self.mover_body_name)
        mocap.set("mocap", "true")
        mocap.set("pos", camera.get("pos"))
        mocap.set("quat", camera.get("quat"))
        new_camera = ET.SubElement(mocap, "camera")
        new_camera.set("mode", "fixed")
        new_camera.set("name", camera.get("name"))
        new_camera.set("pos", "0 0 0")

        wb.remove(camera_elem)
        return ET.tostring(tree, encoding="utf8").decode("utf8")

    def initial_camera_pose(self):
        theta = 2 * np.pi * np.random.random()
        phi_deviation = np.pi / 6
        phi = np.pi / 2 + np.random.uniform(-phi_deviation, phi_deviation)

        x = self.distance * np.sin(phi) * np.cos(theta)
        y = self.distance * np.sin(phi) * np.sin(theta)
        z = self.distance * np.cos(phi) + self.target[2]
        pos = np.array([x, y, z])

        direction = self.target - pos
        direction /= np.linalg.norm(direction)

        world_up = np.array([0, 0, 1])
        right = np.cross(direction, world_up)
        if np.allclose(right, 0):
            right = np.array([1, 0, 0])
        right /= np.linalg.norm(right)
        camera_up = np.cross(right, direction)
        camera_up /= np.linalg.norm(camera_up)

        rotation_matrix = np.column_stack((right, camera_up, -direction))
        r = R.from_matrix(rotation_matrix)
        quat = r.as_quat()
        return pos, quat

    def update_camera_pose(self, current_pos, current_quat, direction_vector, move_amount=0.01):
        radial = current_pos - self.target
        radial /= np.linalg.norm(radial)

        world_up = np.array([0, 0, 1])
        tangent1 = np.cross(world_up, radial)
        if np.allclose(tangent1, 0):
            tangent1 = np.array([1, 0, 0])
        tangent1 /= np.linalg.norm(tangent1)
        tangent2 = np.cross(radial, tangent1)
        tangent2 /= np.linalg.norm(tangent2)

        move_dir = direction_vector[0] * tangent1 + direction_vector[1] * tangent2
        move_dir /= np.linalg.norm(move_dir)

        rotation_axis = np.cross(radial, move_dir)
        rotation = R.from_rotvec(rotation_axis * move_amount)

        new_pos_vector = rotation.apply(radial) * self.distance + self.target
        new_pos = new_pos_vector

        new_direction = self.target - new_pos
        new_direction /= np.linalg.norm(new_direction)

        right = np.cross(new_direction, world_up)
        if np.allclose(right, 0):
            right = np.array([1, 0, 0])
        right /= np.linalg.norm(right)
        camera_up = np.cross(right, new_direction)
        camera_up /= np.linalg.norm(camera_up)

        rotation_matrix = np.column_stack((right, camera_up, -new_direction))
        r = R.from_matrix(rotation_matrix)
        new_quat = r.as_quat()
        return new_pos, new_quat