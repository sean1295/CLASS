import argparse

import numpy as np
import copy
import h5py
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from collections import defaultdict
from tqdm.auto import tqdm
import torch
import zarr
from hydra import initialize, compose

from CLASS.util.robosuite_utils import GymWrapper
from CLASS.envs.robomimic import DynamicCameraWrapper
from CLASS.util.rotations import matrix_to_rotation_6d


class RobomimicDataCollector:
    def __init__(self, dataset_path):
        # default BC config
        dummy_spec = dict(
            obs=dict(
                low_dim=["robot0_eef_pos", "object"],
                rgb=["agentview_image", "robot0_eye_in_hand_image"]
            ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)    
        env_meta["env_kwargs"]["camera_names"] = ['agentview', 'robot0_eye_in_hand']
        env_meta["env_kwargs"]["camera_heights"] = 256
        env_meta["env_kwargs"]["camera_widths"] = 256
        
        abs_env_meta = copy.deepcopy(env_meta)
        joint_env_meta = copy.deepcopy(env_meta)
        
        abs_env_meta['env_kwargs']['controller_configs']['control_delta'] = False

        # Create base environments for action conversion
        base_env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=True, 
        )
        
        base_abs_env = EnvUtils.create_env_from_metadata(
            env_meta=abs_env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )

        joint_env_meta['env_kwargs']['controller_configs']['type'] = 'JOINT_POSITION'
        joint_env_meta['env_kwargs']['controller_configs']['input_max'] = np.pi
        joint_env_meta['env_kwargs']['controller_configs']['input_min'] = -np.pi
        joint_env_meta['env_kwargs']['controller_configs']['output_max'] = np.pi
        joint_env_meta['env_kwargs']['controller_configs']['output_min'] = -np.pi
        
        base_joint_env = EnvUtils.create_env_from_metadata(
            env_meta=joint_env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )

        # Create observation keys list
        obs_keys = [
            'agentview_image', 'robot0_eye_in_hand_image',
            'object-state', 'robot0_joint_pos', 'robot0_joint_pos_cos', 
            'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 
            'robot0_eef_quat', 'robot0_eef_vel_lin', 'robot0_eef_vel_ang', 
            'robot0_gripper_qpos', 'robot0_gripper_qvel'
        ]
        
        # Create static camera environment
        static_env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False, 
            render_offscreen=True,
            use_image_obs=True, 
        )
        static_env = GymWrapper(static_env.env, keys=obs_keys, flatten_obs=False) 
        self.static_env = DynamicCameraWrapper(static_env, dynamic=False)
        
        # Create dynamic camera environment
        dynamic_env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False, 
            render_offscreen=True,
            use_image_obs=True, 
        )
        dynamic_env = GymWrapper(dynamic_env.env, keys=obs_keys, flatten_obs=False) 
        self.dynamic_env = DynamicCameraWrapper(dynamic_env, dynamic=True)
        
        self.env = base_env
        self.abs_env = base_abs_env
        self.joint_env = base_joint_env
        
        # Disable hard resets
        # self.env.env.hard_reset = False
        # self.abs_env.env.hard_reset = False
        # self.joint_env.env.hard_reset = False
        # self.static_env.env.hard_reset = False
        # self.dynamic_env.env.hard_reset = False
        
        # Initialize all environments
        self.env.reset()
        self.abs_env.reset()
        self.joint_env.reset()
        self.static_env.reset()
        self.dynamic_env.reset()
        
        self.file = h5py.File(dataset_path, 'r')
        self.obs_keys = obs_keys
    
    def __len__(self):
        return len(self.file['data'])

    def convert_actions(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Given state and delta action sequence
        generate equivalent goal position and orientation for each step
        keep the original gripper action intact.
        """
        # in case of multi robot
        # reshape (N,14) to (N,2,7)
        # or (N,7) to (N,1,7)
        stacked_actions = actions.reshape(*actions.shape[:-1],-1,7)
        env = self.env
        joint_env = self.joint_env        
        # generate abs actions
        action_goal_pos = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_goal_ori = np.zeros(
            stacked_actions.shape[:-1]+(3,3), 
            dtype=stacked_actions.dtype)
        action_goal_qpos = np.zeros(
            stacked_actions.shape[:-1]+(7,), 
            dtype=stacked_actions.dtype)
        action_gripper = stacked_actions[...,[-1]]
        
        for i in range(len(states)):
            _ = env.reset_to({'states': states[i]})
            _ = joint_env.reset_to({'states': states[i]})
            # taken from robot_env.py L#454
            for idx, robot in enumerate(env.env.robots):
                # run controller goal generator
                robot.control(stacked_actions[i,idx], policy_step=True)
                joint_robot = joint_env.env.robots[idx]
                
                # read pos and ori from robots
                controller = robot.controller
                joint_controller = joint_robot.controller
                action_goal_pos[i,idx] = controller.goal_pos
                action_goal_ori[i,idx] = controller.goal_ori
                
                torques = controller.torques
                desired_torque = np.linalg.solve(joint_controller.mass_matrix, torques - joint_controller.torque_compensation)
                joint_pos = np.array(controller.sim.data.qpos[controller.qpos_index])
                joint_vel = np.array(controller.sim.data.qvel[controller.qvel_index])
                position_error = (desired_torque + np.multiply(joint_vel, joint_controller.kd)) / joint_controller.kp
                desired_qpos =  position_error + joint_pos
                action_goal_qpos[i,idx] = desired_qpos
                
        action_goal_pos = torch.from_numpy(action_goal_pos)
        action_goal_ori = torch.from_numpy(action_goal_ori)
        action_goal_qpos = torch.from_numpy(action_goal_qpos)
        action_goal_ori = matrix_to_rotation_6d(action_goal_ori)
        action_gripper = torch.from_numpy(action_gripper)
        
        stacked_abs_actions = torch.cat([
            action_goal_pos,
            action_goal_ori,
            action_goal_qpos,
            action_gripper
        ], dim=-1)
        abs_actions = stacked_abs_actions.reshape(-1, 17 * len(env.env.robots))
        
        return abs_actions

    def collect_episode_data(self, demo_num):
        """
        Collect all data for a single episode from scratch:
        - Images from static and dynamic cameras
        - State-based observations
        - Converted absolute actions
        """
        demo = self.file[f'data/demo_{demo_num}']
        states = demo['states'][:]
        actions = demo['actions'][:]
        
        episode_data = defaultdict(list)
        
        # Reset environments
        self.env.reset()
        self.static_env.reset(seed = demo_num)
        self.dynamic_env.reset(seed = demo_num)
        print(f"Collecting episode {demo_num} with {len(states)} steps...")
        for i, (state, action) in enumerate(zip(states, actions)):
            # Set simulation state for both environments
            self.env.env.sim.set_state_from_flattened(state)
            self.static_env.sim.set_state_from_flattened(state)
            self.dynamic_env.sim.set_state_from_flattened(state)

            self.env.env.sim.forward()
            self.static_env.env.sim.forward()
            self.dynamic_env.env.sim.forward()
            
            # Get observations from both environments at current state
            obs = self.env.env._get_observations(force_update=True)
            static_obs = self.static_env.env._get_observations(force_update=True)
            dynamic_obs = self.dynamic_env.env._get_observations(force_update=True)
            
            # After collecting observations, step the dynamic environment to update camera
            # This prepares the dynamic camera for the next timestep
            if i < len(actions) - 1:  # Don't step on the last action
                self.env.step(action)
                self.static_env.step(action)
                self.dynamic_env.step(action)
            # Collect all observation types
            for key in self.obs_keys:
                if key in static_obs:
                    if 'image' in key:
                        if key == 'agentview_image':
                            # Collect both static and dynamic versions of agentview
                            episode_data['static_agentview_image'].append(static_obs[key].copy())
                            episode_data['dynamic_agentview_image'].append(dynamic_obs[key].copy())                        
                        # Also keep original key for compatibility
                        episode_data[key].append(obs[key])
                    else:
                        # State-based observations (same for both environments)
                        episode_data[key].append(obs[key])
        # Convert actions
        abs_actions = self.convert_actions(states, actions)
        episode_data['action'] = abs_actions.numpy()
        
        # Convert lists to numpy arrays
        for key, data_list in episode_data.items():
            if key != 'action':  # action is already converted
                episode_data[key] = np.array(data_list)
                
                # Transpose image data to (T, C, H, W) format
                if 'image' in key and len(episode_data[key].shape) == 4:
                    episode_data[key] = np.transpose(episode_data[key], (0, 3, 1, 2))
                    # Flip images if needed
                    episode_data[key] = np.flip(episode_data[key], axis=2)
        
        return episode_data


def collect_robomimic_dataset(hdf5_path, zarr_path, max_episodes=None):
    """
    Collect entire dataset from scratch using only HDF5 for simulation states
    """

    collector = RobomimicDataCollector(hdf5_path)
    
    # Get demo list
    f = h5py.File(hdf5_path, "r")
    demos = list(f["data"].keys())    
    # Sort by the numeric part after "demo_"
    demos = sorted(demos, key=lambda x: int(x.split('_')[1]))
    if max_episodes:
        demos = demos[:max_episodes]
    
    # Collect all episode data
    all_data = defaultdict(list)
    episode_indices = []
    total_steps = 0
    
    print(f"Collecting {len(demos)} episodes...")
    for i, demo in tqdm(enumerate(demos)):
        episode_data = collector.collect_episode_data(i)
        
        # Add episode data to overall collection
        for key, data in episode_data.items():
            all_data[key].append(data)
        
        # Track episode boundaries
        episode_length = len(episode_data['action'])
        episode_indices.extend([i] * episode_length)
        total_steps += episode_length
    
    # Concatenate all data
    print("Concatenating data...")
    final_data = {}
    for key, data_list in all_data.items():
        final_data[key] = np.concatenate(data_list, axis=0)
        print(f"{key}: {final_data[key].shape}")
    
    # Add episode and index information
    final_data['episode'] = np.array(episode_indices)
    final_data['index'] = np.arange(total_steps)
    
    # Create Zarr dataset
    print("Creating Zarr dataset...")
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store, overwrite=True)
    
    # Save all data to Zarr
    for key, data in final_data.items():
        # Determine appropriate chunk size
        if 'image' in key:
            chunks = (min(1000, total_steps), *data.shape[1:])
        else:
            chunks = (min(10000, total_steps), *data.shape[1:])
        
        # Handle first action fix
        if key == 'action' and len(data) > 1:
            data[0] = data[1]
        
        # Create dataset
        root.create_dataset(
            f"data/{key}",
            data=data,
            chunks=chunks,
            dtype=data.dtype,
            overwrite=True,
        )
    
    # Add metadata
    root.attrs['total_samples'] = total_steps
    root.attrs['n_episodes'] = len(demos)
    root.attrs['collected_from_scratch'] = True
    
    print(f" Successfully collected and saved {total_steps} samples from {len(demos)} episodes to {zarr_path}")
    
    f.close()
    store.close()
    return root

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="square_ph", help="Name of config file in configs/")
    args = parser.parse_args()
    print(args.config_name)
    with initialize(version_base=None, config_path="../configs/"):
        cfg = compose(config_name=args.config_name)

    if "mimicgen" in cfg.env_name:
        try:
            import mimicgen   
        except:
            raise ImportError(
                "mimicgen is not installed. "
            )

    zarr_dataset = collect_robomimic_dataset(cfg.ori_dataset_path, cfg.dataset.dataset_path, max_episodes=None)

if __name__ == "__main__":
    main()