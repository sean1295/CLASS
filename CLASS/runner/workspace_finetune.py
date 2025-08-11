import glob
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from CLASS.util.common import conditional_resolver
from CLASS.envs import create_env

# Environment setup
num_cores = 1 
torch.set_num_threads(num_cores)
torch.backends.cudnn.benchmark = True
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("if", conditional_resolver, replace=True)

def find_policy_path(cfg):
    """
    Find the most recently saved policy file based on datetime in filename.
    
    Args:
        cfg: Configuration object with base_dir attribute
        
    Returns:
        str: Path to the most recent policy file, or None if no files found
    """
    # Pattern to match your policy files: YYYYMMDD_HHMMSS_success_XX.pth
    if cfg.dynamic_mode:
        pattern = os.path.join(cfg.base_dir, "*_pretrain_dynamic_success_*.pth")
    else:
        pattern = os.path.join(cfg.base_dir, "*_pretrain_static_success_*.pth")
    
    # Get all matching policy files
    policy_files = glob.glob(pattern)
    
    if not policy_files:
        print(f"No policy files found in {cfg.base_dir}")
        return None
    
    # Extract datetime from each filename and find the most recent
    latest_file = None
    latest_datetime = None
    
    for filepath in policy_files:
        filename = os.path.basename(filepath)
        
        # Extract datetime string (first part before first underscore after date)
        # Format: YYYYMMDD_HHMMSS_success_XX.pth
        try:
            datetime_str = "_".join(filename.split("_")[:2])  # Gets YYYYMMDD_HHMMSS
            file_datetime = datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")
            
            if latest_datetime is None or file_datetime > latest_datetime:
                latest_datetime = file_datetime
                latest_file = filepath
                
        except ValueError:
            # Skip files that don't match the expected datetime format
            print(f"Skipping file with unexpected format: {filename}")
            continue
    
    if latest_file:
        print(f"Found most recent policy: {latest_file}")
        return latest_file
    else:
        print("No valid policy files found")
        return None

def identity_collate(batch):
    return batch[0]

class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg

    def update_config(self):
        new_obs_keys = []
        for key in self.cfg.obs_keys:
            # Use dynamic image observation when dynamic_mode is true
            if "agentview" in key and self.cfg.dynamic_mode:
                new_obs_keys.append("dynamic_" + key)
            else:
                new_obs_keys.append(key)

        self.cfg.obs_keys = new_obs_keys
        self.cfg.policy.obs_keys = new_obs_keys
        self.cfg.dataset.obs_keys = new_obs_keys
        self.cfg.policy.model.image_encoder.views = [key for key in new_obs_keys if "image" in key]

    def run(self):
        assert torch.cuda.is_available(), "CUDA is not available"
        if self.cfg.finetune.enabled:
            wandb_name = self.cfg.wandb.name + "_finetune"
        else:
            wandb_name = self.cfg.wandb.name + "_train"
        if self.cfg.dynamic_mode:
            wandb_name += "_dynamic"
        else:
            wandb_name += "_static"
        wandb.init(project=self.cfg.wandb.project, name=wandb_name, config=OmegaConf.to_container(self.cfg, resolve=True))     
        self.update_config() 

        policy = instantiate(self.cfg.policy, _recursive_=False)        
        train_dataset = instantiate(self.cfg.dataset)  
        
        if self.cfg.finetune.enabled:
            if self.cfg.finetune.pretrain_policy_path is None:
                self.cfg.finetune.pretrain_policy_path = find_policy_path(self.cfg)
            if self.cfg.finetune.pretrain_policy_path is not None:
                policy.load(self.cfg.finetune.pretrain_policy_path)
            else:
                raise FileNotFoundError(f"Cannot finetune: no valid policy path found in {self.cfg.dataset_dir}")
        else:
            policy.set_normalizers(train_dataset.normalizers)
            
        train_bs = int(self.cfg.finetune.train_bs)
        self.cfg.finetune.train_steps = int(np.ceil(len(train_dataset) * self.cfg.finetune.num_epoch / train_bs))
        optimizer, scheduler = policy.get_optimizer_scheduler(self.cfg.finetune)

        print("Start Training...") 
        train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=self.cfg.finetune.train_steps * train_bs)
        train_loader = DataLoader(train_dataset, batch_size=train_bs, sampler=train_sampler)
        train_iterator = iter(train_loader)
        
        env = create_env(self.cfg)               

        pbar = tqdm(range(self.cfg.finetune.train_steps), mininterval=1)
        eval_steps = self.cfg.finetune.eval_steps or [int(ratio * self.cfg.finetune.train_steps)-1 for ratio in np.arange(0.1, 1.1, 0.1)]

        for step in pbar:
            batch = next(train_iterator)
            loss = policy.train_model_step(batch, 
                                           optimizer, 
                                           scheduler,
                                           )
            wandb.log({"train/bc_loss": loss}, step=step)
            if step in eval_steps:
                single_env_cfg = self.cfg.copy()
                single_env_cfg.num_envs = 1
                single_env = create_env(single_env_cfg)
                policy.save_to_ema_model()

                nonparam_results_dict = single_env.nonparam_evaluate(
                        policy,
                        train_dataset,
                        action_horizon=self.cfg.eval.nonparam_horizon,
                        nnn=self.cfg.eval.nnn,
                        temperature=self.cfg.eval.temperature,
                        use_cossim=self.cfg.eval.use_cossim,
                        num_ep=self.cfg.eval.num_ep,
                        max_episode_steps=self.cfg.max_episode_steps,
                        seed_base=self.cfg.eval.seed,
                        render=self.cfg.eval.render,
                        pbar=pbar
                    )

                param_results_dict = env.param_evaluate(
                    policy,
                    action_horizon=self.cfg.eval.param_horizon,
                    num_ep=self.cfg.eval.num_ep,
                    max_episode_steps=self.cfg.max_episode_steps,
                    seed_base=self.cfg.eval.seed,
                    render=self.cfg.eval.render,
                    pbar=pbar
                )

                nonparam_video, param_video = None, None
                if self.cfg.eval.render:
                    nonparam_video = wandb.Video(nonparam_results_dict["imgs"], fps=30, format="mp4")
                    param_video = wandb.Video(param_results_dict["imgs"], fps=30, format="mp4")

                def safe_mean(arr):
                    return np.mean(arr) if len(arr) > 0 else 0

                def safe_max(arr):
                    return np.max(arr, initial=0)

                wandb.log({
                    "eval/finetune_success_rate_nonparam": nonparam_results_dict["Success Rate"],
                    "eval/finetune_rollout_video_nonparam": nonparam_video,    
                    "eval/finetune_avg_completion_dur_nonparam": safe_mean(nonparam_results_dict["Completion Times"]),
                    "eval/finetune_max_completion_dur_nonparam": safe_max(nonparam_results_dict["Completion Times"]),
                    "eval/finetune_success_rate_param": param_results_dict["Success Rate"],
                    "eval/finetune_rollout_video_param": param_video,
                    "eval/finetune_avg_completion_dur_param": safe_mean(param_results_dict["Completion Times"]),
                    "eval/finetune_max_completion_dur_param": safe_max(param_results_dict["Completion Times"]),
                    }, step=step)

                if self.cfg.save_model:                        
                    os.makedirs(self.cfg.base_dir, exist_ok=True)
                    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    policy_filename = f"{current_time_str}_finetune_success_{int(100 * param_results_dict['Success Rate'])}.pth"
                    policy_filepath = self.cfg.base_dir + "/" + policy_filename
                    policy.save(policy_filepath)
        env.close()
