import os
import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from CLASS.util.common import conditional_resolver
from CLASS.envs import create_env
from CLASS.dataset.CLASS import make_class_dataset_from_target

# Environment setup
num_cores = 1 
torch.set_num_threads(num_cores)
torch.backends.cudnn.benchmark = True
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("if", conditional_resolver, replace=True)

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
        wandb_name = self.cfg.wandb.name + "_pretrain"
        if self.cfg.dynamic_mode:
            wandb_name += "_dynamic"
        else:
            wandb_name += "_static"
        wandb.init(project=self.cfg.wandb.project, name=wandb_name, config=OmegaConf.to_container(self.cfg, resolve=True))     
        self.update_config()
        
        policy = instantiate(self.cfg.policy, _recursive_=False)

        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
        dataset_target = cfg_dict["dataset"].pop("_target_")
        cl_dataset = make_class_dataset_from_target(dataset_target, use_sparse=self.cfg.use_sparse)
        pretrain_bs = int(self.cfg.pretrain.train_bs)
        pretrain_dataset = cl_dataset(
            num_sample=pretrain_bs,
            dtw_file_path=self.cfg.dtw_file_path,
            dist_quantile=self.cfg.dist_quantile,
            **cfg_dict["dataset"]
        )
        policy.set_normalizers(pretrain_dataset.normalizers)
        
        self.cfg.pretrain.train_steps = int(np.ceil(len(pretrain_dataset) * self.cfg.pretrain.num_epoch / pretrain_bs))
        self.cfg.pretrain.scheduler.num_warmup_steps = min(500, int(self.cfg.pretrain.train_steps / self.cfg.pretrain.num_epoch * 5)) # 5 epochs
        optimizer, scheduler = policy.get_optimizer_scheduler(self.cfg.pretrain)

        train_sampler = RandomSampler(pretrain_dataset, replacement=True, num_samples=self.cfg.pretrain.train_steps * pretrain_bs)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=1, sampler=train_sampler, collate_fn=identity_collate) 
        pretrain_iterator = iter(pretrain_loader)

        print(f"Start Pretraining... Warmping up for {self.cfg.pretrain.scheduler.num_warmup_steps} steps.")
        pbar = tqdm(range(self.cfg.pretrain.train_steps), mininterval=1)
        #evaluate 10 times throughout training starting from the second half of training.
        eval_steps = self.cfg.pretrain.eval_steps or [int(ratio * self.cfg.pretrain.train_steps)-1 for ratio in np.arange(0.55, 1.05, 0.05)] 

        for step in pbar:
            batch = next(pretrain_iterator)
            loss = policy.pretrain_model_step(batch, 
                                            optimizer, 
                                            scheduler,
                                            self.cfg.temperature,
                                            )
            wandb.log({"train/class_loss": loss}, step=step)  
            if step in eval_steps:
                single_env_cfg = self.cfg.copy()
                single_env_cfg.num_envs = 1
                single_env = create_env(single_env_cfg)
                policy.save_to_ema_model()
                nonparam_results_dict = single_env.nonparam_evaluate(
                    policy,
                    pretrain_dataset,
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
                if self.cfg.eval.render:
                    video = wandb.Video(nonparam_results_dict["imgs"], fps=30, format="mp4")

                def safe_mean(arr):
                    return np.mean(arr) if len(arr) > 0 else 0

                def safe_max(arr):
                    return np.max(arr, initial=0)

                wandb.log({
                    "eval/pretrain_success_rate_nonparam": nonparam_results_dict["Success Rate"],
                    "eval/pretrain_avg_completion_dur_nonparam": safe_mean(nonparam_results_dict["Completion Times"]),
                    "eval/pretrain_max_completion_dur_nonparam": safe_max(nonparam_results_dict["Completion Times"]),
                    "eval/pretrain_rollout_video_nonparam": video
                    }, step=step)
                
                if self.cfg.save_model:                        
                    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    if self.cfg.dynamic_mode:
                        policy_filename = f"{current_time_str}_pretrain_dynamic_success_{int(100 * nonparam_results_dict['Success Rate'])}.pth"
                    else:
                        policy_filename = f"{current_time_str}_pretrain_static_success_{int(100 * nonparam_results_dict['Success Rate'])}.pth"
                    policy_filepath = self.cfg.base_dir + "/" + policy_filename
                    policy.save(policy_filepath)             