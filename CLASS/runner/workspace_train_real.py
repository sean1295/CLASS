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

# Environment setup
num_cores = 1 
torch.set_num_threads(num_cores)
torch.backends.cudnn.benchmark = True
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("if", conditional_resolver, replace=True)


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        assert torch.cuda.is_available(), "CUDA is not available"
        wandb_name = self.cfg.wandb.name + "_train_real"

        wandb.init(project=self.cfg.wandb.project, name=wandb_name, config=OmegaConf.to_container(self.cfg, resolve=True))     

        policy = instantiate(self.cfg.policy, _recursive_=False)        
        train_dataset = instantiate(self.cfg.dataset)  
        
        policy.set_normalizers(train_dataset.normalizers)
            
        train_bs = int(self.cfg.finetune.train_bs)
        self.cfg.finetune.train_steps = int(np.ceil(len(train_dataset) * self.cfg.finetune.num_epoch / train_bs))
        optimizer, scheduler = policy.get_optimizer_scheduler(self.cfg.finetune)

        print("Start Training...") 
        train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=self.cfg.finetune.train_steps * train_bs)
        train_loader = DataLoader(train_dataset, batch_size=train_bs, sampler=train_sampler)
        train_iterator = iter(train_loader)
        
        pbar = tqdm(range(self.cfg.finetune.train_steps), mininterval=1)
        eval_steps = self.cfg.finetune.eval_steps or [int(ratio * self.cfg.finetune.train_steps)-1 for ratio in np.arange(0.1, 1.1, 0.1)]

        for step in pbar:
            batch = next(train_iterator)
            loss = policy.train_model_step(batch, 
                                           optimizer, 
                                           scheduler,
                                           )
            wandb.log({"train/bc_loss": loss}, step=step)
            if (step+1) % 5000 == 0:    
                policy.save_to_ema_model()                               
                policy_filename = f"step_{step}_{self.cfg.action_space}.pth"
                policy_filepath = self.cfg.dataset_dir + "/" + policy_filename
                policy.save(policy_filepath)
                print(f"policy saved to {policy_filepath}")
