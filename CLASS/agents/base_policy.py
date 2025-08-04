import copy
from abc import ABC
from typing import Dict

import torch
import torch.nn as nn
from torch.nn import functional as F
from diffusers.training_utils import EMAModel
from hydra.utils import instantiate, get_method
from omegaconf import DictConfig, OmegaConf
from diffusers.optimization import get_scheduler

from CLASS.util.augmentation import center_transform, crop_transform


class Policy(ABC):
    def __init__(
        self,
        obs_keys,
        proprio_dim,
        latent_dim,
        action_dim,
        obs_horizon,
        pred_horizon,
        model,
        vision_model,
        frozen_encoder,
        spatial_softmax,
        num_kp,
        device,
    ):
        self.device = device
        self.views = [key for key in obs_keys if "image" in key]
        self.proprio_dim = proprio_dim
        self.latent_dim = latent_dim
        self.vision_model = vision_model
        self.spatial_softmax = spatial_softmax
        
        if num_kp > 0 and spatial_softmax:
            self.image_feat_dim = num_kp
        else:
            self.image_feat_dim = 512
            
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        
        if self.latent_dim:
            self.obs_dim = self.obs_horizon * (
                self.image_feat_dim * (1 + int(self.spatial_softmax)) * len(self.views) + self.latent_dim
            )
        else:
            self.obs_dim = self.obs_horizon * (
                self.image_feat_dim * (1 + int(self.spatial_softmax)) * len(self.views) + self.proprio_dim
            )
            
        self.frozen_encoder = frozen_encoder
        self.obs_keys = sorted(obs_keys)
        self.image_keys = sorted([key for key in self.obs_keys if "image" in key])
        self.non_image_keys = sorted([key for key in self.obs_keys if "image" not in key])
        
        self.train_transform = crop_transform(vision_model=vision_model)
        self.eval_transform = center_transform(vision_model=vision_model)
        
        self.build_model(model_config=model)
        self.normalizers = None

    def get_optimizer_scheduler(self, cfg):
        optim_cfg = OmegaConf.to_container(cfg.optim, resolve=True)
        vision_lr = optim_cfg.pop("vision_lr")
        target = optim_cfg.pop("_target_")

        vision_params = list(self.model["model"]["image_encoder"].parameters())
        other_params = [
            p for name, p in self.model["model"].named_parameters()
            if not name.startswith("image_encoder") and p.requires_grad
        ]
        
        param_groups = [
            {"params": vision_params, "lr": vision_lr},
            {"params": other_params, "lr": optim_cfg["lr"]},
        ]

        optimizer_cls = get_method(target)
        optimizer = optimizer_cls(param_groups, **optim_cfg)

        scheduler = get_scheduler(
            **cfg.scheduler,
            optimizer=optimizer,
            num_training_steps=cfg.train_steps
        )

        return optimizer, scheduler

    def set_normalizers(self, normalizers: Dict):
        self.normalizers = copy.deepcopy(normalizers)
        for key in self.normalizers.keys():
            self.normalizers[key].to_device(self.device)

    def build_model(self, model_config: DictConfig = None, recursive=True):
        model_config.policy_head.global_cond_dim = self.obs_dim

        # Set Identity if no proprio output
        if model_config.proprio_encoder.out_features == 0:
            model_config.proprio_encoder._target_ = "torch.nn.Identity"

        # Instantiate modules
        modules = {
            name: instantiate(subcfg, _recursive_=recursive)
            for name, subcfg in model_config.items()
            if not name.startswith("_")
        }

        model = nn.ModuleDict(modules).to(self.device)

        self.model = dict()
        self.model["model"] = model
        
        if self.frozen_encoder:
            self.model["model"]["image_encoder"].requires_grad_(False)

        self.model["ema"] = EMAModel(parameters=self.model['model'].parameters(), power=0.75)
        self.model["ema_model"] = copy.deepcopy(self.model['model'])
        self.model["ema_model"].eval()
        self.model["ema_model"].requires_grad_(False)

    def process_batch(self, batch: Dict[str, torch.Tensor]):
        for key in batch.keys():
            batch[key] = batch[key].to(self.device).float()
        return batch

    def pretrain_model_step(self, batch, optimizer, scheduler, temperature=0.1):
        optimizer.zero_grad()

        img_feats = {}
        for key in self.image_keys:
            if key in batch:
                B, T = batch[key].shape[:2]
                flat = batch[key].to(self.device).float().flatten(end_dim=1)
                transformed = self.train_transform(flat).to(memory_format=torch.channels_last)
                img_feats[key] = transformed.view(B, T, *transformed.shape[1:])

        img_feats = self.encode_image(img_feats, mode="train").flatten(start_dim=1)
        img_feats = F.normalize(img_feats, dim=1)
        sim_matrix = torch.div(torch.matmul(img_feats, img_feats.T), temperature)
        self_mask = torch.eye(B, device=self.device, dtype=torch.bool)
        dist_weights = batch["dist"].to(self.device)

        with torch.no_grad():
            logits_max, _ = torch.max(sim_matrix.masked_fill(self_mask, -float('inf')), dim=1, keepdim=True)

        scaled_sim_stable = sim_matrix - logits_max
        log_denom = torch.logsumexp(scaled_sim_stable.masked_fill(self_mask, float('-inf')), dim=1, keepdim=True)
        log_prob = scaled_sim_stable - log_denom
        denom = dist_weights.sum(dim=1)
        valid_samples_mask = denom > 1e-6

        if not valid_samples_mask.any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        numerator = (dist_weights[valid_samples_mask] * log_prob[valid_samples_mask]).sum(dim=1)
        loss = -(numerator / denom[valid_samples_mask]).mean()

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        self.ema_step()

        return loss

    def train_model_step(self, batch, optimizer, scheduler):
        optimizer.zero_grad()
        obs_batch = self.encode_obs(batch, mode="train")
        action_batch = self.normalize_action(batch['action'])

        loss = self.compute_train_loss(obs_batch, action_batch)
        nn.utils.clip_grad_norm_(self.model["model"].parameters(), 1.0)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        self.ema_step()

        return loss

    def compute_train_loss(self, obs_batch, action_batch):
        action_pred = self.model["model"]["policy_head"](obs_batch)
        loss = F.mse_loss(action_batch, action_pred, reduction="mean")
        return loss

    @torch.no_grad()
    def validate_model(self):
        pass

    def ema_step(self):
        self.model['ema'].step(self.model['model'].parameters())

    def save_to_ema_model(self):
        self.model["ema"].copy_to(self.model["ema_model"].parameters())

    def encode_image(self, obs, mode="train"):
        """Encode image observations (assumes images are already transformed)."""
        if mode == "train":
            model = self.model["model"]
        elif mode == "eval":
            model = self.model["ema_model"]
        else:
            raise ValueError("Mode must be 'train' or 'eval'.")
            
        img_feats = []

        # Only process image keys that are present in the batch
        present_image_keys = [key for key in self.image_keys if key in obs]

        for key in present_image_keys:
            img_feat = obs[key]
            shape = img_feat.shape[:-3]
            if len(shape) > 1:  # 5+D image
                img_feat = img_feat.flatten(end_dim=-4)
            img_feats.append(
                model["image_encoder"](img_feat.to(memory_format=torch.channels_last), key).reshape(*shape, -1)
            )

        if img_feats:
            img_feats = torch.cat(img_feats, dim=-1)
            return img_feats

        return None

    def encode_nproprio(self, obs, mode="train"):
        """Encode proprioceptive observations (assumes features are already normalized)."""
        if mode == "train":
            model = self.model["model"]
        elif mode == "eval":
            model = self.model["ema_model"]
        else:
            raise ValueError("Mode must be 'train' or 'eval'.")
            
        proprio_feats = []

        # Only process non-image keys that are present in the batch
        present_non_image_keys = [key for key in self.non_image_keys if key in obs]

        for key in present_non_image_keys:
            proprio_feat = obs[key]
            proprio_feats.append(proprio_feat)

        if proprio_feats:
            proprio_feats = torch.cat(proprio_feats, dim=-1)
            proprio_feats = model["proprio_encoder"](proprio_feats)
            return proprio_feats

        return None

    def encode_nobs(self, obs, mode="train"):
        """Encode observations by combining image and proprioceptive features."""
        if mode == "train":
            model = self.model["model"]
        elif mode == "eval":
            model = self.model["ema_model"]
        else:
            raise ValueError("Mode must be 'train' or 'eval'.")

        # Encode image and proprioceptive features separately
        img_feats = self.encode_image(obs, mode=mode)
        proprio_feats = self.encode_nproprio(obs, mode=mode)

        # Fuse the features
        nobs = model["fusion_fn"](img_feats, proprio_feats)

        return nobs

    def encode_obs(self, obs, mode="train"):
        """Preprocess observations (normalize + transform) then encode them."""
        processed_obs = {}

        # Normalize non-image keys
        for key in self.non_image_keys:
            if key in obs:
                processed_obs[key] = (
                    self.normalizers[key].normalize(obs[key].to(self.device)).float()
                )

        # Transform image keys
        transform = self.train_transform if mode == "train" else self.eval_transform
        for key in self.image_keys:
            if key in obs:
                processed_obs[key] = transform(obs[key].to(self.device).float())

        return self.encode_nobs(processed_obs, mode=mode)

    def normalize_action(self, action):
        naction = self.normalizers[f"action"].normalize(action.to(self.device))
        return naction.float()

    def unnormalize_action(self, naction):
        action = self.normalizers[f"action"].unnormalize(naction.to(self.device))
        return action.float()

    def get_naction(self, nobs, *args, **kwargs):
        naction = self.model["ema_model"]["policy_head"](nobs)
        return naction

    @torch.no_grad()
    def get_action(self, obs, *args, **kwargs):
        nobs = self.encode_obs(obs, mode="eval")
        is_batched = nobs.ndim == 3
        if not is_batched:
            nobs = nobs.unsqueeze_(0)
        assert nobs.ndim == 3
        naction = self.get_naction(nobs, *args, **kwargs)
        if is_batched:
            action = self.unnormalize_action(naction)
        else:
            action = self.unnormalize_action(naction)[0]

        return action.float()

    def reset(self):
        pass

    def get_latents(self, dataset, batch_size=256):
        latent_data = []
        with torch.no_grad():
            for i in range(0, len(dataset.train_data), batch_size):
                batch_dict = dataset.train_data[i : i + batch_size]
                batch = {
                    k: torch.stack([sample[k] for sample in batch_dict if k in sample]).to(self.device)
                    for k in self.obs_keys
                }
                latents = self.encode_obs(batch, mode="eval")
                latent_data.append(latents)

        return torch.cat(latent_data, dim=0)

    def save(self, filename):
        load_dict = {
            "ema_model": self.model["ema_model"].state_dict(),
            "model": self.model["model"].state_dict(),
            "normalizers": self.normalizers,
            "obs_horizon": self.obs_horizon,
            "pred_horizon": self.pred_horizon,
        }
        torch.save(load_dict, filename)

    def load(self, filename, verbose=False):
        state_dict_loaded = torch.load(filename, map_location=self.device, weights_only=False)
        flag = False

        def get_unique_prefixes(string_list, num_parts=3):
            """
            Extract unique prefixes from a list of dot-separated strings.
            
            Args:
                string_list: List of strings to process
                num_parts: Number of parts to include in prefix (default: 3)
            
            Returns:
                List of unique prefixes
            """
            unique_prefixes = set()

            for item in string_list:
                parts = item.split('.')
                if len(parts) >= num_parts:
                    prefix = '.'.join(parts[:num_parts])
                    unique_prefixes.add(prefix)

            return sorted(list(unique_prefixes))

        def safe_load_state_dict(target_model, source_state_dict, verbose=verbose):
            model_state = target_model.state_dict()

            # Separate compatible and incompatible keys
            compatible_keys = {}
            skipped_keys = []
            shape_mismatches = []

            for k, v in source_state_dict.items():
                if k in model_state:
                    if v.shape == model_state[k].shape:
                        compatible_keys[k] = v
                    else:
                        shape_mismatches.append(f"{k}: saved={v.shape} vs current={model_state[k].shape}")
                else:
                    skipped_keys.append(k)

            missing_keys, unexpected_keys = target_model.load_state_dict(compatible_keys, strict=False)

            # Print detailed information
            if verbose:
                if skipped_keys:
                    print(f"Keys not in current model: {get_unique_prefixes(skipped_keys, 3)}")
                if shape_mismatches:
                    print(f"Shape mismatches: {get_unique_prefixes(shape_mismatches, 3)}")
                if missing_keys:
                    print(f"Missing keys in saved state: {get_unique_prefixes(missing_keys, 3)}")
                if unexpected_keys:
                    print(f"Unexpected keys (this shouldn't happen): {get_unique_prefixes(unexpected_keys, 3)}")

            flag = skipped_keys or shape_mismatches or missing_keys or unexpected_keys
            return flag

        if "model" in state_dict_loaded:
            flag = flag or safe_load_state_dict(self.model["model"], state_dict_loaded["model"], verbose=verbose)
        if "ema_model" in state_dict_loaded:
            flag = safe_load_state_dict(self.model["ema_model"], state_dict_loaded["ema_model"], verbose=False)

        self.model["ema"] = EMAModel(parameters=self.model['model'].parameters(), power=0.75)
        
        normalizers = state_dict_loaded.get("normalizers")
        if normalizers is not None:
            self.normalizers = normalizers
            
        obs_horizon = state_dict_loaded.get("obs_horizon")
        if obs_horizon is not None:
            self.obs_horizon = obs_horizon
            
        pred_horizon = state_dict_loaded.get("pred_horizon")
        if pred_horizon is not None:
            self.pred_horizon = pred_horizon

        if flag:
            print("Partially loaded model (with compatible weights).")
        else:
            print("Fully loaded model!")