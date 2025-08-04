import torch
from hydra.utils import instantiate
from torch.nn import functional as F

from CLASS.agents.base_policy import Policy

class DiffusionPolicy(Policy):
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
        noise_scheduler,
        num_inference_steps,
        device,
    ):
        super().__init__(
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
        )
        self.num_inference_steps = num_inference_steps
        self.noise_scheduler = instantiate(noise_scheduler)
        self.noise_scheduler.set_timesteps(num_inference_steps)

    def compute_train_loss(self, obs_batch, action_batch):
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (action_batch.shape[0],),
            device=self.device,
        ).long()

        noise = torch.randn_like(action_batch, device=self.device)
        noisy_actions = self.noise_scheduler.add_noise(action_batch, noise, timesteps)
        noise_pred = self.model["model"]["policy_head"](noisy_actions, timesteps, obs_batch)

        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(action_batch, noise, timesteps)
        elif prediction_type == "sample":
            target = action_batch
        else:
            raise TypeError("Prediction type not recognized.")

        loss = F.mse_loss(noise_pred, target, reduction="mean")
        return loss

    @torch.no_grad()
    def get_naction(self, nobs, store_output=False, extra_steps=0, initial_noise=None):
        assert nobs.ndim == 3

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        infer_model = self.model["ema_model"]

        naction = initial_noise if initial_noise is not None else torch.randn(
            (nobs.shape[0], self.pred_horizon, self.action_dim), device=self.device
        )

        if store_output:
            outputs = [naction.clone()]

        for k in self.noise_scheduler.timesteps:
            noise_pred = infer_model["policy_head"](naction, k, nobs)
            naction = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            )[0]

            if store_output:
                outputs.append(naction.clone())

        for _ in range(extra_steps):
            noise_pred = infer_model["policy_head"](naction, k, nobs)
            variance_noise = torch.randn_like(noise_pred)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction,
                variance_noise=variance_noise,
            )[0]

            if store_output:
                outputs.append(naction.clone())

        if store_output:
            outputs = torch.cat(outputs)
            return naction, outputs
        return naction