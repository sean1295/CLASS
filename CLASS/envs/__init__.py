import gymnasium as gym
import robomimic.utils.env_utils as EnvUtils
import multiprocessing

from CLASS.util.common import suppress_output
from CLASS.util.robosuite_utils import GymWrapper
from CLASS.envs.robomimic import get_env_meta, RobomimicTorchWrapper, DynamicCameraWrapper


@suppress_output()
def create_single_env(cfg):    

    if "mimic" in cfg.env_name:
        if "gen" in cfg.env_name:
            import mimicgen   
            
        env_meta = get_env_meta(cfg)
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=True,
        )
        obs_keys = [
            (
                key.split("dynamic_")[-1]
                if "dynamic_" in key
                else (
                    key.split("static_")[-1]
                    if "static_" in key
                    else "object-state" if key == "object" else key
                )
            )
            for key in cfg.obs_keys
        ]
        env = GymWrapper(env.env, keys=obs_keys, flatten_obs=False, is_dynamic=cfg.dynamic_mode)
        env.render = lambda: env.sim.render(
            height=256, width=256, camera_name="agentview"
        )
        env.check_success = env.env._check_success
        env.env.hard_reset = False
        if cfg.dynamic_mode:
            env = DynamicCameraWrapper(env, dynamic=True)
    else:
        raise ValueError(f"Unsupported env_name: '{cfg.env_name}'. Expected 'mimic' to be part of the environment name.")

    return env

def create_env(cfg):
    if cfg.num_envs == 1:
        env = create_single_env(cfg)
    elif cfg.num_envs > 1:
        multiprocessing.set_start_method("spawn", force=True)
        env = gym.vector.AsyncVectorEnv(
            [lambda: create_single_env(cfg) for _ in range(cfg.num_envs)]
        )
    else:
        ValueError(
            f"The number of environments {cfg.num_envs} should be higher than 1."
        )
    env = RobomimicTorchWrapper(
        env, action_space=cfg.action_space, device=cfg.device
    )
    return env
