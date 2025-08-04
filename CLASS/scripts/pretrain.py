import warnings
import sys

warnings.showwarning = lambda *args, **kwargs: None
sys.modules['warnings'].warn = lambda *args, **kwargs: None

import os
import argparse
from hydra import initialize, compose

from CLASS.runner.workspace_pretrain import Workspace

os.environ["MUJOCO_GL"] = "egl"
os.environ["HYDRA_FULL_ERROR"] = "2"
os.environ["WANDB_DIR"] = "/scratch/dcs3zc"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="square_ph_dp_abs", help="Name of config file in configs/")
    args = parser.parse_args()

    with initialize(version_base=None, config_path="../configs/"):
        cfg = compose(config_name=args.config_name)

    workspace = Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()   