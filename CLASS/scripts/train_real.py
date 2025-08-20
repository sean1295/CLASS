import warnings
import sys
import os
import argparse
from hydra import initialize, compose

warnings.showwarning = lambda *args, **kwargs: None
sys.modules['warnings'].warn = lambda *args, **kwargs: None
os.environ["HYDRA_FULL_ERROR"] = "2"

from CLASS.runner.workspace_train_real import Workspace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="grasp_cup_dp_force", help="Name of config file in configs/")
    args = parser.parse_args()

    with initialize(version_base=None, config_path="../configs/"):
        cfg = compose(config_name=args.config_name)

    workspace = Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()  