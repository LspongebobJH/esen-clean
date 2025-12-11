# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Add the project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from common.globals import MODELS_PROJECT_ROOT
from common.config import Config
from main import main

logger = logging.getLogger(__name__)

@hydra.main(
    config_path=str(MODELS_PROJECT_ROOT / "configs"), 
    config_name="default", 
    version_base="1.1"
)
def run(cfg: DictConfig):
    '''
    Run with both .yaml config and CLI
    '''
    # save target project code to hydra.run.dir
    hydra_dir = Path(os.getcwd())  #  hydra.run.dir
    project_dir = Path(hydra.utils.get_original_cwd())
    items_to_copy = [
        project_dir / "common",
        project_dir / "fairchem", 
        project_dir / "orb_models",
        project_dir / "pl_datamodule",
        project_dir / "pl_module", 
        project_dir / "calculator.py",
        project_dir / "fe_utils.py",
    ]
    codes_dir = hydra_dir / "codes"
    copy_to_hydra_dir(items_to_copy, codes_dir)

    torch.set_float32_matmul_precision("high")

    argv = sys.argv[1:]
    # Parse command line arguments
    parser = argparse.ArgumentParser(allow_abbrev=False)  # prevent prefix matching issues
    parser.add_argument("--seed", type=int, default=42)
    args, config_changes = parser.parse_known_args(argv)

    # Make merged config options
    # CLI options take priority over YAML file options
    conf_cli = OmegaConf.from_cli(config_changes)
    schema = OmegaConf.structured(Config)
    OmegaConf.set_struct(cfg, False)
    config = OmegaConf.merge(schema, cfg, conf_cli) # schema < cfg < conf_cli
    OmegaConf.set_readonly(config, True)  # should not be written t
        
    main(config, seed=args.seed)


def copy_to_hydra_dir(items_to_copy, target_dir):
    """
    Copy specified items to the target directory.
    """
    os.makedirs(target_dir, exist_ok=True)
    for item in items_to_copy:
        if os.path.exists(item):
            if os.path.isdir(item):
                shutil.copytree(item, os.path.join(target_dir, os.path.basename(item)), dirs_exist_ok=True)
            else:
                shutil.copy2(item, target_dir)
        else:
            print(f"Warning: {item} does not exist and will not be copied.")


if __name__ == "__main__":
    run()