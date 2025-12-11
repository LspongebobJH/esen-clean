# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Note that importing this module has two side effects:
1. It sets the environment variable `PROJECT_ROOT` to the root of the explorers project.
2. It registers a new resolver for OmegaConf, `eval`, which allows us to use `eval` in our config files.
"""
import os
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache
def get_pyg_device() -> torch.device:
    """
    Some operations of pyg don't work on MPS, so fall back to CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


MODELS_PROJECT_ROOT = Path(__file__).resolve().parents[1]
print(f"MODELS_PROJECT_ROOT: {MODELS_PROJECT_ROOT}")

# Set environment variable PROJECT_ROOT so that hydra / OmegaConf can access it.
os.environ["PROJECT_ROOT"] = str(MODELS_PROJECT_ROOT)  # for hydra