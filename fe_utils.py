import glob
import os

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from calculator import WBMCalculator
from fairchem.core.datasets.lmdb_dataset import data_list_collater
# for ESEN
from fairchem.core.preprocessing import AtomsToGraphs


def load_a2g_func(cfg_path: str, name: str) -> callable:
    config: DictConfig = OmegaConf.load(cfg_path)
    if name == "esen":
        a2g_config = config.data_module.train_dataset.config.a2g_args
        a2g_config.r_energy = False
        a2g_config.r_forces = False
        a2g_config.r_stress = False

        def a2g_func(atoms):
            data_object = AtomsToGraphs(**a2g_config).convert(atoms)
            data_object.natoms = len(atoms)
            
            # Collate after dtype conversion
            data_object = data_list_collater(
                [data_object], otf_graph=config.data_module.otf_graph)

            return data_object
    else:
        raise NotImplementedError

    return a2g_func


def load_model(
    cfg_path,
    name,
    ckpt_path=None,
    load_ema=True,
    require_grad=False,
    device="cuda",
) -> pl.LightningModule:
    # instantiate model from config file
    config: DictConfig = OmegaConf.load(cfg_path)
    model = instantiate(config.lightning_module.regressor)

    # load checkpoint
    if ckpt_path is not None:
        # prepare state dict
        ckpt_info = torch.load(ckpt_path)
        state_dict = ckpt_info['state_dict']
        if load_ema:
            if name == "orb":
                ema_state_dict = ckpt_info['ema_shadow_params']
                for k, v in state_dict.items():
                    if 'bn.running_' in k:  # ema params donot have running mean and running var of BN
                        assert k not in ema_state_dict
                        print(f'add {k} to ema state dict')
                        ema_state_dict[k] = v
                state_dict = ema_state_dict
            elif name == "esen" or name == "uma":
                ema_state_dict = ckpt_info['ema_shadow_params']
                for k, v in state_dict.items():
                    if k not in ema_state_dict:  # ema params donot have running mean and running var of BN
                        print(f'add {k} to ema state dict')
                        ema_state_dict[k] = v
                state_dict = ema_state_dict

        # modify the state dict key
        state_dict = {
            k.replace('regressor.', '', 1): state_dict[k] for k in state_dict
        }
        model.load_state_dict(state_dict, strict=True)
    model = model.eval().to(device)

    if not require_grad:
        for param in model.parameters():
            param.requires_grad = False

    model = torch.compile(model, dynamic=True)

    return model


def load_calculator(
    cfg_path: str,
    ckpt_path: str = None,
    name: str = "esen",
    direct_forces: bool = False,
    load_ema: bool = True,
    device: str = "cuda",
    divide_stress_by: float = 1.0,
):
    assert name == "esen", "only esen is supported for now"
    
    # load the model
    model = load_model(
        cfg_path=cfg_path,
        name=name,
        ckpt_path=ckpt_path,
        device=device,
        load_ema=load_ema,
        require_grad=not direct_forces,
    )

    # load the calculator
    calculator = WBMCalculator(
        model=model,
        a2g_func=load_a2g_func(cfg_path, name=name),
        device=device,
        divide_stress_by=divide_stress_by,
    )

    return calculator


def load_model_by_epoch(
    name: str,
    ckpt_folder: str,
    epoch: str,
    load_ema: bool = True,
    require_grad: bool = False,
    device: str = "cuda",
) -> pl.LightningModule:
    cfg_path = os.path.join(ckpt_folder, 'config.yaml')
    ckpt_path = find_checkpoint_by_epoch(ckpt_folder, epoch)
    print(f"loading checkpoint from {ckpt_path}")
    return load_model(
        cfg_path=cfg_path,
        name=name,
        ckpt_path=ckpt_path,
        device=device,
        load_ema=load_ema,
        require_grad=require_grad,
    )


def find_checkpoint_by_epoch(ckpt_folder: str, epoch: str):
    path_pattern = os.path.join(
        ckpt_folder, f"**/epoch={epoch}-*.ckpt")
    files = glob.glob(
        path_pattern,
        recursive=True
    )
    assert len(files) == 1, (
        f'Expect 1 checkpoint, but find {len(files)}\n'
        f'ckpt_folder: {ckpt_folder}\n'
        f'epoch: {epoch}\n'
        f'path_pattern: {path_pattern}\n'
        f'{files}\n'
    )
    return files[0]


def load_calculator_by_epoch(
    name: str,
    ckpt_folder: str,
    epoch: str,
    device: str = "cuda",
    direct_forces: bool = False,
    load_ema: bool = True,
    divide_stress_by: float = 1.0,
):
    cfg_path = os.path.join(ckpt_folder, 'config.yaml')
    ckpt_path = find_checkpoint_by_epoch(ckpt_folder, epoch)
    print(f"loading checkpoint from {ckpt_path}")
    return load_calculator(
        cfg_path=cfg_path,
        ckpt_path=ckpt_path,
        name=name,
        device=device,
        direct_forces=direct_forces,
        load_ema=load_ema,
        divide_stress_by=divide_stress_by,
    )


def read_config(ckpt_folder: str):
    cfg_path = os.path.join(ckpt_folder, 'config.yaml')
    return OmegaConf.load(cfg_path)