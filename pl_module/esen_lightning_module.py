# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import (Any, Dict, Generic, Optional, Protocol, Sequence, TypeVar,
                    Union)

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import AdamW, Optimizer
from torch_geometric.data import Batch

T = TypeVar("T", bound=Batch) # T must be BatchedData or its children


class OptimizerPartial(Protocol):
    """Callable to instantiate an optimizer."""

    def __call__(self, params: Any) -> Optimizer:
        raise NotImplementedError


class SchedulerPartial(Protocol):
    """Callable to instantiate a learning rate scheduler."""

    def __call__(self, optimizer: Optimizer) -> Any:
        raise NotImplementedError


def get_default_optimizer(params):
    return AdamW(params=params, lr=1e-4, weight_decay=0, amsgrad=True)


class EsenLightningModule(pl.LightningModule):
    """LightningModule for instantiating and training a GraphRegressor"""

    def __init__(
        self,
        regressor,
        # dens parameters
        use_denoising_pos: bool = False,
        denoising_pos_params: Optional[dict] = None,
        # optimizer and scheduler partials
        optimizer_partial: Optional[OptimizerPartial] = None,
        scheduler_partial: Optional[Dict[str, Union[Any, SchedulerPartial]]] = None,
        pretrained_ckpt: Optional[str] = None,
        load_energy_head: bool = False,
    ):
        """_summary_

        Args:
            graphregressor: The regression model to predict efs.
            optimizer_partial: Used to instantiate optimizer.
            scheduler_partials: used to instantiate learning rate schedulers

            pretrained_ckpt: path to the pretrained ckpt
            load_energy_head: whether to load energy head from `pretrained_ckpt`
        """
        super().__init__()
        optimizer_partial = optimizer_partial or get_default_optimizer
        self.save_hyperparameters(
            ignore=("optimizer_partial", "scheduler_partials", "regressor")
        )

        self.regressor = regressor

        self.use_denoising_pos = use_denoising_pos
        self.denoising_pos_params = DenoisingPosParams(**denoising_pos_params) \
            if denoising_pos_params is not None else None
        assert not (self.use_denoising_pos and self.denoising_pos_params is None), \
            "If use_denoising_pos is True, denoising_pos_params must be provided."

        self._optimizer_partial = optimizer_partial
        self._scheduler_partial = scheduler_partial
        self.load_energy_head = load_energy_head

        if pretrained_ckpt is not None:
            self.load_pretrained_ckpt(pretrained_ckpt)

    def load_pretrained_ckpt(self, ckpt_path):
        ckpt_info = torch.load(ckpt_path, map_location=self.device)

        print(f'load state dict from {ckpt_path}')
        state_dict = ckpt_info['state_dict']
        
        # direct load from pre-trained ckpt
        updated_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('regressor.', '', 1)
            if ("energy_head" in new_k and not self.load_energy_head) or \
                "forces_head" in new_k or \
                "stress_head" in new_k:
                continue

            # only load the first 5 blocks
            # if re.search(r'\.blocks\.[5-9]', new_k):
            #     continue
            
            updated_state_dict[new_k] = v
        missing_keys, unexpected_keys = self.regressor.load_state_dict(
            updated_state_dict, 
            strict=False
        )
        
        print(f'missing keys: {missing_keys}')
        print(f'unexpected keys: {unexpected_keys}')
        # print('\n\n\n' + '#'*20 + '\n\n\n')
        # exit()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: Optional[str] = None,
        **kwargs,
    ):
        """Load model from checkpoint. kwargs are passed to hydra's instantiate and can override
        arguments from the checkpoint config."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # The config should have been saved in the checkpoint by AddConfigCallback in run.py
        config = checkpoint["config"]
        try:
            lightning_module = instantiate(config.lightning_module, **kwargs)
        except InstantiationException as e:
            print("Could not instantiate model from the checkpoint.")
            print(
                "If the error is due to an unexpected argument because the checkpoint and the code have diverged, try using load_from_checkpoint_and_config instead."
            )
            raise e
        assert isinstance(lightning_module, cls)

        # Restore state of the BFNLightningModule.
        lightning_module.load_state_dict(checkpoint["state_dict"])
        return lightning_module

    def configure_optimizers(self) -> Any:
        optimizer = self._optimizer_partial(params=self.regressor.parameters())
        if self._scheduler_partial is not None:
            lr_schedulers = [
                {
                    **self._scheduler_partial,
                    "scheduler": self._scheduler_partial["scheduler"](
                        optimizer=optimizer,
                    ),
                }
            ]

            return [
                optimizer,
            ], lr_schedulers
        else:
            return optimizer

    def training_step(self, train_batch: T, batch_idx: int) -> STEP_OUTPUT:
        # print(f'batch_idx = {batch_idx}, natoms = {torch.sum(train_batch["natoms"])}, bs={len(train_batch["natoms"])}')
        if self.use_denoising_pos and np.random.rand() < self.denoising_pos_params.prob:
            if self.denoising_pos_params.fixed_noise_std:
                train_batch = add_gaussian_noise_to_position(
                    train_batch,
                    std=self.denoising_pos_params.std,
                    corrupt_ratio=self.denoising_pos_params.corrupt_ratio,
                    all_atoms=self.denoising_pos_params.all_atoms,
                )
            else:
                train_batch = add_gaussian_noise_schedule_to_position(
                    train_batch,
                    std_low=self.denoising_pos_params.std_low,
                    std_high=self.denoising_pos_params.std_high,
                    num_steps=self.denoising_pos_params.num_steps,
                    corrupt_ratio=self.denoising_pos_params.corrupt_ratio,
                    all_atoms=self.denoising_pos_params.all_atoms,
                )

        return self._calc_loss(train_batch, True)

    def validation_step(self, val_batch: T, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        # print(f"dataloader_idx: {dataloader_idx}, bs: {len(val_batch["natoms"])}")
        if dataloader_idx == 0:
            return self._calc_loss(val_batch, False)
        else:
            val_emb = self.regressor.backbone(val_batch)
            # avoid using energy_head.loss
            # because this will cause label leakage in normalizer
            val_e_pred = self.regressor.energy_head(val_batch, val_emb).view(-1)
            raw_val_e_pred = self.regressor.energy_head.denormalize(val_e_pred, val_batch)
            raw_target = val_batch["energy"].view(-1)
            # energy_output = self.regressor.energy_head.loss(val_batch, val_emb)
            energy_metrics = {
                "raw_total_energy_mae": torch.abs(raw_val_e_pred - raw_target).mean().item(),
                "raw_peratom_energy_mae": torch.abs(
                    raw_val_e_pred / val_batch.natoms - \
                    raw_target / val_batch.natoms).mean().item()  # raw loss (energy / atom)
            }
            for k, v in energy_metrics.items():
                self.log(
                    f"{k}_val",
                    v,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=len(val_batch["natoms"]),
                    sync_dist=True,
            )
            return energy_metrics["raw_total_energy_mae"]
        
    def test_step(self, test_batch: T, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._calc_loss(test_batch, False)

    def _calc_loss(self, batch: T, train: bool) -> Optional[STEP_OUTPUT]:
        batch_output = self.regressor.loss(batch) # a base.ModelOutput obj with two attrs: loss tensor and log dict
        loss, metrics = batch_output.loss, batch_output.log
        
        step_type = "train" if train else "val"
        batch_size = len(batch["natoms"])
        self.log(
            f"loss_{step_type}",
            loss,
            on_step=train,
            on_epoch=True,
            prog_bar=train,
            batch_size=batch_size,
            sync_dist=True,
        )
        for k, v in metrics.items():
            self.log(
                f"{k}_{step_type}",
                v,
                on_step=train,
                on_epoch=True,
                prog_bar=train,
                batch_size=batch_size,
                sync_dist=True,
            )
        return loss
    

@dataclass
class DenoisingPosParams:
    prob: float = 0.0
    fixed_noise_std: bool = False
    std: float = None
    num_steps: int = None
    std_low: float = None
    std_high: float = None
    corrupt_ratio: float = None
    all_atoms: bool = False
    denoising_pos_coefficient: float = None


def add_gaussian_noise_to_position(batch, std, corrupt_ratio=None, all_atoms=False):
    """
    1.  Update `pos` in `batch`.
    2.  Add `noise_vec` to `batch`, which will serve as the target for denoising positions.
    3.  Add `denoising_pos_forward` to switch to denoising mode during training.
    4.  Add `noise_mask` for partially corrupted structures when `corrupt_ratio` is not None.
    5.  If `all_atoms` == True, we add noise to all atoms including fixed ones.
    6.  Check whether `batch` has `md`. We do not add noise to structures from MD split.
    """
    noise_vec = torch.zeros_like(batch.pos)
    noise_vec = noise_vec.normal_(mean=0.0, std=std)

    if corrupt_ratio is not None:
        noise_mask = torch.rand(
            (batch.pos.shape[0]),
            dtype=batch.pos.dtype,
            device=batch.pos.device,
        )
        noise_mask = noise_mask < corrupt_ratio # 0 means not add noise, predict forces. 1 means predict noise
        noise_vec[(~noise_mask)] *= 0
        batch.noise_mask = noise_mask

    # Not add noise to structures from MD split
    if hasattr(batch, "md"):
        batch_index = batch.batch
        md_index = batch.md.bool()
        md_index = md_index[batch_index]
        noise_mask = ~md_index
        noise_vec[(~noise_mask)] *= 0
        if hasattr(batch, "noise_mask"):
            batch.noise_mask = batch.noise_mask * noise_mask
        else:
            batch.noise_mask = noise_mask

    pos = batch.pos
    new_pos = pos + noise_vec
    if all_atoms:
        batch.pos = new_pos
    else:
        free_mask = batch.fixed == 0.0
        batch.pos[free_mask] = new_pos[free_mask]

    batch.noise_vec = noise_vec
    batch.denoising_pos_forward = True

    return batch


def add_gaussian_noise_schedule_to_position(
    batch, std_low, std_high, num_steps, corrupt_ratio=None, all_atoms=False
):
    """
    1.  Similar to above, update positions in batch with gaussian noise, but
        additionally, also save the sigmas the noise vectors are sampled from.
    2.  Add `noise_mask` for partially corrupted structures when `corrupt_ratio`
        is not None.
    3.  If `all_atoms` == True, we add noise to all atoms including fixed ones.
    4.  Check whether `batch` has `md`. We do not add noise to structures from MD split.
    """
    sigmas = torch.tensor(
        np.exp(np.linspace(np.log(std_low), np.log(std_high), num_steps)),
        dtype=torch.float32,
    )
    # select a sigma for each structure, and project it all atoms in the structure.
    ts = torch.randint(0, num_steps, size=(batch.natoms.size(0),))
    batch.sigmas = sigmas[ts][batch.batch][:, None]  # (natoms, 1)
    noise_vec = torch.zeros_like(batch.pos)
    noise_vec = noise_vec.normal_() * batch.sigmas

    if corrupt_ratio is not None:
        noise_mask = torch.rand(
            (batch.pos.shape[0]),
            dtype=batch.pos.dtype,
            device=batch.pos.device,
        )
        noise_mask = noise_mask < corrupt_ratio
        # noise_vec[(~noise_mask)] *= 0
        batch.noise_mask = noise_mask

    # Not add noise to structures from MD split
    if hasattr(batch, "md"):
        batch_index = batch.batch
        md_index = batch.md.bool()
        md_index = md_index[batch_index]
        noise_mask = ~md_index
        # noise_vec[(~noise_mask)] *= 0
        if hasattr(batch, "noise_mask"):
            batch.noise_mask = batch.noise_mask * noise_mask
        else:
            batch.noise_mask = noise_mask

    if hasattr(batch, "noise_mask"):
        noise_vec[(~batch.noise_mask)] *= 0

    # only add noise to free atoms
    pos = batch.pos
    new_pos = pos + noise_vec
    if all_atoms:
        batch.pos = new_pos
    else:
        free_mask = batch.fixed == 0.0
        batch.pos[free_mask] = new_pos[free_mask]

    batch.noise_vec = noise_vec
    batch.denoising_pos_forward = True

    return batch
