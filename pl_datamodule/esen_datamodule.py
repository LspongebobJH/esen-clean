# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from typing import List, Literal, Optional, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fairchem.core.common import distutils
from fairchem.core.common.data_parallel import BalancedBatchSampler
from fairchem.core.datasets.base_dataset import Subset
from fairchem.core.datasets.lmdb_dataset import data_list_collater
from fairchem.core.datasets.samplers.max_atom_distributed_sampler import \
    MaxAtomDistributedBatchSampler


class EsenDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_workers: DictConfig,
        batch_size: DictConfig,
        train_dataset=None,
        val_dataset=None,
        extra_val_datasets=None,
        test_dataset=None,
        pin_memory: bool = True,
        otf_graph=True,
        data_seed=42,
        # for MaxAtomDistributedBatchSampler
        sampler_type: Literal["balanced", "max_atoms"] = "balanced",
        max_atoms: Optional[int] = None,
        min_atoms: Optional[int] = None,
        **kwargs,
    ):
        """
        extra_val_datasets: dict of datasets to use for validation 
            in the form {'0': dataset_0, ..., 'i': dataset_i}. We
            do not use List type for convenience of hydra config.
        sampler_type: Type of sampler to use ("balanced" or "max_atoms")
        max_atoms: Maximum atoms per batch (required when sampler_type="max_atoms")
        min_atoms: Minimum atoms per batch (optional, defaults to 0)
        """
        super().__init__()
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.data_seed = data_seed
        self.otf_graph = otf_graph

        # Sampler configuration
        self.sampler_type = sampler_type
        self.max_atoms = max_atoms
        self.min_atoms = min_atoms or 0
        # Validate sampler configuration
        if self.sampler_type == "max_atoms" and self.max_atoms is None:
            raise ValueError("max_atoms must be specified when using max_atoms sampler")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.extra_val_datasets = extra_val_datasets or {} # extra val datasets for monitor
        self.test_dataset = test_dataset
        self.datasets = [train_dataset, val_dataset, test_dataset]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_balanced_sampler(
        self, 
        dataset, 
        batch_size: int, 
        shuffle: bool, 
        balancing_mode: str="atoms",
        on_error: str="raise",
        drop_last: bool=False
    ) -> BalancedBatchSampler:
        """Create the original BalancedBatchSampler."""
        if balancing_mode is not None:
            if on_error is None:
                on_error = "raise"
        else:
            balancing_mode = "atoms"

        if on_error is None:
            on_error = "warn_and_no_balance"

        num_replicas = distutils.get_world_size()
        rank = distutils.get_rank()

        return BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=num_replicas,
            rank=rank,
            device=self.device,
            mode=balancing_mode,
            shuffle=shuffle,
            on_error=on_error,
            seed=self.data_seed,
            drop_last=drop_last,
        )
    
    def _create_max_atoms_sampler(
        self,
        dataset,
        max_atoms: int,
        shuffle: bool,
        drop_last: bool = False
    ) -> MaxAtomDistributedBatchSampler:
        """Create the MaxAtomDistributedBatchSampler."""
        num_replicas = distutils.get_world_size()
        rank = distutils.get_rank()

        return MaxAtomDistributedBatchSampler(
            dataset=dataset,
            max_atoms=max_atoms,
            num_replicas=num_replicas,
            rank=rank,
            seed=self.data_seed,
            shuffle=shuffle,
            drop_last=drop_last,
            min_atoms=self.min_atoms,
        )
    
    def _create_sampler(
        self, 
        dataset, 
        batch_size: int, 
        shuffle: bool, 
        drop_last: bool = False
    ) -> Union[BalancedBatchSampler, MaxAtomDistributedBatchSampler]:
        if self.sampler_type == "balanced":
            return self._create_balanced_sampler(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        elif self.sampler_type == "max_atoms":
            return self._create_max_atoms_sampler(
                dataset=dataset,
                max_atoms=self.max_atoms,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        else:
            raise ValueError(f"Unknown sampler_type: {self.sampler_type}")
        
    def _create_datalaoder(self, dataset, is_training: bool, drop_last: bool = True):
        bs = self.batch_size.train if is_training else self.batch_size.val
        nw = self.num_workers.train if is_training else self.num_workers.val

        sampler = self._create_sampler(
            dataset=dataset,
            batch_size=bs,
            shuffle=is_training,
            drop_last=drop_last, # NOTE: also drop last for val/test to avoid null batches
        )
        return DataLoader(
            dataset,
            collate_fn=partial(data_list_collater, otf_graph=self.otf_graph),
            num_workers=nw,
            pin_memory=self.pin_memory,
            batch_sampler=sampler,
            timeout=10 * 60 if nw > 0 else 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._create_datalaoder(self.train_dataset, is_training=True)

    def val_dataloader(self) -> List[DataLoader]:
        val_loaders = [self._create_datalaoder(self.val_dataset, is_training=False, drop_last=False)]
                
        for i in range(len(self.extra_val_datasets)):
            val_loaders.append(
                self._create_datalaoder(self.extra_val_datasets[str(i)], is_training=False)
            )
        
        return val_loaders

    def test_dataloader(self) -> DataLoader | None:
        return self._create_datalaoder(self.test_dataset, is_training=False) if self.test_dataset is not None else None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )
    

if __name__ == "__main__":
    from fairchem.core.datasets.ase_datasets import AseDBDataset
    ds_train = AseDBDataset(
        config={
            "src": ["/workspace/data/MPtrj/lmdb/train/MPtrj_uncorrect-E_GPa-S_train.lmdb"],
            "a2g_args": {
                "r_energy":True, 
                "r_forces":True, 
                "r_stress":True,
                "r_edges": True,
            }
        }
    )
    ds_val = AseDBDataset(
        config={
            "src": ["/workspace/data/MPtrj/lmdb/val/MPtrj_uncorrect-E_GPa-S_val.lmdb"],
            "a2g_args": {
                "r_energy":True, 
                "r_forces":True, 
                "r_stress":True,
                "r_edges": True,
            }
        }
    )
    dm = EsenDataModule(
        num_workers=DictConfig({"train": 2, "val": 2}),
        batch_size=DictConfig({"train": 32, "val": 32}),
        train_dataset=ds_train,
        val_dataset=ds_val,
        data_seed=42,
        otf_graph=False,
        sampler_type="max_atoms",
        max_atoms=300,
    )
    for batch in dm.train_dataloader():
        print(f"total atoms: {batch['natoms'].sum()}, bs={len(batch['natoms'])}")
        break