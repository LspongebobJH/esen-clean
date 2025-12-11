"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Iterator

import numba as nb
import numpy as np
import torch
from torch.utils.data import Sampler, BatchSampler

from fairchem.core.common import gp_utils

if TYPE_CHECKING:
    from fairchem.core.datasets.base_dataset import BaseDataset


@nb.njit  # JIT compilation for performance optimization
def get_batches(
    natoms_list: np.array,  # List of number of atoms in each sample
    indices: np.array,  # Indices of the samples
    max_atoms: int,  # Maximum number of atoms allowed in a batch
    min_atoms: int,  # Minimum number of atoms allowed in a batch
) -> tuple[list[list[int]], list[int], int]:
    """
    Greedily creates batches from a list of samples with varying numbers of atoms.
    
    This function implements a greedy algorithm to group molecular samples into batches
    such that the total number of atoms in each batch doesn't exceed max_atoms,
    which prevents GPU memory overflow during training.

    Args:
        natoms_list: Array of number of atoms in each sample.
        indices: Array of indices of the samples.
        max_atoms: Maximum number of atoms allowed in a batch.
        min_atoms: Minimum number of atoms required for a valid batch.

    Returns:
        tuple[list[list[int]], list[int], int]:
            A tuple containing:
            - List of batches (each batch is a list of sample indices)
            - List of total atom counts for each batch
            - Number of samples filtered out due to exceeding max_atoms
    """

    # Input validation
    assert max_atoms > 0, "max_atoms must be positive"
    assert len(natoms_list) > 0, "natoms_list cannot be empty"
    assert len(natoms_list) == len(indices), "natoms_list and indices must have same length"

    # Initialize batch tracking variables
    batches = []  # List of batches, where each batch is a list of indices
    run_sum = 0  # Running total of atoms in the current batch being constructed
    cur_batch = nb.typed.List.empty_list(nb.int64)  # Current batch being constructed
    atom_counts = nb.typed.List.empty_list(nb.int64)  # Total atom count for each completed batch
    samples_filtered = 0  # Count of samples exceeding max_atoms limit

    # Greedy batch construction algorithm
    for idx, atoms in zip(indices, natoms_list):
        # Filter out samples that are too large to fit in any batch
        if atoms > max_atoms:
            samples_filtered += 1
            continue

        # Try to add current sample to the current batch
        if run_sum + atoms <= max_atoms:
            # Sample fits in current batch - add it
            cur_batch.append(idx)
            run_sum += atoms
        else:
            # Sample doesn't fit - finalize current batch and start new one
            if run_sum >= min_atoms:  # Only keep batches meeting minimum size requirement
                batches.append(cur_batch)
                atom_counts.append(run_sum)
            
            # Start new batch with current sample
            cur_batch = nb.typed.List([idx])
            run_sum = atoms

    # Handle the final batch
    if run_sum >= min_atoms:
        batches.append(cur_batch)
        atom_counts.append(run_sum)

    # Convert numba typed lists to regular Python lists for return
    return [list(x) for x in batches], list(atom_counts), samples_filtered


class MaxAtomDistributedBatchSampler(BatchSampler):
    """
    A custom batch sampler for distributed training with molecular data.
    
    This sampler addresses the challenge of training on molecular datasets where
    samples have vastly different sizes (number of atoms). It creates batches
    with dynamic sizes based on atom count constraints, ensuring efficient
    GPU memory usage while maintaining balanced workload across multiple GPUs.
    
    Key Features:
    - Dynamic batch sizing based on atom count limits
    - Even distribution of computational load across GPUs
    - Deterministic shuffling for reproducible distributed training
    - Support for resuming training from checkpoints

    Args:
        dataset (BaseDataset): The molecular dataset to sample from.
        max_atoms (int): Maximum number of atoms allowed in a single batch.
        num_replicas (int): Number of GPUs/processes in distributed training.
        rank (int): Rank of the current GPU/process (0-indexed).
        seed (int): Random seed for reproducible shuffling.
        shuffle (bool): Whether to shuffle the dataset each epoch. Defaults to True.
        drop_last (bool): Whether to drop incomplete batches. Defaults to False.
        min_atoms (int): Minimum atoms required for a valid batch. Defaults to 0.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        max_atoms: int,
        num_replicas: int,
        rank: int,
        seed: int,
        shuffle: bool = True,
        drop_last: bool = False,
        min_atoms: int = 0,
        sampler=None, # NOTE: add for pytorch_lightning compatibility
    ) -> None:
        self.dataset = dataset
        self.max_atoms = max_atoms
        self.min_atoms = min_atoms
        
        # Distributed training configuration
        self.num_replicas = num_replicas
        self.rank = rank  # Current GPU identifier
        assert self.num_replicas > 0, "num_replicas must be positive"
        assert self.rank < self.num_replicas, "rank must be less than num_replicas"

        # Special handling for gradient parallel training
        if gp_utils.initialized():
            assert (
                min_atoms >= gp_utils.get_gp_world_size()
            ), "Min atoms needs to be at least gp world size for gradient parallelism!"

        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Training state tracking
        self.epoch = 0 
        self.start_iter = 0

        # Pre-generate all batches once during initialization
        # This ensures consistent batch distribution across epochs and GPUs
        # List of batches (each batch is a list of sample indices)
        self.all_batches = self._prepare_batches()
        
        # Calculate batch size per GPU for distributed training
        if self.drop_last and len(self.all_batches) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.all_batches) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.all_batches) / self.num_replicas)
        
        self.total_size = self.num_samples * self.num_replicas
        
        # Sanity check: ensure we have enough batches for all GPUs
        assert (
            len(self.all_batches) >= self.num_replicas
        ), "Dataset too small: fewer batches than GPUs!"

    def _prepare_batches(self) -> list[int]:
        """
        Pre-generate all batches for the entire dataset.
        
        This method performs the expensive batch generation once during initialization
        rather than every epoch, significantly improving training startup time.
        
        Returns:
            List of batches, where each batch is a list of sample indices.
        """
        # Shuffle is mandatory here because molecular datasets are often sorted
        # by atom count, leading to highly uneven batch distribution without shuffling
        rng = np.random.default_rng(self.seed)
        original_indices = rng.permutation(len(self.dataset))
        
        # Retrieve atom count metadata for all samples
        # This is the performance bottleneck as it requires dataset access
        t0 = time.time()
        natoms_list = self.dataset.get_metadata("natoms", original_indices.tolist())
        t1 = time.time()
        
        # Generate batches using the greedy algorithm
        indices, atoms_count, samples_filtered = get_batches(
            np.array(natoms_list), original_indices, self.max_atoms, self.min_atoms
        )
        t2 = time.time()
        
        # Log performance and batch statistics for monitoring
        print(
            f"Sampler batch generation times: get natoms: {t1 - t0:.3f}s, total: {t2 - t0:.3f}s"
        )
        print(
            f"MaxAtomDistributedSampler generated {len(indices)} batches with "
            f"total atoms {np.sum(natoms_list)}, max: {max(atoms_count)}, "
            f"min: {min(atoms_count)}, mean: {np.mean(atoms_count):.1f}, "
            f"std: {np.std(atoms_count):.1f}"
        )
        print(
            f"{samples_filtered} samples were removed because they exceed {self.max_atoms} atoms"
        )
        
        return indices

    def __len__(self) -> int:
        """Return the number of batches this GPU will process per epoch."""
        return self.num_samples

    def __iter__(self) -> Iterator[list[int]]:
        """
        Generate an iterator over batches for the current GPU.
        
        This method implements the core distributed sampling logic:
        1. Optionally shuffle batch order based on current epoch
        2. Ensure equal number of batches across all GPUs via padding/dropping
        3. Select the subset of batches assigned to this GPU
        4. Support resuming from a specific iteration
        
        Returns:
            Iterator over batches assigned to this GPU.
        """
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.all_batches), generator=g).tolist()
        else:
            indices = list(range(len(self.all_batches)))

        # Ensure equal number of batches across all GPUs
        if not self.drop_last:
            # Pad with repeated batches to reach total_size
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                # Simple case: repeat from beginning
                indices += indices[:padding_size]
            else:
                # Complex case: repeat entire list multiple times
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Drop excess batches to reach exact total_size
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size, "Batch count mismatch after padding/dropping"

        # Distributed sampling: assign batches to this GPU using interleaved pattern
        # GPU 0 gets indices [0, num_replicas, 2*num_replicas, ...]
        # GPU 1 gets indices [1, num_replicas+1, 2*num_replicas+1, ...]
        # This ensures even distribution of computational complexity
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples, "GPU batch count mismatch"
        
        # Extract actual batches corresponding to selected indices
        batch_slice = [self.all_batches[i] for i in indices]
        
        # Support for resuming training from a specific iteration
        assert (
            self.start_iter < len(batch_slice)
        ), f"Start iteration {self.start_iter} exceeds available batches {len(batch_slice)}"
        
        # Return iterator starting from the specified iteration
        return iter(batch_slice[self.start_iter:])

    def set_epoch_and_start_iteration(self, epoch: int, start_iter: int) -> None:
        """
        Set the current epoch and starting iteration for training resumption.
        
        This method is crucial for:
        1. Ensuring different data order across epochs (via epoch-based shuffling)
        2. Supporting checkpoint resumption (via start_iter)
        
        Args:
            epoch: Current training epoch number.
            start_iter: Iteration number to start from (for resuming training).
        """
        self.epoch = epoch
        self.start_iter = start_iter