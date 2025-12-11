"""
Copyright (c) Meta, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from abc import ABCMeta
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import psutil
import torch
from torch import randperm
from torch.utils.data import Dataset
from torch.utils.data import Subset as Subset_
from tqdm import tqdm

from fairchem.core.common import distutils
from fairchem.core.common.registry import registry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike


T_co = TypeVar("T_co", covariant=True)


class UnsupportedDatasetError(ValueError):
    pass


class BaseDataset(Dataset[T_co], metaclass=ABCMeta):
    """Base Dataset class for all OCP datasets."""

    def __init__(self, config: dict):
        """Initialize

        Args:
            config (dict): dataset configuration
        """
        self.config = config
        self.paths = []

        if "src" in self.config:
            if isinstance(config["src"], str):
                self.paths = [Path(self.config["src"])]
            else:
                self.paths = tuple(Path(path) for path in sorted(config["src"]))

        self.lin_ref = None
        if self.config.get("lin_ref", False):
            lin_ref = torch.tensor(
                np.load(self.config["lin_ref"], allow_pickle=True)["coeff"]
            )
            self.lin_ref = torch.nn.Parameter(lin_ref, requires_grad=False)
        
        # Initialize lazy loading for metadata
        self._lazy_metadata = None
        self._metadata_cache = {}
        self._use_lazy_loading = self.config.get("lazy_metadata_loading", False)
        
        # Initialize streaming metadata for ultra-large datasets
        self._use_streaming = self.config.get("streaming_metadata", False)
        self._metadata_file_map = None
        self._metadata_offsets = None

    def __len__(self) -> int:
        return self.num_samples

    def metadata_hasattr(self, attr) -> bool:
        return attr in self._metadata

    @cached_property
    def indices(self):
        return np.arange(self.num_samples, dtype=int)

    @cached_property
    def _metadata(self) -> dict[str, ArrayLike]:
        # logic to read metadata file here
        metadata_npzs = []
        
        # Check if we should use memory optimization
        use_memory_optimization = self.config.get("memory_optimized_metadata", False)
        batch_size = self.config.get("metadata_batch_size", 1000)
        
        if self.config.get("metadata_path", None) is not None:
            if use_memory_optimization:
                # Use memory mapping for large files
                # NOTE: yuanhangtangle,2025-07-31 seems that mmap_mode only works for npy files?
                if distutils.is_master():
                    print("loading metadata with `memory_optimized_metadata=True`")
                metadata_npzs.append(
                    np.load(self.config["metadata_path"], allow_pickle=True, mmap_mode='r')
                )
            else:
                metadata_npzs.append(
                    np.load(self.config["metadata_path"], allow_pickle=True)
                )
        else:
            print(f"Loading metadata from {len(self.paths)} files")
            _n = len(self.paths) // 10 if len(self.paths) > 10 else 1
            
            if use_memory_optimization and len(self.paths) > batch_size:
                # Process in batches to reduce memory usage
                metadata = self._load_metadata_in_batches(batch_size)
                return metadata
            else:
                # Original loading method
                for i, path in enumerate(self.paths):
                    if i % _n == 0 and distutils.is_master():
                        print(f"Loading metadata from {i} / {len(self.paths)} files")
                        print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024 / 1024:.2f} GB")
                    if path.is_file():
                        metadata_file = path.parent / "metadata.npz"
                    else:
                        metadata_file = path / "metadata.npz"
                    if metadata_file.is_file():
                        metadata_npzs.append(np.load(metadata_file, allow_pickle=True))

        if len(metadata_npzs) == 0:
            logging.warning(
                f"Could not find dataset metadata.npz files in '{self.paths}'"
            )
            return {}

        metadata = {
            field: np.concatenate([metadata[field] for metadata in metadata_npzs])
            for field in metadata_npzs[0]
        }

        assert np.issubdtype(
            metadata["natoms"].dtype, np.integer
        ), f"Metadata natoms must be an integer type! not {metadata['natoms'].dtype}"
        assert metadata["natoms"].shape[0] == len(
            self
        ), "Loaded metadata and dataset size mismatch."

        return metadata

    def _load_metadata_in_batches(self, batch_size: int) -> dict[str, ArrayLike]:
        """Load metadata in batches to reduce memory usage."""
        if distutils.is_master():
            print(f"Loading metadata in batches of {batch_size}")
        
        # First pass: collect all metadata files and their sizes
        metadata_files = []
        total_samples = 0
        
        for path in self.paths:
            if path.is_file():
                metadata_file = path.parent / "metadata.npz"
            else:
                metadata_file = path / "metadata.npz"
            if metadata_file.is_file():
                metadata_files.append(metadata_file)
        
        if not metadata_files:
            logging.warning(f"Could not find dataset metadata.npz files in '{self.paths}'")
            return {}
        
        # Load first file to get field names
        first_metadata = np.load(metadata_files[0], allow_pickle=True)
        field_names = list(first_metadata.keys())
        
        # Initialize result arrays
        metadata = {}
        for field in field_names:
            # Get dtype from first file
            dtype = first_metadata[field].dtype
            metadata[field] = []
        
        # Process files in batches
        for i in range(0, len(metadata_files), batch_size):
            batch_files = metadata_files[i:i + batch_size]
            if distutils.is_master():
                print(f"Processing batch {i//batch_size + 1}/{(len(metadata_files) + batch_size - 1)//batch_size}")
            
            batch_metadata_npzs = []
            _iter = tqdm(batch_files, desc="Loading metadata in batches", disable=not distutils.is_master())
            for metadata_file in _iter:
                batch_metadata_npzs.append(np.load(metadata_file, allow_pickle=True))
            
            # Concatenate batch
            if distutils.is_master():
                print(f"Concatenating batch")
            batch_metadata = {
                field: np.concatenate([metadata[field] for metadata in batch_metadata_npzs])
                for field in field_names
            }
            
            # Append to result
            if distutils.is_master():
                print(f"Appending batch to result")
            for field in field_names:
                metadata[field].append(batch_metadata[field])
            
            # Clear batch data to free memory
            del batch_metadata_npzs, batch_metadata
            
            if distutils.is_master():
                print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024 / 1024:.2f} GB")
        
        # Final concatenation
        final_metadata = {
            field: np.concatenate(metadata[field]) for field in field_names
        }
        
        # Validation
        assert np.issubdtype(
            final_metadata["natoms"].dtype, np.integer
        ), f"Metadata natoms must be an integer type! not {final_metadata['natoms'].dtype}"
        assert final_metadata["natoms"].shape[0] == len(
            self
        ), "Loaded metadata and dataset size mismatch."
        
        return final_metadata

    def get_metadata(self, attr, idx):
        if self._use_streaming:
            return self._get_metadata_streaming(attr, idx)
        
        if self._use_lazy_loading:
            if distutils.is_master():
                print("loading metadata with `lazy_metadata_loading=True`")
            return self._get_metadata_lazy(attr, idx)
        
        if attr in self._metadata:
            metadata_attr = self._metadata[attr]
            if isinstance(idx, list):
                return [metadata_attr[_idx] for _idx in idx]
            return metadata_attr[idx]
        return None

    def _get_metadata_lazy(self, attr, idx):
        """Lazy load specific metadata attribute."""
        if attr not in self._metadata_cache:
            # Load only the specific attribute
            self._metadata_cache[attr] = self._load_single_metadata_attribute(attr)
        
        metadata_attr = self._metadata_cache[attr]
        if isinstance(idx, list):
            return [metadata_attr[_idx] for _idx in idx]
        return metadata_attr[idx]

    def _load_single_metadata_attribute(self, attr):
        """Load a single metadata attribute from all files."""
        if self.config.get("metadata_path", None) is not None:
            metadata = np.load(self.config["metadata_path"], allow_pickle=True)
            return metadata[attr]
        
        # Load from multiple files
        attr_data = []
        for path in self.paths:
            if path.is_file():
                metadata_file = path.parent / "metadata.npz"
            else:
                metadata_file = path / "metadata.npz"
            if metadata_file.is_file():
                metadata = np.load(metadata_file, allow_pickle=True)
                if attr in metadata:
                    attr_data.append(metadata[attr])
        
        if not attr_data:
            raise KeyError(f"Attribute '{attr}' not found in any metadata files")
        
        return np.concatenate(attr_data)

    def _get_metadata_streaming(self, attr, idx):
        """Stream metadata from disk without loading everything into memory."""
        if self._metadata_file_map is None:
            self._build_metadata_file_map()
        
        if isinstance(idx, list):
            # Optimize batch access by grouping indices by file
            return self._get_metadata_batch_streaming(attr, idx)
        return self._get_single_metadata_value(attr, idx)

    def _get_metadata_batch_streaming(self, attr, indices):
        """Optimized batch access by grouping indices by file."""
        # Group indices by file to minimize file I/O
        file_groups = {}
        
        for global_idx in indices:
            file_idx = self._get_file_index_for_global_idx(global_idx)
            if file_idx not in file_groups:
                file_groups[file_idx] = []
            file_groups[file_idx].append(global_idx)
        
        # Load each file once and extract all needed values
        result = [None] * len(indices)
        index_to_result = {idx: i for i, idx in enumerate(indices)}
        
        for file_idx, global_indices in file_groups.items():
            file_info = self._metadata_file_map[file_idx]
            metadata = np.load(file_info['file'], allow_pickle=True)
            
            if attr not in metadata:
                raise KeyError(f"Attribute '{attr}' not found in file {file_info['file']}")
            
            attr_data = metadata[attr]
            
            for global_idx in global_indices:
                local_idx = global_idx - file_info['start_idx']
                result[index_to_result[global_idx]] = attr_data[local_idx]
        
        return result

    def _get_file_index_for_global_idx(self, global_idx):
        """Find which file contains a given global index."""
        for i, file_info in enumerate(self._metadata_file_map):
            if (file_info['start_idx'] <= global_idx < 
                file_info['start_idx'] + file_info['size']):
                return i
        raise IndexError(f"Global index {global_idx} out of range")

    def _build_metadata_file_map(self):
        """Build a mapping from global index to file and local index."""
        if distutils.is_master():
            print("Building metadata file mapping for streaming access...")
        
        self._metadata_file_map = []
        self._metadata_offsets = [0]
        current_offset = 0
        
        for path in self.paths:
            if path.is_file():
                metadata_file = path.parent / "metadata.npz"
            else:
                metadata_file = path / "metadata.npz"
            
            if metadata_file.is_file():
                # Load only the size information, not the full data
                metadata = np.load(metadata_file, allow_pickle=True)
                if "natoms" in metadata:
                    file_size = len(metadata["natoms"])
                    self._metadata_file_map.append({
                        'file': metadata_file,
                        'size': file_size,
                        'start_idx': current_offset
                    })
                    current_offset += file_size
                    self._metadata_offsets.append(current_offset)
        
        if distutils.is_master():
            print(f"Built mapping for {len(self._metadata_file_map)} files, total {current_offset} samples")

    def _get_single_metadata_value(self, attr, global_idx):
        """Get a single metadata value by global index."""
        # Find which file contains this index
        file_idx = None
        local_idx = None
        
        for i, file_info in enumerate(self._metadata_file_map):
            if (file_info['start_idx'] <= global_idx < 
                file_info['start_idx'] + file_info['size']):
                file_idx = i
                local_idx = global_idx - file_info['start_idx']
                break
        
        if file_idx is None:
            raise IndexError(f"Global index {global_idx} out of range")
        
        # Load the specific file and get the value
        file_info = self._metadata_file_map[file_idx]
        metadata = np.load(file_info['file'], allow_pickle=True)
        
        if attr not in metadata:
            raise KeyError(f"Attribute '{attr}' not found in file {file_info['file']}")
        
        return metadata[attr][local_idx]


class Subset(Subset_, BaseDataset):
    """A pytorch subset that also takes metadata if given."""

    def __init__(
        self,
        dataset: BaseDataset,
        indices: Sequence[int],
        metadata: dict[str, ArrayLike],
    ) -> None:
        super().__init__(dataset, indices)
        self.metadata = metadata
        self.indices = indices
        self.num_samples = len(indices)
        self.config = dataset.config

    @cached_property
    def _metadata(self) -> dict[str, ArrayLike]:
        return self.dataset._metadata

    def get_metadata(self, attr, idx):
        if isinstance(idx, list):
            return self.dataset.get_metadata(attr, [[self.indices[i] for i in idx]])
        return self.dataset.get_metadata(attr, self.indices[idx])


def create_dataset(config: dict[str, Any], split: str) -> Subset:
    """Create a dataset from a config dictionary

    Args:
        config (dict): dataset config dictionary
        split (str): name of split

    Returns:
        Subset: dataset subset class
    """
    # Initialize the dataset
    dataset_cls = registry.get_dataset_class(config.get("format", "lmdb"))
    assert issubclass(dataset_cls, Dataset), f"{dataset_cls} is not a Dataset"

    # remove information about other splits, only keep specified split
    # this may only work with the mt config not main config
    current_split_config = config.copy()
    if "splits" in current_split_config:
        current_split_config.pop("splits")
        current_split_config.update(config["splits"][split])

    seed = current_split_config.get("seed", 0)
    if split != "train":
        seed += (
            1  # if we use same dataset for train / val , make sure its diff sampling
        )

    g = torch.Generator()
    g.manual_seed(seed)

    dataset = dataset_cls(current_split_config)

    # Get indices of the dataset
    indices = dataset.indices
    max_atoms = current_split_config.get("max_atoms", None)
    if max_atoms is not None:
        if not dataset.metadata_hasattr("natoms"):
            raise ValueError("Cannot use max_atoms without dataset metadata")
        indices = indices[dataset.get_metadata("natoms", indices) <= max_atoms]

    for subset_to in current_split_config.get("subset_to", []):
        if not dataset.metadata_hasattr(subset_to["metadata_key"]):
            raise ValueError(
                f"Cannot use {subset_to} without dataset metadata key {subset_to['metadata_key']}"
            )
        rhv = subset_to["rhv"]
        if isinstance(rhv, str):
            with open(rhv) as f:
                rhv = f.read().splitlines()
                rhv = [int(x) for x in rhv]
        if subset_to["op"] == "abs_le":
            indices = indices[
                np.abs(dataset.get_metadata(subset_to["metadata_key"], indices)) <= rhv
            ]
        elif subset_to["op"] == "in":
            indices = indices[
                np.isin(dataset.get_metadata(subset_to["metadata_key"], indices), rhv)
            ]

    # Apply dataset level transforms
    # TODO is no_shuffle mutually exclusive though? or what is the purpose of no_shuffle?
    first_n = current_split_config.get("first_n")
    sample_n = current_split_config.get("sample_n")
    no_shuffle = current_split_config.get("no_shuffle")
    # this is true if at most one of the mutually exclusive arguments are set
    if sum(arg is not None for arg in (first_n, sample_n, no_shuffle)) > 1:
        raise ValueError(
            "sample_n, first_n, no_shuffle are mutually exclusive arguments. Only one can be provided."
        )
    if first_n is not None:
        max_index = first_n
    elif sample_n is not None:
        # shuffle by default, user can disable to optimize if they have confidence in dataset
        # shuffle all datasets by default to avoid biasing the sampling in concat dataset
        # TODO only shuffle if split is train
        max_index = sample_n
        indices = (
            indices
            if len(indices) == 1
            else indices[randperm(len(indices), generator=g)]
        )
    else:
        max_index = len(indices)
        indices = (
            indices
            if (no_shuffle or len(indices) == 1)
            else indices[randperm(len(indices), generator=g)]
        )

    if max_index > len(indices):
        msg = (
            f"Cannot take {max_index} data points from a dataset of only length {len(indices)}.\n"
            f"Make sure to set first_n or sample_n to a number =< the total samples in dataset."
        )
        if max_atoms is not None:
            msg = msg[:-1] + f"that are smaller than the given max_atoms {max_atoms}."
        raise ValueError(msg)

    indices = indices[:max_index]

    return Subset(dataset, indices, metadata=dataset._metadata)
