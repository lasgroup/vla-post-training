from collections.abc import Sequence
import etils.epath as epath
import jax
import logging
import numpy as np
import torch
from typing import Literal

import lerobot.datasets.lerobot_dataset as lerobot_dataset
import openpi.training.config as _config
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.data_loader import Dataset, DataLoader, DataLoaderImpl, transform_dataset, TorchDataLoader
from openpi.training.data_loader import create_torch_dataset as original_create_torch_dataset


class BalancedConcatSampler(torch.utils.data.Sampler):
    def __init__(self, concat_dataset, num_samples: int | None = None):
        self.concat = concat_dataset
        self.lengths = [len(d) for d in concat_dataset.datasets]
        self.cum_offsets = np.cumsum([0, *self.lengths])
        self.num_samples = num_samples or sum(self.lengths)

    def __iter__(self):
        for _ in range(self.num_samples):
            ds_idx = np.random.randint(len(self.lengths))
            local_idx = np.random.randint(self.lengths[ds_idx])
            yield int(self.cum_offsets[ds_idx] + local_idx)

    def __len__(self):
        return self.num_samples


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    collected_data_paths: Sequence[epath.Path] | None = None
) -> Dataset:
    """Create a dataset for training. Adapted to support additional repo paths."""

    dataset = original_create_torch_dataset(
        data_config,
        action_horizon,
        model_config,
    )

    # if collected_data_paths is not None:
    if data_config.additional_repo_paths:
        datasets = [dataset]
        for repo_path in data_config.additional_repo_paths:  # collected_data_paths
            meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=repo_path)
            ds = lerobot_dataset.LeRobotDataset(
                repo_id,
                root=repo_path,
                delta_timestamps={
                    key: [t / meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
                },
            )
            if data_config.prompt_from_task:
                ds = TransformedDataset(ds, [_transforms.PromptFromLeRobotTask(meta.tasks)])
            datasets.append(ds)
        return torch.utils.data.ConcatDataset(datasets)

    return dataset


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    collected_data_paths: Sequence[epath.Path] | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        raise NotImplementedError("RLDS data loader is not implemented yet.")
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        collected_data_paths=collected_data_paths,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    collected_data_paths: Sequence[epath.Path] | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    # If multiple datasets were concatenated, create a sampler that gives equal
    # total probability to each sub-dataset (so each dataset is chosen 50/50).
    if sampler is None and isinstance(dataset, torch.utils.data.ConcatDataset) and len(dataset.datasets) > 1:
        sampler = BalancedConcatSampler(dataset)

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)
