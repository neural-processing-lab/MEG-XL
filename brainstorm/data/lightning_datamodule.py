"""PyTorch Lightning DataModule for Armeni MEG Dataset."""

import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from typing import Optional, List, Callable
import numpy as np

from .armeni_dataset import ArmeniMEGDataset
from .samplers import RecordingShuffleSampler


class ArmeniMEGDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the Armeni MEG dataset.

    This module handles:
    - Creating separate train/val datasets with session-based splitting
    - Custom collate function for batch preparation
    - RecordingShuffleSampler for efficient training
    - Proper cleanup of HDF5 file handles

    Parameters
    ----------
    data_root : str
        Root directory of the Armeni dataset
    segment_length : float
        Length of each segment in seconds
    cache_dir : str
        Directory for storing preprocessed cache files
    subjects : List[str]
        List of subjects to include (e.g., ["sub-001", "sub-002"])
    val_session : str
        Session to use for validation (e.g., "ses-010")
    tasks : List[str]
        List of tasks to include (e.g., ["compr"])
    l_freq : float
        Low frequency cutoff for band-pass filter
    h_freq : float
        High frequency cutoff for band-pass filter
    target_sfreq : float
        Target sampling frequency after resampling
    channel_filter : Callable[[str], bool]
        Filter function for channels
    batch_size : int
        Batch size for training and validation
    num_workers : int
        Number of DataLoader workers
    pin_memory : bool
        Whether to use pinned memory for faster GPU transfer
    persistent_workers : bool
        Whether to keep workers alive between epochs
    use_recording_sampler : bool
        Whether to use RecordingShuffleSampler for training
    sampler_seed : int
        Random seed for the sampler
    debug_mode : bool
        If True, uses only sub-001 with ses-001 for training and ses-010 for validation.
        Useful for quick debugging and testing.

    Example
    -------
    >>> datamodule = ArmeniMEGDataModule(
    ...     data_root="/path/to/armeni2022",
    ...     segment_length=10.0,
    ...     subjects=["sub-001", "sub-002", "sub-003"],
    ...     val_session="ses-010",
    ...     tasks=["compr"],
    ...     batch_size=8,
    ... )
    >>> datamodule.setup("fit")
    >>> train_loader = datamodule.train_dataloader()
    >>> val_loader = datamodule.val_dataloader()
    """

    def __init__(
        self,
        data_root: str,
        segment_length: float,
        cache_dir: str = "./data/cache",
        subjects: Optional[List[str]] = None,
        val_session: str = "ses-010",
        tasks: Optional[List[str]] = None,
        l_freq: float = 0.1,
        h_freq: float = 40.0,
        target_sfreq: float = 50.0,
        channel_filter: Callable[[str], bool] = lambda x: x.startswith('M'),
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        use_recording_sampler: bool = True,
        sampler_seed: int = 42,
        debug_mode: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['channel_filter'])

        # Store parameters
        self.data_root = data_root
        self.segment_length = segment_length
        self.cache_dir = cache_dir
        self.subjects = subjects
        self.val_session = val_session
        self.tasks = tasks
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.target_sfreq = target_sfreq
        self.channel_filter = channel_filter
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.use_recording_sampler = use_recording_sampler
        self.sampler_seed = sampler_seed
        self.debug_mode = debug_mode

        # Datasets (initialized in setup)
        self.train_dataset: Optional[ArmeniMEGDataset] = None
        self.val_dataset: Optional[ArmeniMEGDataset] = None

    def _get_train_sessions(self) -> List[str]:
        """
        Get list of training sessions (all except val_session).

        Returns
        -------
        train_sessions : List[str]
            List of session names for training (e.g., ["ses-001", "ses-002", ...])
        """
        # Generate all session IDs from ses-001 to ses-012, excluding val_session
        all_sessions = [f"ses-{i:03d}" for i in range(1, 13)]
        train_sessions = [s for s in all_sessions if s != self.val_session]
        return train_sessions

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for batching MEG segments.

        Converts list of dicts from dataset to tuple format expected by model.
        Stacks all samples' sensor information along the batch dimension.

        Parameters
        ----------
        batch : List[Dict[str, Any]]
            List of samples from dataset.__getitem__

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            - raw_meg: [B, C, T] stacked MEG signals
            - sensor_xyzdir: [B, C, 6] sensor positions and directions
            - sensor_types: [B, C] sensor types
            - sensor_mask: [B, C] sensor mask
        """
        meg = torch.stack([item["meg"] for item in batch])
        sensor_xyzdir = torch.stack([item["sensor_xyzdir"] for item in batch])
        sensor_types = torch.stack([item["sensor_types"] for item in batch])
        sensor_mask = torch.stack([item["sensor_mask"] for item in batch])
        return (meg, sensor_xyzdir, sensor_types, sensor_mask)

    def setup(self, stage: Optional[str] = None):
        """
        Setup train and validation datasets.

        Parameters
        ----------
        stage : str, optional
            Either "fit", "validate", "test", or "predict"
        """
        if stage == "fit" or stage is None:
            # Override settings for debug mode
            if self.debug_mode:
                print("\n🐛 DEBUG MODE ENABLED 🐛")
                print("Using sub-001 only: ses-001 for train, ses-010 for val\n")
                train_subjects = ["sub-001"]
                train_sessions = ["ses-001"]
                val_subjects = ["sub-001"]
            else:
                train_subjects = self.subjects
                train_sessions = self._get_train_sessions()
                val_subjects = self.subjects

            # Training dataset
            print(f"\n=== Setting up training dataset ===")
            print(f"Subjects: {train_subjects}")
            print(f"Sessions: {train_sessions}")
            print(f"Tasks: {self.tasks}")

            self.train_dataset = ArmeniMEGDataset(
                data_root=self.data_root,
                segment_length=self.segment_length,
                cache_dir=self.cache_dir,
                subjects=train_subjects,
                sessions=train_sessions,
                tasks=self.tasks,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                target_sfreq=self.target_sfreq,
                channel_filter=self.channel_filter,
            )

            print(f"Training dataset size: {len(self.train_dataset)} segments")

            # Validation dataset
            print(f"\n=== Setting up validation dataset ===")
            print(f"Subjects: {val_subjects}")
            print(f"Sessions: [{self.val_session}]")
            print(f"Tasks: {self.tasks}")

            self.val_dataset = ArmeniMEGDataset(
                data_root=self.data_root,
                segment_length=self.segment_length,
                cache_dir=self.cache_dir,
                subjects=val_subjects,
                sessions=[self.val_session],
                tasks=self.tasks,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                target_sfreq=self.target_sfreq,
                channel_filter=self.channel_filter,
            )

            # In debug mode, subsample validation to 5%
            if self.debug_mode:
                total_segments = len(self.val_dataset)
                subset_size = max(1, int(0.1 * total_segments))  # At least 1 segment

                rng = np.random.RandomState(self.sampler_seed)
                subset_indices = sorted(rng.choice(total_segments, size=subset_size, replace=False))

                self.val_dataset = Subset(self.val_dataset, subset_indices)
                print(f"Debug mode: Using {subset_size}/{total_segments} validation segments (10% random sample)")

            print(f"Validation dataset size: {len(self.val_dataset)} segments\n")

    def train_dataloader(self) -> DataLoader:
        """
        Create training DataLoader with RecordingShuffleSampler.

        Returns
        -------
        DataLoader
            Training data loader with custom sampler and collate function
        """
        if self.use_recording_sampler:
            sampler = RecordingShuffleSampler(
                self.train_dataset,
                seed=self.sampler_seed,
            )
            shuffle = None
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=True,  # Drop incomplete batches for consistent training
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create validation DataLoader (sequential, no shuffling).

        Returns
        -------
        DataLoader
            Validation data loader with custom collate function
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )

    def teardown(self, stage: Optional[str] = None):
        """
        Cleanup HDF5 file handles when done.

        Parameters
        ----------
        stage : str, optional
            Either "fit", "validate", "test", or "predict"
        """
        if self.train_dataset is not None:
            self.train_dataset.close()
        if self.val_dataset is not None:
            # Handle case where val_dataset is wrapped in Subset (debug mode)
            if isinstance(self.val_dataset, Subset):
                self.val_dataset.dataset.close()
            else:
                self.val_dataset.close()


if __name__ == "__main__":
    """Test the DataModule."""
    import time
    import numpy as np

    print("Testing ArmeniMEGDataModule...")

    # Create DataModule
    datamodule = ArmeniMEGDataModule(
        data_root="/path/to/armeni2022",
        segment_length=1.0,
        subjects=["sub-001"],
        val_session="ses-010",
        tasks=["compr"],
        batch_size=4,
        num_workers=4,  # Use 4 for benchmarking
        debug_mode=True,
        h_freq=128.0,
        target_sfreq=256.0,
    )

    # Setup
    datamodule.setup("fit")

    # Get dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")

    # Test one batch
    print("\nTesting train batch...")
    for batch in train_loader:
        meg, sensor_xyz = batch
        print(f"  MEG shape: {meg.shape}")
        print(f"  Sensor XYZ shape: {sensor_xyz.shape}")
        break

    print("\nTesting val batch...")
    for batch in val_loader:
        meg, sensor_xyz = batch
        print(f"  MEG shape: {meg.shape}")
        print(f"  Sensor XYZ shape: {sensor_xyz.shape}")
        break

    # Benchmark dataloading time
    print("\n" + "=" * 80)
    print("BENCHMARKING DATALOADER")
    print("=" * 80)

    num_benchmark_batches = min(50, len(train_loader))
    print(f"\nBenchmarking {num_benchmark_batches} batches from train loader...")

    start_time = time.time()

    for i, batch in enumerate(train_loader):
        meg, sensor_xyz = batch
        if i >= num_benchmark_batches - 1:
            break
    total_time = time.time() - start_time

    # Calculate statistics
    mean_time = total_time / num_benchmark_batches

    print(f"\n=== Benchmark Results ===")
    print(f"Total batches: {num_benchmark_batches}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {num_benchmark_batches / total_time:.2f} batches/s")
    print(f"\nPer-batch timing:")
    print(f"  Mean:   {mean_time*1000:.2f}ms")

    # Calculate samples per second
    batch_size = meg.shape[0]
    samples_per_sec = (num_benchmark_batches * batch_size) / total_time
    print(f"\nData throughput:")
    print(f"  Batch size: {batch_size}")
    print(f"  Samples/s: {samples_per_sec:.2f}")
    print(f"  Data shape per sample: {meg.shape[1:]} (C, T)")

    # Cleanup
    datamodule.teardown("fit")
    print("\n✅ DataModule test and benchmark passed!")
