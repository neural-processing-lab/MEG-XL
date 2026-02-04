"""PyTorch Lightning DataModule for multi-dataset MEG pre-training."""

import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from typing import Optional, List, Dict, Callable, Any
import numpy as np

from .armeni_dataset import ArmeniMEGDataset
from .schoffelen_dataset import SchoffelenMEGDataset
from .gwilliams_dataset import GwilliamsMEGDataset
from .camcan_dataset import CamCANMEGDataset
from .libribrain_dataset import LibriBrainMEGDataset
from .smn4lang_dataset import SMN4LangMEGDataset
from .multi_dataset import MultiMEGDataset
from .subsampled_dataset import SubsampledRecordingDataset


class MultiMEGDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for multi-dataset MEG pre-training.

    This module handles:
    - Creating multiple datasets (Armeni, Schoffelen, etc.) with max_channel_dim=306
    - Combining datasets with MultiMEGDataset wrapper
    - Separate validation splits per dataset
    - RecordingShuffleSampler for efficient training with shuffled recordings
    - Custom collate function for batching with dataset tracking

    Parameters
    ----------
    datasets_config : List[Dict[str, Any]]
        List of dataset configurations. Each dict should contain:
        - type: str ("armeni", "schoffelen", "gwilliams", "camcan", "libribrain", or "smn4lang")
        - data_root: str (path to dataset)
        - subjects: List[str] or None (subjects to include)
        - tasks: List[str] or None (tasks to include)
        - validation_only: bool, optional (if True, skip training data; default: False)

        For Armeni datasets, also include:
        - val_session: str (session for validation, e.g., "ses-010")

        For LibriBrain datasets, also include:
        - val_session: str (session for validation, e.g., "ses-2")
        - sessions: List[str] or None (sessions to include)

        For SMN4Lang datasets, also include:
        - val_runs: List[str] (runs for validation, e.g., ["run-60"])
        - runs: List[str] or None (runs to include, e.g., ["run-1", "run-2"])

        For Schoffelen, Gwilliams, and CamCAN datasets, also include:
        - val_subjects: List[str] (subjects for validation)

    segment_length : float
        Length of each segment in seconds
    cache_dir : str
        Directory for storing preprocessed cache files
    l_freq : float
        Low frequency cutoff for band-pass filter
    h_freq : float
        High frequency cutoff for band-pass filter
    target_sfreq : float
        Target sampling frequency after resampling
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
        If True, uses minimal data for debugging

    Example
    -------
    >>> datamodule = MultiMEGDataModule(
    ...     datasets_config=[
    ...         {
    ...             "type": "armeni",
    ...             "data_root": "/path/to/armeni2022",
    ...             "subjects": ["sub-001", "sub-002"],
    ...             "tasks": ["compr"],
    ...             "val_session": "ses-010",
    ...         },
    ...         {
    ...             "type": "schoffelen",
    ...             "data_root": "/path/to/schoffelen2019",
    ...             "subjects": ["sub-A2002", "sub-A2003"],
    ...             "tasks": ["auditory"],
    ...             "val_subjects": ["sub-A2003"],
    ...         },
    ...         {
    ...             "type": "camcan",
    ...             "data_root": "/path/to/shafto2014/cc700/meg/pipeline/release005/BIDSsep",
    ...             "subjects": None,
    ...             "tasks": ["rest", "smt"],
    ...             "val_subjects": ["sub-CC110033", "sub-CC120065"],
    ...         },
    ...         {
    ...             "type": "libribrain",
    ...             "data_root": "/path/to/LibriBrain",
    ...             "subjects": None,
    ...             "sessions": None,
    ...             "tasks": None,
    ...             "val_session": "ses-2",
    ...         }
    ...     ],
    ...     segment_length=10.0,
    ...     batch_size=8,
    ... )
    >>> datamodule.setup("fit")
    >>> train_loader = datamodule.train_dataloader()
    >>> val_loader = datamodule.val_dataloader()
    """

    def __init__(
        self,
        datasets_config: List[Dict[str, Any]],
        segment_length: float,
        cache_dir: str = "./data/cache",
        l_freq: float = 0.1,
        h_freq: float = 40.0,
        target_sfreq: float = 50.0,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        use_recording_sampler: bool = True,
        sampler_seed: int = 42,
        debug_mode: bool = False,
        shuffle_segments: bool = False,
        shuffle_segment_duration: float = 3.0,
        recording_subsample_prop: Optional[float] = None,
    ):
        super().__init__()
        # Don't save channel_filter as it's not serializable
        self.save_hyperparameters()

        # Store parameters
        self.datasets_config = datasets_config
        self.segment_length = segment_length
        self.cache_dir = cache_dir
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.target_sfreq = target_sfreq
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.use_recording_sampler = use_recording_sampler
        self.sampler_seed = sampler_seed
        self.debug_mode = debug_mode
        self.shuffle_segments = shuffle_segments
        self.shuffle_segment_duration = shuffle_segment_duration
        self.recording_subsample_prop = recording_subsample_prop

        # Validate recording_subsample_prop
        if recording_subsample_prop is not None:
            if not (0.0 < recording_subsample_prop <= 1.0):
                raise ValueError(
                    f"recording_subsample_prop must be in (0.0, 1.0], got {recording_subsample_prop}"
                )

        # IMPORTANT: Always use max_channel_dim=306 for multi-dataset training
        self.max_channel_dim = 306

        # Datasets (initialized in setup)
        self.train_dataset: Optional[MultiMEGDataset] = None
        self.val_dataset: Optional[MultiMEGDataset] = None

    def _get_train_sessions_armeni(self, val_session: str) -> List[str]:
        """
        Get list of training sessions for Armeni dataset (all except val_session).

        Parameters
        ----------
        val_session : str
            Session to exclude (e.g., "ses-010")

        Returns
        -------
        train_sessions : List[str]
            List of session names for training
        """
        all_sessions = [f"ses-{i:03d}" for i in range(1, 13)]
        train_sessions = [s for s in all_sessions if s != val_session]
        return train_sessions

    def _get_train_sessions_libribrain(self, data_root: str, val_session: str) -> List[str]:
        """
        Get list of training sessions for LibriBrain dataset.

        Discovers all sessions by scanning h5 filenames in the serialized directory,
        then excludes val_session.

        Parameters
        ----------
        data_root : str
            Root directory of LibriBrain dataset
        val_session : str
            Session to exclude (e.g., "ses-2")

        Returns
        -------
        train_sessions : List[str]
            List of session names for training
        """
        import os
        import re

        # LibriBrain structure: data_root/serialized/{task}/derivatives/serialised/*.h5
        # Extract sessions from h5 filenames
        sessions = set()

        serialized_dir = os.path.join(data_root, "serialized")
        if not os.path.isdir(serialized_dir):
            raise ValueError(f"LibriBrain serialized directory not found: {serialized_dir}")

        # Scan all task directories
        for task_dir in os.listdir(serialized_dir):
            task_path = os.path.join(serialized_dir, task_dir)
            if not os.path.isdir(task_path) or task_dir.startswith('.'):
                continue

            # Look in derivatives/serialised subdirectory
            serialised_path = os.path.join(task_path, "derivatives", "serialised")
            if not os.path.isdir(serialised_path):
                continue

            # Parse h5 filenames to extract sessions
            for filename in os.listdir(serialised_path):
                if not filename.endswith('.h5'):
                    continue

                # Extract session from filename (e.g., "sub-0_ses-9_task-Sherlock1_...")
                session_match = re.search(r'ses-(\d+)', filename)
                if session_match:
                    sessions.add(f"ses-{session_match.group(1)}")

        # Remove validation session and return sorted list
        train_sessions = sorted([s for s in sessions if s != val_session])

        if len(train_sessions) == 0:
            raise ValueError(
                f"No training sessions found after excluding {val_session}. "
                f"Available sessions: {sorted(sessions)}"
            )

        return train_sessions

    def _subsample_by_recordings(self, dataset, proportion: float, seed: int):
        """
        Subsample dataset by randomly selecting a proportion of recordings.

        Parameters
        ----------
        dataset : Dataset
            Dataset with segment_index attribute
        proportion : float
            Proportion of recordings to keep (0.0 < proportion <= 1.0)
        seed : int
            Random seed for reproducibility

        Returns
        -------
        SubsampledRecordingDataset
            Wrapped dataset with only selected recordings
        """
        # Get unique recording indices
        recording_indices = sorted(set(rec_idx for rec_idx, _ in dataset.segment_index))
        n_recordings = len(recording_indices)
        n_to_keep = max(1, int(proportion * n_recordings))

        # Randomly select recordings
        rng = np.random.RandomState(seed)
        selected = set(rng.choice(recording_indices, size=n_to_keep, replace=False))

        print(f"Recording subsampling: {n_to_keep}/{n_recordings} recordings "
              f"({proportion*100:.1f}%)")

        return SubsampledRecordingDataset(dataset, selected)

    def _create_dataset(
        self,
        config: Dict[str, Any],
        split: str,  # "train" or "val"
    ) -> Any:
        """
        Create a single dataset instance based on configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Dataset configuration
        split : str
            Either "train" or "val"

        Returns
        -------
        Dataset
            ArmeniMEGDataset or SchoffelenMEGDataset instance
        """
        dataset_type = config["type"]
        data_root = config["data_root"]
        subjects = config.get("subjects", None)
        tasks = config.get("tasks", None)

        # Get channel filter for the dataset type
        if dataset_type == "armeni":
            channel_filter = lambda x: x.startswith('M')  # MEG channels only
        elif dataset_type == "schoffelen":
            from .schoffelen_dataset import SCHOFFELEN_VALID_CHANNELS
            channel_filter = lambda x: x.split('-')[0] in SCHOFFELEN_VALID_CHANNELS
        elif dataset_type == "gwilliams":
            # For Gwilliams, pick_types is used internally in preprocess_gwilliams_recording
            channel_filter = lambda x: x.startswith('MEG')
        elif dataset_type == "camcan":
            # For CamCAN, pick_types is used internally in preprocess_camcan
            channel_filter = lambda x: x.startswith('MEG')
        elif dataset_type == "libribrain":
            # For LibriBrain, use MEG prefix filter
            channel_filter = lambda x: x.startswith('MEG')
        elif dataset_type == "smn4lang":
            # For SMN4Lang, pick_types is used internally in preprocess_smn4lang_recording
            channel_filter = lambda x: x.startswith('MEG')
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        if dataset_type == "armeni":
            val_session = config.get("val_session", "ses-010")

            if split == "train":
                sessions = self._get_train_sessions_armeni(val_session)
                if self.debug_mode:
                    subjects = ["sub-001"]
                    sessions = ["ses-001"]
            else:  # val
                sessions = [val_session]
                if self.debug_mode:
                    subjects = ["sub-001"]

            return ArmeniMEGDataset(
                data_root=data_root,
                segment_length=self.segment_length,
                cache_dir=self.cache_dir,
                subjects=subjects,
                sessions=sessions,
                tasks=tasks,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                target_sfreq=self.target_sfreq,
                channel_filter=channel_filter,
                max_channel_dim=self.max_channel_dim,
                shuffle_segments=self.shuffle_segments if split == "train" else False,
                shuffle_segment_duration=self.shuffle_segment_duration,
            )

        elif dataset_type == "schoffelen":
            val_subjects = config.get("val_subjects", None)

            if split == "train":
                # Exclude validation subjects from training
                if val_subjects is not None and subjects is not None:
                    train_subjects = [s for s in subjects if s not in val_subjects]
                else:
                    train_subjects = subjects

                if self.debug_mode:
                    train_subjects = ["sub-A2002"]

                dataset_subjects = train_subjects
            else:  # val
                if self.debug_mode:
                    dataset_subjects = ["sub-A2003"]
                else:
                    dataset_subjects = val_subjects

            return SchoffelenMEGDataset(
                data_root=data_root,
                segment_length=self.segment_length,
                cache_dir=self.cache_dir,
                subjects=dataset_subjects,
                tasks=tasks,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                target_sfreq=self.target_sfreq,
                channel_filter=channel_filter,
                max_channel_dim=self.max_channel_dim,
                shuffle_segments=self.shuffle_segments if split == "train" else False,
                shuffle_segment_duration=self.shuffle_segment_duration,
            )

        elif dataset_type == "gwilliams":
            val_subjects = config.get("val_subjects", None)
            sessions = config.get("sessions", None)

            if split == "train":
                # Exclude validation subjects from training
                if val_subjects is not None and subjects is not None:
                    train_subjects = [s for s in subjects if s not in val_subjects]
                else:
                    train_subjects = subjects

                if self.debug_mode:
                    train_subjects = ["sub-01"]

                dataset_subjects = train_subjects
            else:  # val
                if self.debug_mode:
                    dataset_subjects = ["sub-03"]
                else:
                    dataset_subjects = val_subjects

            return GwilliamsMEGDataset(
                data_root=data_root,
                segment_length=self.segment_length,
                cache_dir=self.cache_dir,
                subjects=dataset_subjects,
                sessions=sessions,
                tasks=tasks,
                val_subjects=val_subjects if split == "train" else None,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                target_sfreq=self.target_sfreq,
                channel_filter=channel_filter,
                max_channel_dim=self.max_channel_dim,
                shuffle_segments=self.shuffle_segments if split == "train" else False,
                shuffle_segment_duration=self.shuffle_segment_duration,
            )

        elif dataset_type == "camcan":
            val_subjects = config.get("val_subjects", None)
            sessions = config.get("sessions", None)

            if split == "train":
                # Exclude validation subjects from training
                if val_subjects is not None and subjects is not None:
                    train_subjects = [s for s in subjects if s not in val_subjects]
                else:
                    train_subjects = subjects

                if self.debug_mode:
                    train_subjects = ["sub-CC110033"] if train_subjects else ["sub-CC110033"]

                dataset_subjects = train_subjects
            else:  # val
                if self.debug_mode:
                    dataset_subjects = ["sub-CC120065"] if val_subjects else ["sub-CC120065"]
                else:
                    dataset_subjects = val_subjects

            return CamCANMEGDataset(
                data_root=data_root,
                segment_length=self.segment_length,
                cache_dir=self.cache_dir,
                subjects=dataset_subjects,
                sessions=sessions,
                tasks=tasks,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                target_sfreq=self.target_sfreq,
                channel_filter=channel_filter,
                max_channel_dim=self.max_channel_dim,
                shuffle_segments=self.shuffle_segments if split == "train" else False,
                shuffle_segment_duration=self.shuffle_segment_duration,
            )

        elif dataset_type == "libribrain":
            val_session = config.get("val_session", "ses-2")
            sessions = config.get("sessions", None)

            if split == "train":
                # Exclude validation session from training
                if sessions is None:
                    # Need to discover all sessions and exclude val_session
                    train_sessions = self._get_train_sessions_libribrain(
                        data_root=data_root,
                        val_session=val_session
                    )
                else:
                    # User provided explicit session list
                    train_sessions = [s for s in sessions if s != val_session]

                if self.debug_mode:
                    subjects = ["sub-0"]
                    train_sessions = ["ses-1"]
                    tasks = ["Sherlock1"]

                dataset_sessions = train_sessions
            else:  # val
                dataset_sessions = [val_session]
                tasks = ["Sherlock1"]
                if self.debug_mode:
                    subjects = ["sub-0"]
                    tasks = ["Sherlock1"]

            return LibriBrainMEGDataset(
                data_root=data_root,
                segment_length=self.segment_length,
                cache_dir=self.cache_dir,
                subjects=subjects,
                sessions=dataset_sessions,
                tasks=tasks,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                target_sfreq=self.target_sfreq,
                channel_filter=channel_filter,
                max_channel_dim=self.max_channel_dim,
                shuffle_segments=self.shuffle_segments if split == "train" else False,
                shuffle_segment_duration=self.shuffle_segment_duration,
            )

        elif dataset_type == "smn4lang":
            val_runs = config.get("val_runs", None)
            runs = config.get("runs", None)

            if split == "train":
                # Exclude validation runs from training
                if val_runs is not None and runs is not None:
                    train_runs = [r for r in runs if r not in val_runs]
                else:
                    train_runs = runs

                if self.debug_mode:
                    train_runs = ["run-1"]

                dataset_runs = train_runs
            else:  # val
                if self.debug_mode:
                    dataset_runs = ["run-2"]
                else:
                    dataset_runs = val_runs

            return SMN4LangMEGDataset(
                data_root=data_root,
                segment_length=self.segment_length,
                cache_dir=self.cache_dir,
                subjects=subjects,
                runs=dataset_runs,
                tasks=tasks,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                target_sfreq=self.target_sfreq,
                channel_filter=channel_filter,
                max_channel_dim=self.max_channel_dim,
                shuffle_segments=self.shuffle_segments if split == "train" else False,
                shuffle_segment_duration=self.shuffle_segment_duration,
            )

        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for batching MEG segments from multiple datasets.

        Stacks MEG data, sensor positions, orientations, types, and masks from
        heterogeneous datasets. Returns dataset IDs for tracking which dataset
        each sample came from.

        Parameters
        ----------
        batch : List[Dict[str, Any]]
            List of samples from dataset.__getitem__

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            - meg: [B, C_padded, T] stacked MEG signals
            - sensor_xyzdir: [B, C_padded, 6] sensor positions and directions
            - sensor_types: [B, C_padded] sensor types
            - sensor_mask: [B, C_padded] sensor mask
            - dataset_ids: [B] dataset indices (0, 1, 2, ...)
        """
        meg = torch.stack([item["meg"] for item in batch])
        sensor_xyzdir = torch.stack([item["sensor_xyzdir"] for item in batch])
        sensor_types = torch.stack([item["sensor_types"] for item in batch])
        sensor_mask = torch.stack([item["sensor_mask"] for item in batch])
        dataset_ids = torch.tensor([item["dataset_idx"] for item in batch], dtype=torch.long)

        return (meg, sensor_xyzdir, sensor_types, sensor_mask, dataset_ids)

    def setup(self, stage: Optional[str] = None):
        """
        Setup train and validation datasets.

        Parameters
        ----------
        stage : str, optional
            Either "fit", "validate", "test", or "predict"
        """
        if stage == "fit" or stage is None:
            print("\n" + "=" * 80)
            print("MULTI-DATASET SETUP")
            print("=" * 80)
            print(f"Max channel dim: {self.max_channel_dim}")
            print(f"Number of datasets: {len(self.datasets_config)}")
            if self.debug_mode:
                print("🐛 DEBUG MODE ENABLED 🐛")
            print("")

            # Create training datasets
            train_datasets = []
            dataset_names = []

            for i, config in enumerate(self.datasets_config):
                # Check validation_only flag
                is_validation_only = config.get("validation_only", False)

                if is_validation_only:
                    print(f"\n=== Training Dataset {i+1}: {config['type']} ===")
                    print(f"  SKIPPED (validation_only=True)")
                    continue

                print(f"\n=== Training Dataset {i+1}: {config['type']} ===")
                dataset = self._create_dataset(config, split="train")

                train_datasets.append(dataset)
                dataset_names.append(config['type'])
                print(f"  Segments: {len(dataset)}")

            # Ensure at least one training dataset exists
            if len(train_datasets) == 0:
                raise ValueError(
                    "No training datasets configured! All datasets have validation_only=True. "
                    "At least one dataset must be available for training."
                )

            # Combine training datasets
            self.train_dataset = MultiMEGDataset(
                datasets=train_datasets,
                dataset_names=dataset_names,
            )

            print(f"\n{'=' * 80}")
            print(f"Total training segments: {len(self.train_dataset)}")
            print(f"{'=' * 80}")

            # Apply recording subsampling if configured
            if self.recording_subsample_prop is not None:
                original_segments = len(self.train_dataset)
                self.train_dataset = self._subsample_by_recordings(
                    self.train_dataset,
                    proportion=self.recording_subsample_prop,
                    seed=self.sampler_seed,
                )
                print(f"Segments after subsampling: {len(self.train_dataset)} "
                      f"(was {original_segments})")

            # Create validation datasets
            val_datasets = []

            for i, config in enumerate(self.datasets_config):
                print(f"\n=== Validation Dataset {i+1}: {config['type']} ===")
                dataset = self._create_dataset(config, split="val")
                val_datasets.append(dataset)
                print(f"  Segments: {len(dataset)}")

            # Build validation dataset names (includes ALL datasets)
            val_dataset_names = [config['type'] for config in self.datasets_config]

            # Combine validation datasets
            self.val_dataset = MultiMEGDataset(
                datasets=val_datasets,
                dataset_names=val_dataset_names,
            )

            # In debug mode, subsample validation to 10%
            if self.debug_mode:
                total_segments = len(self.val_dataset)
                subset_size = max(1, int(0.1 * total_segments))

                rng = np.random.RandomState(self.sampler_seed)
                subset_indices = sorted(rng.choice(total_segments, size=subset_size, replace=False))

                self.val_dataset = Subset(self.val_dataset, subset_indices)
                print(f"\nDebug mode: Using {subset_size}/{total_segments} validation segments (10% random sample)")

            print(f"\n{'=' * 80}")
            print(f"Total validation segments: {len(self.val_dataset)}")
            print(f"{'=' * 80}\n")

    def train_dataloader(self) -> DataLoader:
        """
        Create training DataLoader with RecordingShuffleSampler.

        Lightning automatically wraps the sampler with DistributedSampler when
        using DDP/FSDP, which partitions the shuffled recording indices across GPUs.
        This keeps most recordings together on single GPUs for I/O efficiency.

        Returns
        -------
        DataLoader
            Training data loader with custom sampler and collate function
        """
        if self.use_recording_sampler:
            from .samplers import RecordingShuffleSampler
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

    def get_dataset_name_mapping(self) -> Dict[int, str]:
        """
        Get mapping from dataset ID to dataset name for validation logging.

        Returns
        -------
        Dict[int, str]
            Mapping from dataset_idx (0, 1, 2, ...) to dataset_name ("armeni", "schoffelen", etc.)
        """
        if self.val_dataset is None:
            return {}

        # Handle case where val_dataset is wrapped in Subset (debug mode)
        if isinstance(self.val_dataset, Subset):
            dataset = self.val_dataset.dataset
        else:
            dataset = self.val_dataset

        # Build mapping from dataset_names list
        return {i: name for i, name in enumerate(dataset.dataset_names)}

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
    """Test the MultiMEGDataModule."""
    import time

    print("Testing MultiMEGDataModule...")

    # Create DataModule
    datamodule = MultiMEGDataModule(
        datasets_config=[
            {
                "type": "armeni",
                "data_root": "/path/to/armeni2022",
                "subjects": ["sub-001"],
                "tasks": ["compr"],
                "val_session": "ses-010",
            },
            {
                "type": "schoffelen",
                "data_root": "/path/to/schoffelen2019",
                "subjects": ["sub-A2002"],
                "tasks": ["auditory"],
                "val_subjects": ["sub-A2002"],
            }
        ],
        segment_length=1.0,
        batch_size=4,
        num_workers=4,
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
        meg, sensor_xyzdir, sensor_types, sensor_mask, dataset_ids = batch
        print(f"  MEG shape: {meg.shape}")
        print(f"  Sensor xyzdir shape: {sensor_xyzdir.shape}")
        print(f"  Sensor types shape: {sensor_types.shape}")
        print(f"  Sensor mask shape: {sensor_mask.shape}")
        print(f"  Dataset IDs: {dataset_ids}")
        print(f"  Unique datasets in batch: {torch.unique(dataset_ids).tolist()}")
        break

    print("\nTesting val batch...")
    for batch in val_loader:
        meg, sensor_xyzdir, sensor_types, sensor_mask, dataset_ids = batch
        print(f"  MEG shape: {meg.shape}")
        print(f"  Sensor xyzdir shape: {sensor_xyzdir.shape}")
        print(f"  Sensor types shape: {sensor_types.shape}")
        print(f"  Sensor mask shape: {sensor_mask.shape}")
        print(f"  Dataset IDs: {dataset_ids}")
        break

    # Benchmark dataloading time
    print("\n" + "=" * 80)
    print("BENCHMARKING DATALOADER")
    print("=" * 80)

    num_benchmark_batches = min(50, len(train_loader))
    print(f"\nBenchmarking {num_benchmark_batches} batches from train loader...")

    start_time = time.time()

    for i, batch in enumerate(train_loader):
        meg, sensor_xyzdir, sensor_types, sensor_mask, dataset_ids = batch
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
    print("\n✅ MultiMEGDataModule test and benchmark passed!")
