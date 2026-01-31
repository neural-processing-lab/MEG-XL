"""Multi-dataset wrapper for combining MEG datasets."""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple
import numpy as np


class MultiMEGDataset(Dataset):
    """
    Wrapper that combines multiple MEG datasets (e.g., Armeni + Schoffelen).

    This dataset enables multi-dataset pre-training by combining datasets while
    maintaining compatibility with RecordingShuffleSampler for efficient I/O.

    Features:
    - Maintains segment_index attribute for RecordingShuffleSampler compatibility
    - Handles heterogeneous channel counts via padding to max_channel_dim
    - Preserves dataset identity in metadata for tracking
    - Supports mixed batches containing samples from different datasets

    Parameters
    ----------
    datasets : List[Dataset]
        List of MEG dataset instances (e.g., [ArmeniMEGDataset, SchoffelenMEGDataset])
        Each dataset must have:
        - segment_index: List[Tuple[int, int]] mapping to (recording_idx, segment_idx)
        - recordings: List of recording metadata
        - __getitem__ returning dict with keys: meg, subject, session, task, sensor_xyz, etc.
    dataset_names : List[str], optional
        Names for each dataset (e.g., ["armeni", "schoffelen"]). If None, uses indices.

    Example
    -------
    >>> armeni_dataset = ArmeniMEGDataset(..., max_channel_dim=306)
    >>> schoffelen_dataset = SchoffelenMEGDataset(..., max_channel_dim=306)
    >>> multi_dataset = MultiMEGDataset(
    ...     datasets=[armeni_dataset, schoffelen_dataset],
    ...     dataset_names=["armeni", "schoffelen"]
    ... )
    >>> sample = multi_dataset[0]
    >>> print(sample['dataset_name'])  # 'armeni' or 'schoffelen'
    """

    def __init__(
        self,
        datasets: List[Dataset],
        dataset_names: List[str] = None,
    ):
        self.datasets = datasets
        self.dataset_names = dataset_names or [f"dataset_{i}" for i in range(len(datasets))]

        if len(self.datasets) != len(self.dataset_names):
            raise ValueError(
                f"Number of datasets ({len(self.datasets)}) must match "
                f"number of dataset names ({len(self.dataset_names)})"
            )

        # Rebuild segment_index for RecordingShuffleSampler compatibility
        # Format: [(adjusted_rec_idx, seg_idx), ...] (2-tuples for sampler compatibility)
        # Store dataset identity separately
        self.segment_index = []
        self.segment_to_dataset = []  # Maps global_idx -> dataset_idx
        self.cumulative_recordings = [0]
        recording_offset = 0

        for dataset_idx, dataset in enumerate(self.datasets):
            for rec_idx, seg_idx in dataset.segment_index:
                # Store 2-tuple for sampler compatibility
                self.segment_index.append((
                    rec_idx + recording_offset,
                    seg_idx
                ))
                # Store dataset identity separately
                self.segment_to_dataset.append(dataset_idx)

            recording_offset += len(dataset.recordings)
            self.cumulative_recordings.append(recording_offset)


    def _get_dataset_local_idx(self, dataset_idx: int, rec_idx_local: int, seg_idx: int) -> int:
        """
        Convert (recording_idx, segment_idx) to a global index within a specific dataset.

        Parameters
        ----------
        dataset_idx : int
            Index of the dataset
        rec_idx_local : int
            Recording index within the dataset (not adjusted)
        seg_idx : int
            Segment index within the recording

        Returns
        -------
        int
            Global index within the specific dataset
        """
        dataset = self.datasets[dataset_idx]

        # Find the global index in the dataset that corresponds to this (rec_idx, seg_idx)
        for global_idx, (r_idx, s_idx) in enumerate(dataset.segment_index):
            if r_idx == rec_idx_local and s_idx == seg_idx:
                return global_idx

        raise ValueError(
            f"Could not find segment (rec={rec_idx_local}, seg={seg_idx}) "
            f"in dataset {dataset_idx}"
        )

    def __len__(self) -> int:
        """Return total number of segments across all datasets."""
        return len(self.segment_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single segment from the combined dataset.

        Parameters
        ----------
        idx : int
            Global segment index across all datasets

        Returns
        -------
        sample : Dict[str, Any]
            Dictionary containing all fields from the underlying dataset, plus:
            - dataset_idx: int (0, 1, 2, ...)
            - dataset_name: str (e.g., "armeni", "schoffelen")
        """
        # Get adjusted recording index and segment index from segment_index
        rec_idx_adjusted, seg_idx = self.segment_index[idx]

        # Get dataset index from separate mapping
        dataset_idx = self.segment_to_dataset[idx]

        # Get original recording index for the specific dataset
        recording_offset = self.cumulative_recordings[dataset_idx]
        rec_idx_local = rec_idx_adjusted - recording_offset

        # Get the dataset
        dataset = self.datasets[dataset_idx]

        # Find the global index in the specific dataset
        dataset_local_idx = self._get_dataset_local_idx(dataset_idx, rec_idx_local, seg_idx)

        # Get the sample from the dataset
        sample = dataset[dataset_local_idx]

        # Add dataset identity metadata
        sample['dataset_idx'] = dataset_idx
        sample['dataset_name'] = self.dataset_names[dataset_idx]

        return sample

    def close(self):
        """Close all underlying datasets."""
        for dataset in self.datasets:
            if hasattr(dataset, 'close'):
                dataset.close()

    def __del__(self):
        """Cleanup when destroyed."""
        self.close()


if __name__ == "__main__":
    """Test MultiMEGDataset with dummy data."""
    from .armeni_dataset import ArmeniMEGDataset
    from .schoffelen_dataset import SchoffelenMEGDataset

    print("Testing MultiMEGDataset...")

    # Create individual datasets
    armeni = ArmeniMEGDataset(
        data_root="/path/to/armeni2022",
        segment_length=1.0,
        subjects=["sub-001"],
        sessions=["ses-001"],
        tasks=["compr"],
        max_channel_dim=306,
    )

    schoffelen = SchoffelenMEGDataset(
        data_root="/path/to/schoffelen2019",
        segment_length=1.0,
        subjects=["sub-A2002"],
        tasks=["auditory"],
        max_channel_dim=306,
    )

    # Combine datasets
    multi_dataset = MultiMEGDataset(
        datasets=[armeni, schoffelen],
        dataset_names=["armeni", "schoffelen"]
    )

    print(f"\nArmeni dataset: {len(armeni)} segments")
    print(f"Schoffelen dataset: {len(schoffelen)} segments")
    print(f"Multi dataset: {len(multi_dataset)} segments")

    # Test sampling
    print("\nTesting samples:")
    for i in [0, len(armeni), len(multi_dataset) - 1]:
        sample = multi_dataset[i]
        print(f"  Sample {i}: dataset={sample['dataset_name']}, "
              f"subject={sample['subject']}, meg.shape={sample['meg'].shape}")

    # Cleanup
    multi_dataset.close()
    print("\n✅ MultiMEGDataset test passed!")
