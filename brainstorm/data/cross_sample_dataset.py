"""Cross-sample mixing dataset wrapper for MEG data.

This module provides a dataset wrapper that creates 150s samples by randomly
mixing 3s segments from across all recordings within a single dataset type.
This breaks temporal continuity at the sample level while preserving sensor
layout consistency.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Any, List, Tuple

from .preprocessing import preprocess_segment_with_subsegments
from .utils import norm_sensor_positions


class CrossSampleMixingDataset(Dataset):
    """
    Wrapper that creates samples from randomly mixed segments across recordings.

    Instead of returning contiguous 150s segments from a single recording, this
    dataset creates each sample by randomly selecting 50 x 3s segments from across
    all recordings in the wrapped dataset and concatenating them.

    This is useful for ablation experiments to test whether temporal continuity
    within a sample matters for model performance.

    Parameters
    ----------
    base_dataset : Dataset
        The underlying dataset to wrap (e.g., ArmeniMEGDataset, SchoffelenMEGDataset).
        Must have:
        - file_handles: List of HDF5 file handles
        - target_sfreq: float (sampling frequency)
        - max_channel_dim: int or None (for channel padding)
        - segment_length: float (total sample duration in seconds)
    segment_duration : float
        Total duration of each output sample in seconds (default: 150.0)
    subsegment_duration : float
        Duration of each randomly selected segment in seconds (default: 3.0)

    Example
    -------
    >>> base_dataset = ArmeniMEGDataset(...)
    >>> mixed_dataset = CrossSampleMixingDataset(base_dataset)
    >>> sample = mixed_dataset[0]  # Returns 150s sample from 50 random 3s segments
    """

    def __init__(
        self,
        base_dataset: Dataset,
        segment_duration: float = 150.0,
        subsegment_duration: float = 3.0,
    ):
        self.base_dataset = base_dataset
        self.segment_duration = segment_duration
        self.subsegment_duration = subsegment_duration
        self.sfreq = base_dataset.target_sfreq  # 50 Hz
        self.max_channel_dim = base_dataset.max_channel_dim

        # Calculate samples per subsegment: 3.0s * 50Hz = 150 samples
        self.samples_per_subsegment = int(subsegment_duration * self.sfreq)
        # Number of subsegments per output sample: 150s / 3s = 50
        self.n_subsegments_per_sample = int(segment_duration / subsegment_duration)

        # Build index of all available subsegments across all recordings
        self.subsegment_index = self._build_subsegment_index()

        print(f"CrossSampleMixingDataset initialized:")
        print(f"  - {len(self.base_dataset.file_handles)} recordings")
        print(f"  - {len(self.subsegment_index)} total {subsegment_duration}s subsegments")
        print(f"  - Each sample composed of {self.n_subsegments_per_sample} random subsegments")

    def _build_subsegment_index(self) -> List[Tuple[int, int, int]]:
        """
        Build an index of all available subsegments across all recordings.

        Returns
        -------
        index : List[Tuple[int, int, int]]
            List of (recording_idx, start_sample, end_sample) tuples
        """
        index = []
        for rec_idx, h5_file in enumerate(self.base_dataset.file_handles):
            n_samples = h5_file.attrs["n_samples"]
            n_subsegments = n_samples // self.samples_per_subsegment
            for subseg_idx in range(n_subsegments):
                start = subseg_idx * self.samples_per_subsegment
                end = start + self.samples_per_subsegment
                index.append((rec_idx, start, end))
        return index

    def __len__(self) -> int:
        """Return number of virtual samples (same as base dataset for compatibility)."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample composed of randomly mixed subsegments.

        Parameters
        ----------
        idx : int
            Sample index (used only to maintain compatibility with DataLoader;
            the actual content is randomly composed each time)

        Returns
        -------
        sample : Dict[str, Any]
            Dictionary containing:
            - meg: torch.Tensor of shape (n_channels, n_timepoints)
            - subject: str (from first selected recording)
            - session: str (from first selected recording)
            - task: str (from first selected recording)
            - sensor_xyzdir: torch.Tensor of shape (n_channels, 6)
            - sensor_types: torch.Tensor of shape (n_channels,)
            - sensor_mask: torch.Tensor of shape (n_channels,)
            - start_time: float (always 0.0 for mixed samples)
            - end_time: float (segment_duration)
            - recording_idx: int (first selected recording)
            - segment_idx: int (always -1 for mixed samples)
            - is_cross_mixed: bool (True)
        """
        # Randomly select n_subsegments_per_sample (50) subsegments
        selected_indices = np.random.choice(
            len(self.subsegment_index),
            size=self.n_subsegments_per_sample,
            replace=True  # Allow same subsegment to appear multiple times
        )

        # Get sensor types from first recording (same for all within dataset)
        first_rec_idx = self.subsegment_index[selected_indices[0]][0]
        first_h5_file = self.base_dataset.file_handles[first_rec_idx]
        sensor_types = first_h5_file["sensor_types"][:]
        n_channels = sensor_types.shape[0]

        # Load and preprocess each subsegment
        subsegments = []
        for subseg_global_idx in selected_indices:
            rec_idx, start, end = self.subsegment_index[subseg_global_idx]
            h5_file = self.base_dataset.file_handles[rec_idx]

            # Load the 3s chunk
            meg_chunk = h5_file["data"][:, start:end]

            # Preprocess the 3s chunk independently
            # Since subsegment_duration equals chunk size, this processes as single chunk
            meg_chunk = preprocess_segment_with_subsegments(
                meg_data=meg_chunk,
                sensor_types=sensor_types,
                sfreq=self.sfreq,
                subsegment_duration=self.subsegment_duration,
                baseline_duration=0.5,
                clip_range=(-5, 5)
            )
            subsegments.append(meg_chunk)

        # Concatenate all subsegments into one 150s sample
        meg_data = np.concatenate(subsegments, axis=1)

        # Get sensor positions from first recording (same for all within dataset)
        sensor_xyzdir = first_h5_file["sensor_xyzdir"][:]
        sensor_xyzdir = norm_sensor_positions(sensor_xyzdir.copy())

        # Pad channel dimension if needed
        if self.max_channel_dim is not None:
            n_channels_original = meg_data.shape[0]
            meg_data = np.pad(
                meg_data,
                ((0, self.max_channel_dim - n_channels_original), (0, 0))
            )
            sensor_xyzdir = np.pad(
                sensor_xyzdir,
                ((0, self.max_channel_dim - n_channels_original), (0, 0))
            )
            sensor_types = np.pad(
                sensor_types,
                (0, self.max_channel_dim - n_channels_original)
            )
            sensor_mask = np.zeros(self.max_channel_dim, dtype=np.float32)
            sensor_mask[:n_channels_original] = 1.0
        else:
            sensor_mask = np.ones(n_channels, dtype=np.float32)

        # Convert to torch tensors
        meg_tensor = torch.from_numpy(meg_data).float()
        sensor_xyzdir_tensor = torch.from_numpy(sensor_xyzdir).float()
        sensor_mask_tensor = torch.from_numpy(sensor_mask).float()
        sensor_types_tensor = torch.from_numpy(sensor_types).int()

        return {
            "meg": meg_tensor,
            "subject": first_h5_file.attrs["subject"],
            "session": first_h5_file.attrs["session"],
            "task": first_h5_file.attrs["task"],
            "sensor_xyzdir": sensor_xyzdir_tensor,
            "sensor_types": sensor_types_tensor,
            "start_time": 0.0,
            "end_time": float(self.segment_duration),
            "recording_idx": first_rec_idx,
            "segment_idx": -1,  # Indicates mixed sample
            "sensor_mask": sensor_mask_tensor,
            "is_cross_mixed": True,
        }

    @property
    def segment_index(self):
        """Compatibility property for RecordingShuffleSampler."""
        return self.base_dataset.segment_index

    @property
    def file_handles(self):
        """Compatibility property - expose base dataset's file handles."""
        return self.base_dataset.file_handles

    @property
    def recordings(self):
        """Compatibility property - expose base dataset's recordings."""
        return self.base_dataset.recordings
