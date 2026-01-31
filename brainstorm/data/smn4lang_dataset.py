"""PyTorch Dataset for the ds004078 (Shain et al., 2020) MEG dataset."""

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import mne
from .utils import norm_sensor_positions

from .preprocessing import (
    cache_preprocessed,
    load_cached,
    get_cache_path,
    preprocess_segment_with_subsegments,
    shuffle_temporal_segments
)


def preprocess_smn4lang_recording(
    raw_path: str,
    l_freq: float = 0.1,
    h_freq: float = 40.0,
    target_sfreq: float = 50.0,
    channel_filter: Callable[[str], bool] = lambda _: True
) -> mne.io.Raw:
    """
    Preprocess a single MEG recording.

    Pipeline:
    1. Load raw data (FIF format)
    2. Pick only MEG channels (not ref_meg)
    3. Band-pass filter [l_freq, h_freq] Hz
    4. Resample to target_sfreq Hz

    Parameters
    ----------
    raw_path : str
        Path to the raw MEG file (.fif file)
    l_freq : float
        Low frequency cutoff for band-pass filter (default: 0.1 Hz)
    h_freq : float
        High frequency cutoff for band-pass filter (default: 40.0 Hz)
    target_sfreq : float
        Target sampling frequency after resampling (default: 50.0 Hz)
    channel_filter : Callable[[str], bool]
        Filter function for channels (not used for FIF, pick_types handles MEG selection)

    Returns
    -------
    raw : mne.io.Raw
        Preprocessed raw MEG data
    """
    # Load raw data
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)

    # First, use MNE's pick_types to get only MEG channels (not ref_meg)
    meg_picks = mne.pick_types(raw.info, meg=True, ref_meg=False, exclude=[])
    raw.pick(meg_picks)

    # Band-pass filter
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False, n_jobs=-1)

    # Resample
    raw.resample(sfreq=target_sfreq, verbose=False, n_jobs=-1)

    return raw


class SMN4LangMEGDataset(Dataset):
    """
    PyTorch Dataset for the ds004078 (Shain et al., 2020) MEG dataset.

    This dataset handles:
    - Discovery of MEG recordings from the ds004078 dataset
    - Lazy preprocessing with caching (band-pass filter, resample, channel selection)
    - Segmentation of continuous recordings into fixed-length windows
    - Efficient loading using persistent HDF5 file handles

    Note: This dataset treats "runs" as separate recordings, similar to how sessions
    are handled in the Armeni dataset. Each run (run-1 through run-60) is a separate
    recording file.

    Parameters
    ----------
    data_root : str
        Root directory of the ds004078 dataset (e.g., "/path/to/ds004078")
    segment_length : float
        Length of each segment in seconds
    cache_dir : str, optional
        Directory for storing preprocessed cache files (default: "./data/cache_smn4lang")
    subjects : List[str], optional
        List of subjects to include (e.g., ["sub-01", "sub-02"]). If None, use all.
    runs : List[str], optional
        List of runs to include (e.g., ["run-1", "run-2"]). If None, use all.
    tasks : List[str], optional
        List of tasks to include (e.g., ["RDR"]). If None, use all.
    l_freq : float
        Low frequency cutoff for band-pass filter (default: 0.1 Hz)
    h_freq : float
        High frequency cutoff for band-pass filter (default: 40.0 Hz)
    target_sfreq : float
        Target sampling frequency after resampling (default: 50.0 Hz)
    channel_filter : Callable[[str], bool]
        Filter function for channels. Channels for which this function returns True will be kept.
    max_channel_dim : int, optional
        Maximum channel dimension for padding. If None, no padding is applied.

    Example
    -------
    >>> dataset = SMN4LangMEGDataset(
    ...     data_root="/path/to/ds004078",
    ...     segment_length=30.0,
    ...     subjects=["sub-01"],
    ...     runs=["run-1", "run-2"],
    ...     tasks=["RDR"]
    ... )
    >>> sample = dataset[0]
    >>> print(sample['meg'].shape)  # (n_channels, n_timepoints)
    """

    def __init__(
        self,
        data_root: str,
        segment_length: float,
        cache_dir: str = "./data/cache_smn4lang",
        subjects: Optional[List[str]] = None,
        runs: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        l_freq: float = 0.1,
        h_freq: float = 40.0,
        target_sfreq: float = 50.0,
        channel_filter: Callable[[str], bool] = lambda x: x.startswith('MEG'),
        max_channel_dim: Optional[int] = None,
        shuffle_segments: bool = False,
        shuffle_segment_duration: float = 3.0,
    ):
        self.data_root = Path(data_root)
        self.segment_length = segment_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.l_freq = l_freq
        self.h_freq = h_freq
        self.target_sfreq = target_sfreq
        self.channel_filter = channel_filter
        self.max_channel_dim = max_channel_dim
        self.shuffle_segments = shuffle_segments
        self.shuffle_segment_duration = shuffle_segment_duration

        # Filters
        self.subjects = subjects
        self.runs = runs
        self.tasks = tasks

        # Discover all recordings
        self.recordings = self._discover_recordings()

        if len(self.recordings) == 0:
            raise ValueError(
                f"No recordings found in {self.data_root} with the specified filters. "
                f"Subjects: {subjects}, Runs: {runs}, Tasks: {tasks}"
            )

        # Preprocess and cache all recordings
        self._preprocess_all()

        # Open file handles for all cached recordings
        self.file_handles: List[h5py.File] = []
        self._open_file_handles()

        # Build segment index: maps global index -> (recording_idx, segment_idx)
        self.segment_index = self._build_segment_index()

    def _discover_recordings(self) -> List[Dict[str, Any]]:
        """
        Discover all MEG recordings matching the specified filters.

        Returns
        -------
        recordings : List[Dict[str, Any]]
            List of recording metadata dictionaries
        """
        recordings = []

        # Iterate through subject directories
        subject_dirs = sorted(self.data_root.glob("sub-*"))

        for subject_dir in subject_dirs:
            subject = subject_dir.name

            # Apply subject filter
            if self.subjects is not None and subject not in self.subjects:
                continue

            # Look for MEG data (no session directory in ds004078)
            meg_dir = subject_dir / "meg"
            if not meg_dir.exists():
                continue

            # Find task MEG files with runs
            # Pattern: sub-XX_task-RDR_run-YY_meg.fif
            meg_files = sorted(meg_dir.glob(f"{subject}_task-*_run-*_meg.fif"))

            for meg_file in meg_files:
                # Extract task and run from filename
                # Format: sub-XX_task-TASK_run-YY_meg.fif
                parts = meg_file.name.split("_")
                task = None
                run = None

                for part in parts:
                    if part.startswith("task-"):
                        task = part.replace("task-", "")
                    elif part.startswith("run-"):
                        run = part  # Keep "run-" prefix for consistency

                if task is None or run is None:
                    continue

                # Apply task filter
                if self.tasks is not None and task not in self.tasks:
                    continue

                # Apply run filter
                if self.runs is not None and run not in self.runs:
                    continue

                # Generate cache path (pass run as session for compatibility)
                cache_path = get_cache_path(
                    self.cache_dir, subject, run, task,
                    l_freq=self.l_freq,
                    h_freq=self.h_freq,
                    target_sfreq=self.target_sfreq,
                    channel_filter_name="MEG_only"
                )

                recordings.append({
                    "subject": subject,
                    "run": run,
                    "task": task,
                    "raw_path": meg_file,
                    "cache_path": cache_path
                })

        return recordings

    def _preprocess_all(self) -> None:
        """
        Preprocess all recordings that haven't been cached yet.
        """
        for i, rec in enumerate(self.recordings):
            if not rec["cache_path"].exists():
                print(f"Preprocessing recording {i+1}/{len(self.recordings)}: "
                      f"{rec['subject']} {rec['task']} {rec['run']}")

                # Preprocess
                raw = preprocess_smn4lang_recording(
                    str(rec["raw_path"]),
                    l_freq=self.l_freq,
                    h_freq=self.h_freq,
                    target_sfreq=self.target_sfreq,
                    channel_filter=self.channel_filter
                )

                # Cache (store run as "session" for compatibility)
                metadata = {
                    "subject": rec["subject"],
                    "session": rec["run"],  # Map run to session for cache compatibility
                    "task": rec["task"],
                    "dataset": "smn4lang"
                }
                cache_preprocessed(
                    raw, rec["cache_path"], metadata,
                    l_freq=self.l_freq,
                    h_freq=self.h_freq,
                    target_sfreq=self.target_sfreq,
                    channel_filter_name="MEG_only"
                )

                print(f"  Cached to {rec['cache_path']}")
            else:
                print(f"Using cached recording {i+1}/{len(self.recordings)}: "
                      f"{rec['subject']} {rec['task']} {rec['run']}")

    def _open_file_handles(self) -> None:
        """
        Open HDF5 file handles for all cached recordings.
        """
        self.file_handles = []
        failed_recordings = []

        for i, rec in enumerate(self.recordings):
            try:
                h5_file = load_cached(rec["cache_path"])
                self.file_handles.append(h5_file)
            except Exception as e:
                print(f"Warning: Failed to open cache file {rec['cache_path']}: {e}")
                failed_recordings.append(i)
                continue

        # Remove failed recordings from the list
        for idx in reversed(failed_recordings):
            self.recordings.pop(idx)

    def _build_segment_index(self) -> List[Tuple[int, int]]:
        """
        Build an index mapping global segment index to (recording_idx, segment_idx).

        Returns
        -------
        segment_index : List[Tuple[int, int]]
            List of (recording_idx, segment_idx_within_recording) tuples
        """
        segment_index = []

        for rec_idx, h5_file in enumerate(self.file_handles):
            n_samples = h5_file.attrs["n_samples"]
            sfreq = h5_file.attrs["sample_freq"]

            # Calculate number of samples per segment
            samples_per_segment = int(self.segment_length * sfreq)

            # Calculate number of complete segments (skip partial segments)
            n_segments = n_samples // samples_per_segment

            # Add all segments from this recording
            for seg_idx in range(n_segments):
                segment_index.append((rec_idx, seg_idx))

        return segment_index

    def __len__(self) -> int:
        """Return total number of segments across all recordings."""
        return len(self.segment_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single segment.

        Parameters
        ----------
        idx : int
            Global segment index

        Returns
        -------
        sample : Dict[str, Any]
            Dictionary containing:
            - meg: torch.Tensor of shape (n_channels, n_timepoints)
            - subject: str
            - session: str (contains run information)
            - task: str
            - sensor_xyzdir: torch.Tensor of shape (n_channels, 6)
            - sensor_types: torch.Tensor of shape (n_channels,)
            - start_time: float (seconds)
            - end_time: float (seconds)
            - recording_idx: int
            - segment_idx: int
            - sensor_mask: torch.Tensor of shape (n_channels,)
        """
        # Get recording and segment indices
        rec_idx, seg_idx = self.segment_index[idx]

        # Get HDF5 file handle
        h5_file = self.file_handles[rec_idx]

        # Get recording metadata
        rec = self.recordings[rec_idx]

        # Get data properties
        sfreq = h5_file.attrs["sample_freq"]
        samples_per_segment = int(self.segment_length * sfreq)

        # Calculate sample range
        start_sample = seg_idx * samples_per_segment
        end_sample = start_sample + samples_per_segment

        # Load segment data
        meg_data = h5_file["data"][:, start_sample:end_sample]

        # Load sensor types (needed for separate scaling)
        sensor_types = h5_file["sensor_types"][:]

        # Preprocess with sub-segmentation
        meg_data = preprocess_segment_with_subsegments(
            meg_data=meg_data,
            sensor_types=sensor_types,
            sfreq=sfreq,
            subsegment_duration=3.0,
            baseline_duration=0.5,
            clip_range=(-5, 5)
        )

        # Optionally shuffle temporal segments (for ablation experiments)
        if self.shuffle_segments:
            meg_data = shuffle_temporal_segments(
                meg_data, self.shuffle_segment_duration, sfreq
            )

        # Load sensor positions (same for all segments in this recording)
        sensor_xyzdir = h5_file["sensor_xyzdir"][:]

        # Calculate timing
        start_time = start_sample / sfreq
        end_time = end_sample / sfreq

        sensor_xyzdir = norm_sensor_positions(sensor_xyzdir)

        # Pad channel dimension and sensor positions if needed
        if self.max_channel_dim is not None:
            n_channels = meg_data.shape[0]
            meg_data = np.pad(meg_data, ((0, self.max_channel_dim - n_channels), (0, 0)))
            sensor_xyzdir = np.pad(sensor_xyzdir, ((0, self.max_channel_dim - sensor_xyzdir.shape[0]), (0, 0)))
            sensor_types = np.pad(sensor_types, (0, self.max_channel_dim - sensor_types.shape[0]))
            sensor_mask = np.zeros(self.max_channel_dim, dtype=np.float32)
            sensor_mask[:n_channels] = 1.0
        else:
            sensor_mask = np.ones(meg_data.shape[0], dtype=np.float32)

        # Convert to torch tensors
        meg_tensor = torch.from_numpy(meg_data).float()
        sensor_xyzdir_tensor = torch.from_numpy(sensor_xyzdir).float()
        sensor_mask_tensor = torch.from_numpy(sensor_mask).float()
        sensor_types_tensor = torch.from_numpy(sensor_types).int()

        return {
            "meg": meg_tensor,
            "subject": h5_file.attrs["subject"],
            "session": h5_file.attrs["session"],  # Contains run info
            "task": h5_file.attrs["task"],
            "sensor_xyzdir": sensor_xyzdir_tensor,
            "sensor_types": sensor_types_tensor,
            "start_time": float(start_time),
            "end_time": float(end_time),
            "recording_idx": rec_idx,
            "segment_idx": seg_idx,
            "sensor_mask": sensor_mask_tensor
        }

    def __del__(self):
        """Close all file handles when the dataset is destroyed."""
        self.close()

    def close(self):
        """Explicitly close all HDF5 file handles."""
        if hasattr(self, 'file_handles'):
            for h5_file in self.file_handles:
                try:
                    h5_file.close()
                except:
                    pass
            self.file_handles = []


if __name__ == "__main__":
    # Test dataset creation
    dataset = SMN4LangMEGDataset(
        data_root="/path/to/ds004078",
        segment_length=30.0,
        subjects=["sub-01"],
        runs=["run-1"],
        tasks=["RDR"],
        cache_dir="./data/cache_smn4lang",
        l_freq=0.1,
        h_freq=40.0,
        target_sfreq=50.0,
    )

    print(f"\n=== Dataset Info ===")
    print(f"Total recordings: {len(dataset.recordings)}")
    print(f"Total segments: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n=== Sample Info ===")
        print(f"MEG shape: {sample['meg'].shape}")
        print(f"Subject: {sample['subject']}")
        print(f"Session (run): {sample['session']}")
        print(f"Task: {sample['task']}")
        print(f"Sensor xyzdir shape: {sample['sensor_xyzdir'].shape}")
        print(f"Sensor types shape: {sample['sensor_types'].shape}")
        print(f"Sensor mask shape: {sample['sensor_mask'].shape}")
        print(f"Start time: {sample['start_time']:.2f}s")
        print(f"End time: {sample['end_time']:.2f}s")

        print("\n=== Test Passed ===")
        breakpoint()
    else:
        print("Dataset is empty.")
