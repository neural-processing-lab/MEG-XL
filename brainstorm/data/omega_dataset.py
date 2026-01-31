"""PyTorch Dataset for the Omega MEG dataset."""

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings
from .utils import norm_sensor_positions

from .preprocessing import (
    preprocess_recording,
    cache_preprocessed,
    load_cached,
    get_cache_path,
    preprocess_segment_with_subsegments
)


class OmegaMEGDataset(Dataset):
    """
    PyTorch Dataset for the Omega MEG dataset.

    This dataset handles:
    - Discovery of MEG recordings from the Omega dataset
    - Lazy preprocessing with caching (band-pass filter, resample, channel selection)
    - Segmentation of continuous recordings into fixed-length windows
    - Efficient loading using persistent HDF5 file handles
    - Multiple runs per session with optional run filtering

    The dataset automatically excludes "noise" (empty room) recordings.

    Parameters
    ----------
    data_root : str
        Root directory of the Omega dataset (e.g., "/path/to/Omega")
    segment_length : float
        Length of each segment in seconds
    cache_dir : str, optional
        Directory for storing preprocessed cache files (default: "./data/cache")
    subjects : List[str], optional
        List of subjects to include (e.g., ["sub-MNI0452", "sub-PD0757"]). If None, use all.
    sessions : List[str], optional
        List of sessions to include (e.g., ["ses-01", "ses-02"]). If None, use all.
    tasks : List[str], optional
        List of tasks to include (e.g., ["rest"]). If None, use all non-noise tasks.
        Note: "noise" tasks are always excluded regardless of this parameter.
    runs : List[str], optional
        List of runs to include (e.g., ["run-01", "run-02"]). If None, use all runs.
    l_freq : float
        Low frequency cutoff for band-pass filter (default: 0.1 Hz)
    h_freq : float
        High frequency cutoff for band-pass filter (default: 40.0 Hz)
    target_sfreq : float
        Target sampling frequency after resampling (default: 50.0 Hz)
    channel_filter : Callable[[str], bool]
        Filter function for channels. Channels for which this function returns True will be kept.
    max_channel_dim : int, optional
        If specified, pad channel dimension to this size for multi-dataset compatibility

    Example
    -------
    >>> dataset = OmegaMEGDataset(
    ...     data_root="/path/to/Omega",
    ...     segment_length=10.0,
    ...     subjects=["sub-MNI0452"],
    ...     sessions=["ses-02"],
    ...     runs=["run-02", "run-03"]
    ... )
    >>> sample = dataset[0]
    >>> print(sample['meg'].shape)  # (n_channels, n_timepoints)
    >>> print(sample['run'])  # "run-02"
    """

    def __init__(
        self,
        data_root: str,
        segment_length: float,
        cache_dir: str = "./data/cache",
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        runs: Optional[List[str]] = None,
        l_freq: float = 0.1,
        h_freq: float = 40.0,
        target_sfreq: float = 50.0,
        channel_filter: Callable[[str], bool] = lambda x : x.startswith('M'),  # A function that filters channels
        max_channel_dim: Optional[int] = None
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

        # Filters
        self.subjects = subjects
        self.sessions = sessions
        self.tasks = tasks
        self.runs = runs

        # Discover all recordings
        self.recordings = self._discover_recordings()

        if len(self.recordings) == 0:
            raise ValueError(
                f"No recordings found in {self.data_root} with the specified filters. "
                f"Subjects: {subjects}, Sessions: {sessions}, Tasks: {tasks}, Runs: {runs}"
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

            # Iterate through session directories
            session_dirs = sorted(subject_dir.glob("ses-*"))

            for session_dir in session_dirs:
                session = session_dir.name

                # Apply session filter
                if self.sessions is not None and session not in self.sessions:
                    continue

                # Look for MEG data
                meg_dir = session_dir / "meg"
                if not meg_dir.exists():
                    continue

                # Find task MEG files
                meg_files = sorted(meg_dir.glob("*_task-*_meg.ds"))

                for meg_file in meg_files:
                    # Extract task and run from filename
                    # Format: sub-{ID}_ses-{N}_task-{TASK}_run-{N}_meg.ds
                    # Example: sub-MNI0452_ses-02_task-rest_run-03_meg.ds
                    parts = meg_file.name.split("_")
                    task = None
                    run = None
                    for part in parts:
                        if part.startswith("task-"):
                            task = part.replace("task-", "")
                        elif part.startswith("run-"):
                            run = part  # Keep full "run-02" format for filtering

                    if task is None or run is None:
                        continue

                    # ALWAYS skip noise (empty room) recordings - hardcoded exclusion
                    if task == "noise":
                        continue

                    # Apply task filter if specified
                    if self.tasks is not None and task not in self.tasks:
                        continue

                    # Apply run filter if specified
                    if self.runs is not None and run not in self.runs:
                        continue

                    recordings.append({
                        "subject": subject,
                        "session": session,
                        "task": task,
                        "run": run,
                        "raw_path": meg_file,
                        "cache_path": get_cache_path(
                            self.cache_dir, subject, session, f"{task}_{run}",
                            l_freq=self.l_freq,
                            h_freq=self.h_freq,
                            target_sfreq=self.target_sfreq,
                            channel_filter_name="MEG_only"  # Can be made configurable if needed
                        )
                    })

        return recordings

    def _preprocess_all(self) -> None:
        """
        Preprocess all recordings that haven't been cached yet.
        """
        for i, rec in enumerate(self.recordings):
            if not rec["cache_path"].exists():
                print(f"Preprocessing recording {i+1}/{len(self.recordings)}: "
                      f"{rec['subject']} {rec['session']} {rec['task']} {rec['run']}")

                # Preprocess
                raw = preprocess_recording(
                    str(rec["raw_path"]),
                    l_freq=self.l_freq,
                    h_freq=self.h_freq,
                    target_sfreq=self.target_sfreq,
                    channel_filter=self.channel_filter
                )

                # Cache
                metadata = {
                    "subject": rec["subject"],
                    "session": rec["session"],
                    "task": rec["task"],
                    "run": rec["run"],
                    "dataset": "omega"
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
                      f"{rec['subject']} {rec['session']} {rec['task']} {rec['run']}")

    def _open_file_handles(self) -> None:
        """
        Open HDF5 file handles for all cached recordings.
        """
        self.file_handles = []
        for rec in self.recordings:
            h5_file = load_cached(rec["cache_path"])
            self.file_handles.append(h5_file)

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
            - session: str
            - task: str
            - run: str
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

        # Load sensor positions (same for all segments in this recording)
        sensor_xyzdir = h5_file["sensor_xyzdir"][:]

        # Calculate timing
        start_time = start_sample / sfreq
        end_time = end_sample / sfreq

        sensor_xyzdir = norm_sensor_positions(sensor_xyzdir)

        # Pad channel dimension and sensor positions if needed
        if self.max_channel_dim is not None:
            n_channels_original = meg_data.shape[0]
            meg_data = np.pad(meg_data, ((0, self.max_channel_dim - n_channels_original), (0, 0)))
            sensor_xyzdir = np.pad(sensor_xyzdir, ((0, self.max_channel_dim - sensor_xyzdir.shape[0]), (0, 0)))
            sensor_types = np.pad(sensor_types, (0, self.max_channel_dim - sensor_types.shape[0]))
            sensor_mask = np.zeros(self.max_channel_dim, dtype=np.float32)
            sensor_mask[:n_channels_original] = 1.0
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
            "session": h5_file.attrs["session"],
            "task": h5_file.attrs["task"],
            "run": h5_file.attrs["run"],
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
    import matplotlib.pyplot as plt
    from brainstorm.neuro_tokenizers.biocodec.model import BioCodecModel

    # Create dataset
    dataset = OmegaMEGDataset(
        data_root="/path/to/Omega",
        segment_length=150.0,
        subjects=["sub-MNI0452"],
        sessions=["ses-02"],
        runs=["run-02"]
    )
    print(f"Dataset length: {len(dataset)} segments")
    print(f"Number of recordings: {len(dataset.recordings)}")

    sample = dataset[0]
    print(f"\nFirst sample:")
    print(f"  MEG shape: {sample['meg'].shape}")
    print(f"  Subject: {sample['subject']}")
    print(f"  Session: {sample['session']}")
    print(f"  Task: {sample['task']}")
    print(f"  Run: {sample['run']}")
    print(f"  Time range: {sample['start_time']:.1f}s - {sample['end_time']:.1f}s")

    breakpoint()