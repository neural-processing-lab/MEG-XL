"""PyTorch Dataset for the CamCAN (Shafto 2014) MEG dataset."""

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings
import mne
from .utils import norm_sensor_positions

from .preprocessing import (
    cache_preprocessed,
    load_cached,
    get_cache_path,
    preprocess_segment_with_subsegments,
    shuffle_temporal_segments
)

def preprocess_camcan(
    raw_path: str,
    l_freq: float = 0.1,
    h_freq: float = 40.0,
    target_sfreq: float = 50.0,
    channel_filter: Callable[[str], bool] = lambda _: True
) -> mne.io.Raw:
    """
    Preprocess a single MEG recording.

    Pipeline:
    1. Load raw data
    2. Band-pass filter [l_freq, h_freq] Hz
    3. Resample to target_sfreq Hz
    4. Keep only channels where channel_filter returns True
    5. Apply robust scaling (median=0, Q1=-1, Q3=1) per channel

    Parameters
    ----------
    raw_path : str
        Path to the raw MEG file (.ds directory for CTF format)
    l_freq : float
        Low frequency cutoff for band-pass filter (default: 0.1 Hz)
    h_freq : float
        High frequency cutoff for band-pass filter (default: 40.0 Hz)
    target_sfreq : float
        Target sampling frequency after resampling (default: 50.0 Hz)
    channel_filter : Callable[[str], bool]
        Filter function for channels

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

class CamCANMEGDataset(Dataset):
    """
    PyTorch Dataset for the CamCAN (Shafto 2014) MEG dataset.

    This dataset handles:
    - Discovery of MEG recordings from the CamCAN BIDSsep directory structure
    - Support for 'rest' and 'smt' tasks
    - Lazy preprocessing with caching
    - Segmentation of continuous recordings into fixed-length windows

    Directory Structure Expected:
    root/
      rest/
        sub-CC123/
          ses-rest/
            meg/
              ...task-rest_meg.fif
      smt/
        sub-CC123/
          ses-smt/
             meg/
               ...task-smt_meg.fif

    Parameters
    ----------
    data_root : str
        Root directory containing the 'rest' and 'smt' folders
        (e.g., ".../shafto2014/cc700/meg/pipeline/release005/BIDSsep")
    segment_length : float
        Length of each segment in seconds
    cache_dir : str, optional
        Directory for storing preprocessed cache files (default: "./data/cache_camcan")
    subjects : List[str], optional
        List of subjects to include (e.g., ["sub-CC110033"]). If None, use all.
    sessions : List[str], optional
        List of sessions to include (e.g., ["ses-rest"]). If None, use all.
    tasks : List[str], optional
        List of tasks to include (e.g., ["rest", "smt"]). If None, use both.
    l_freq : float
        Low frequency cutoff for band-pass filter (default: 0.1 Hz)
    h_freq : float
        High frequency cutoff for band-pass filter (default: 40.0 Hz)
    target_sfreq : float
        Target sampling frequency after resampling (default: 50.0 Hz)
    channel_filter : Callable[[str], bool]
        Filter function for channels. (Default: filters for channels starting with 'MEG')
    max_channel_dim : int, optional
        If set, pads the channel dimension to this size.
    """

    def __init__(
        self,
        data_root: str,
        segment_length: float,
        cache_dir: str = "./data/cache_camcan",
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,  # Default to ["rest", "smt"]
        l_freq: float = 0.1,
        h_freq: float = 40.0,
        target_sfreq: float = 50.0,
        channel_filter: Callable[[str], bool] = lambda _: True,
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
        self.sessions = sessions
        # If no tasks specified, default to discovering both supported tasks
        self.tasks = tasks if tasks is not None else ["rest", "smt"]

        # Discover all recordings
        self.recordings = self._discover_recordings()

        if len(self.recordings) == 0:
            raise ValueError(
                f"No recordings found in {self.data_root} with the specified filters. "
                f"Subjects: {subjects}, Sessions: {sessions}, Tasks: {tasks}"
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
        Adapts to the BIDSsep structure where top-level folders are tasks.

        Returns
        -------
        recordings : List[Dict[str, Any]]
            List of recording metadata dictionaries
        """
        recordings = []

        # Iterate through the requested tasks first (since they are top level folders)
        for task_name in self.tasks:
            task_dir = self.data_root / task_name

            if not task_dir.exists():
                warnings.warn(f"Task directory not found: {task_dir}")
                continue

            # Iterate through subject directories inside the task folder
            subject_dirs = sorted(task_dir.glob("sub-*"))

            for subject_dir in subject_dirs:
                subject = subject_dir.name

                # Apply subject filter
                if self.subjects is not None and subject not in self.subjects:
                    continue

                # Iterate through session directories
                # Note: In this dataset, sessions usually match task (e.g., ses-rest)
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
                    # Pattern: sub-CCXXX_ses-XXX_task-XXX_meg.fif
                    fif_pattern = f"*_task-{task_name}_meg.fif"
                    meg_files = sorted(meg_dir.glob(fif_pattern))

                    for meg_file in meg_files:
                        recordings.append({
                            "subject": subject,
                            "session": session,
                            "task": task_name,
                            "raw_path": meg_file,
                            "cache_path": get_cache_path(
                                self.cache_dir, subject, session, task_name,
                                l_freq=self.l_freq,
                                h_freq=self.h_freq,
                                target_sfreq=self.target_sfreq,
                                channel_filter_name="MEG_only"
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
                      f"{rec['subject']} {rec['session']} {rec['task']}")

                try:
                    # Preprocess
                    # Note: CamCAN is Neuromag (.fif). MNE handles this automatically
                    # based on extension.
                    raw = preprocess_camcan(
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
                        "dataset": "camcan"
                    }
                    cache_preprocessed(
                        raw, rec["cache_path"], metadata,
                        l_freq=self.l_freq,
                        h_freq=self.h_freq,
                        target_sfreq=self.target_sfreq,
                        channel_filter_name="MEG_only"
                    )

                    print(f"  Cached to {rec['cache_path']}")

                except Exception as e:
                    print(f"  Failed to preprocess {rec['raw_path']}: {e}")
                    # We might want to remove this recording from self.recordings
                    # or just skip it. For now, we print error.
            else:
                print(f"Using cached recording {i+1}/{len(self.recordings)}: "
                      f"{rec['subject']} {rec['session']} {rec['task']}")

    def _open_file_handles(self) -> None:
        """
        Open HDF5 file handles for all cached recordings.
        """
        self.file_handles = []
        valid_recordings = []
        for rec in self.recordings:
            if rec["cache_path"].exists():
                try:
                    h5_file = load_cached(rec["cache_path"])
                    self.file_handles.append(h5_file)
                    valid_recordings.append(rec)
                except Exception as e:
                    print(f"Error loading cache {rec['cache_path']}: {e}")
            else:
                print(f"Cache missing for {rec['subject']}, skipping.")
        
        # Update recordings list to only include those that successfully loaded
        self.recordings = valid_recordings

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

            # Calculate number of complete segments
            if samples_per_segment > 0:
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
            Dictionary containing processed MEG data and metadata.
        """
        # Get recording and segment indices
        rec_idx, seg_idx = self.segment_index[idx]

        # Get HDF5 file handle
        h5_file = self.file_handles[rec_idx]

        # Get data properties
        sfreq = h5_file.attrs["sample_freq"]
        samples_per_segment = int(self.segment_length * sfreq)

        # Calculate sample range
        start_sample = seg_idx * samples_per_segment
        end_sample = start_sample + samples_per_segment

        # Load segment data
        meg_data = h5_file["data"][:, start_sample:end_sample]

        # Load sensor types
        sensor_types = h5_file["sensor_types"][:]

        # Preprocess with sub-segmentation
        # Note: CamCAN has planar grads and magnetometers, so scaling in this
        # function is important (assuming it handles sensor_types correctly).
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

        # Load sensor positions
        sensor_xyzdir = h5_file["sensor_xyzdir"][:]

        # Calculate timing
        start_time = start_sample / sfreq
        end_time = end_sample / sfreq

        sensor_xyzdir = norm_sensor_positions(sensor_xyzdir)

        # Pad channel dimension and sensor positions if needed
        if self.max_channel_dim is not None:
            curr_channels = meg_data.shape[0]
            if curr_channels < self.max_channel_dim:
                pad_width = self.max_channel_dim - curr_channels
                meg_data = np.pad(meg_data, ((0, pad_width), (0, 0)))
                sensor_xyzdir = np.pad(sensor_xyzdir, ((0, pad_width), (0, 0)))
                sensor_types = np.pad(sensor_types, (0, pad_width))
                sensor_mask = np.zeros(self.max_channel_dim, dtype=np.float32)
                sensor_mask[:curr_channels] = 1.0
            else:
                # If we have more channels than max (unlikely if max is high enough), slice
                meg_data = meg_data[:self.max_channel_dim]
                sensor_xyzdir = sensor_xyzdir[:self.max_channel_dim]
                sensor_types = sensor_types[:self.max_channel_dim]
                sensor_mask = np.ones(self.max_channel_dim, dtype=np.float32)
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
        for h5_file in self.file_handles:
            try:
                h5_file.close()
            except:
                pass
        self.file_handles = []

if __name__ == "__main__":
    # Example usage based on your paths
    dataset = CamCANMEGDataset(
        data_root="/path/to/shafto2014/cc700/meg/pipeline/release005/BIDSsep",
        segment_length=30.0,
        subjects=["sub-CC120065"], # Example from your logs
        tasks=["rest"],
        cache_dir="./data/cache_camcan",
        l_freq=0.1,
        h_freq=125.0,
        target_sfreq=250.0,
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Loaded sample with shape: {sample['meg'].shape}")
        print(f"Task: {sample['task']}")
        breakpoint()
    else:
        print("Dataset is empty.")