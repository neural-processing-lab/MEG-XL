"""PyTorch Dataset for the LibriBrain MEG dataset."""

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings
from .utils import norm_sensor_positions

from .preprocessing import (
    load_libribrain_sensors,
    preprocess_libribrain_h5,
    cache_preprocessed,
    load_cached,
    get_cache_path,
    preprocess_segment_with_subsegments,
    compute_preproc_hash,
    shuffle_temporal_segments
)


class LibriBrainMEGDataset(Dataset):
    """
    PyTorch Dataset for the LibriBrain MEG dataset.

    LibriBrain is unique among MEG datasets because:
    - Data is already pre-serialized in h5 format (not BIDS raw format)
    - Directory structure is task-first (Sherlock1/, Sherlock2/, ...)
    - Sensor information is shared across all recordings in a single JSON file
    - Data is pre-preprocessed at 250Hz with [0.1, 125]Hz bandpass

    This dataset handles:
    - Discovery of h5 recordings from the task-first directory structure
    - Conditional re-preprocessing when requested parameters differ from existing
    - Efficient loading using persistent HDF5 file handles
    - Segmentation of continuous recordings into fixed-length windows

    Parameters
    ----------
    data_root : str
        Root directory of the LibriBrain dataset (e.g., "/path/to/LibriBrain")
    segment_length : float
        Length of each segment in seconds
    cache_dir : str, optional
        Directory for storing re-preprocessed cache files (default: "./data/cache")
    subjects : List[str], optional
        List of subjects to include (e.g., ["sub-0"]). If None, use all.
    sessions : List[str], optional
        List of sessions to include (e.g., ["ses-1", "ses-2"]). If None, use all.
    tasks : List[str], optional
        List of tasks to include (e.g., ["Sherlock1", "Sherlock3"]). If None, use all.
    l_freq : float
        Low frequency cutoff for band-pass filter (default: 0.1 Hz)
    h_freq : float
        High frequency cutoff for band-pass filter (default: 40.0 Hz)
    target_sfreq : float
        Target sampling frequency after resampling (default: 50.0 Hz)
    channel_filter : Callable[[str], bool]
        Filter function for channels. Channels for which this function returns True will be kept.
    max_channel_dim : int, optional
        If specified, pad channel dimension to this size for batch consistency

    Example
    -------
    >>> dataset = LibriBrainMEGDataset(
    ...     data_root="/path/to/LibriBrain",
    ...     segment_length=30.0,
    ...     tasks=["Sherlock1", "Sherlock2"]
    ... )
    >>> sample = dataset[0]
    >>> print(sample['meg'].shape)  # (n_channels, n_timepoints)
    """

    def __init__(
        self,
        data_root: str,
        segment_length: float,
        cache_dir: str = "./data/cache",
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
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
        self.sessions = sessions
        self.tasks = tasks

        # Load sensor information from JSON (shared across all recordings)
        self._load_sensor_info()

        # Discover all recordings
        self.recordings = self._discover_recordings()

        if len(self.recordings) == 0:
            raise ValueError(
                f"No recordings found in {self.data_root} with the specified filters. "
                f"Subjects: {subjects}, Sessions: {sessions}, Tasks: {tasks}"
            )

        # Preprocess and cache recordings if needed
        self._preprocess_all()

        # Open file handles for all recordings (original or cached)
        self.file_handles: List[h5py.File] = []
        self._open_file_handles()

        # Build segment index: maps global index -> (recording_idx, segment_idx)
        self.segment_index = self._build_segment_index()

    def _load_sensor_info(self) -> None:
        """
        Load sensor information from the shared JSON file.

        LibriBrain stores all sensor information in a single JSON file at the
        dataset root, rather than embedding it in each recording.
        """
        sensor_json_path = self.data_root / "meg_sensors_information.json"

        if not sensor_json_path.exists():
            raise FileNotFoundError(
                f"Sensor information file not found: {sensor_json_path}. "
                "LibriBrain requires meg_sensors_information.json at the dataset root."
            )

        self.sensor_xyzdir_dict, self.sensor_types_dict = load_libribrain_sensors(
            str(sensor_json_path)
        )

        print(f"Loaded sensor info for {len(self.sensor_xyzdir_dict)} channels")

    def _get_available_tasks(self) -> List[str]:
        """
        Scan the serialized directory for available task directories.

        Returns
        -------
        tasks : List[str]
            List of task names (e.g., ["Sherlock1", "Sherlock2", ...])
        """
        serialized_dir = self.data_root / "serialized"
        if not serialized_dir.exists():
            return []

        tasks = []
        for task_dir in sorted(serialized_dir.iterdir()):
            if task_dir.is_dir() and not task_dir.name.startswith('.'):
                # Check if it has the expected structure
                derivatives_dir = task_dir / "derivatives" / "serialised"
                if derivatives_dir.exists():
                    tasks.append(task_dir.name)

        return tasks

    def _parse_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse LibriBrain h5 filename to extract metadata.

        Parameters
        ----------
        filename : str
            Example: "sub-0_ses-9_task-Sherlock1_run-1_proc-bads+headpos+sss+notch+bp+ds_meg.h5"

        Returns
        -------
        metadata : Dict[str, str]
            Dictionary with keys: subject, session, run
        """
        # Remove .h5 extension and split by underscore
        parts = filename.replace('.h5', '').split('_')
        metadata = {}

        for part in parts:
            if part.startswith('sub-'):
                metadata['subject'] = part
            elif part.startswith('ses-'):
                metadata['session'] = part
            elif part.startswith('run-'):
                metadata['run'] = part

        return metadata

    def _discover_recordings(self) -> List[Dict[str, Any]]:
        """
        Discover all h5 recordings matching the specified filters.

        LibriBrain uses a task-first directory structure unlike other datasets.

        Returns
        -------
        recordings : List[Dict[str, Any]]
            List of recording metadata dictionaries
        """
        recordings = []

        # Get tasks to iterate through
        if self.tasks is not None:
            tasks_to_check = self.tasks
        else:
            tasks_to_check = self._get_available_tasks()

        if len(tasks_to_check) == 0:
            warnings.warn(f"No tasks found in {self.data_root}/serialized/")
            return recordings

        # Iterate through task directories
        for task in tasks_to_check:
            task_dir = self.data_root / "serialized" / task / "derivatives" / "serialised"

            if not task_dir.exists():
                warnings.warn(f"Task directory not found: {task_dir}")
                continue

            # Find all h5 files for this task
            h5_files = sorted(task_dir.glob(f"*_task-{task}_*.h5"))

            for h5_file in h5_files:
                # Parse filename to extract metadata
                metadata = self._parse_filename(h5_file.name)

                # Apply subject filter
                if self.subjects is not None and metadata.get('subject') not in self.subjects:
                    continue

                # Apply session filter
                if self.sessions is not None and metadata.get('session') not in self.sessions:
                    continue

                # Generate cache path for re-preprocessed data
                cache_path = self.cache_dir / (
                    f"{metadata.get('subject', 'sub-0')}_{metadata.get('session', 'ses-unknown')}_"
                    f"task-{task}_{metadata.get('run', 'run-unknown')}_"
                    f"preproc-{compute_preproc_hash(self.l_freq, self.h_freq, self.target_sfreq, 'MEG_only')}.h5"
                )

                recordings.append({
                    "subject": metadata.get('subject', 'sub-0'),
                    "session": metadata.get('session', 'ses-unknown'),
                    "task": task,
                    "run": metadata.get('run', 'run-unknown'),
                    "raw_path": h5_file,  # Original LibriBrain h5 file
                    "cache_path": cache_path,  # Re-preprocessed cache (if needed)
                    "use_original": False,  # Will be set in _preprocess_all()
                })

        return recordings

    def _needs_reprocessing(self, h5_file: h5py.File) -> bool:
        """
        Check if re-preprocessing is needed based on requested vs existing parameters.

        LibriBrain data is pre-processed at 250Hz with [0.1, 125]Hz bandpass.
        Re-preprocessing is needed if:
        - Requested target_sfreq differs from existing sample_frequency
        - Requested l_freq > existing highpass_cutoff (need stricter highpass)
        - Requested h_freq < existing lowpass_cutoff (need stricter lowpass)

        Parameters
        ----------
        h5_file : h5py.File
            Open h5 file handle to check attributes

        Returns
        -------
        needs_reprocessing : bool
            True if re-preprocessing is needed
        """
        existing_sfreq = h5_file.attrs.get('sample_frequency', 250.0)
        existing_l_freq = h5_file.attrs.get('highpass_cutoff', 0.1)
        existing_h_freq = h5_file.attrs.get('lowpass_cutoff', 125.0)

        # Check if sampling rate differs
        if abs(self.target_sfreq - existing_sfreq) > 0.1:
            return True

        # Check if we need stricter highpass filter
        if self.l_freq > existing_l_freq + 0.01:
            return True

        # Check if we need stricter lowpass filter
        if self.h_freq < existing_h_freq - 0.1:
            return True

        return False

    def _preprocess_all(self) -> None:
        """
        Preprocess recordings that need re-processing.

        This method decides for each recording whether to:
        1. Use the original h5 file (if parameters match existing preprocessing)
        2. Create a re-preprocessed cache (if parameters differ)
        """
        for i, rec in enumerate(self.recordings):
            # Check if re-preprocessing is needed
            with h5py.File(rec["raw_path"], 'r') as h5_orig:
                needs_reprocess = self._needs_reprocessing(h5_orig)

            if not needs_reprocess:
                # Can use original h5 file
                rec["use_original"] = True
                print(f"Using original h5 file {i+1}/{len(self.recordings)}: "
                      f"{rec['subject']} {rec['session']} {rec['task']} {rec['run']}")
            else:
                # Need to re-preprocess
                rec["use_original"] = False

                if not rec["cache_path"].exists():
                    print(f"Re-preprocessing recording {i+1}/{len(self.recordings)}: "
                          f"{rec['subject']} {rec['session']} {rec['task']} {rec['run']}")

                    # Re-preprocess using the h5 loader
                    raw = preprocess_libribrain_h5(
                        str(rec["raw_path"]),
                        self.sensor_xyzdir_dict,
                        self.sensor_types_dict,
                        l_freq=self.l_freq,
                        h_freq=self.h_freq,
                        target_sfreq=self.target_sfreq,
                        channel_filter=self.channel_filter
                    )

                    # Cache the re-preprocessed data
                    metadata = {
                        "subject": rec["subject"],
                        "session": rec["session"],
                        "task": rec["task"],
                        "run": rec["run"],
                        "dataset": "libribrain"
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
                    print(f"Using cached re-preprocessed recording {i+1}/{len(self.recordings)}: "
                          f"{rec['subject']} {rec['session']} {rec['task']} {rec['run']}")
                    
                    # Map channel names to sensor info
                    ch_names_bytes = h5py.File(rec["cache_path"], 'r')['channel_names'][:]
                    # Bytes to strings
                    ch_names = [name.decode('utf-8') for name in ch_names_bytes]
                    # Filter channels according to channel_filter
                    filtered_ch_names = [ch for ch in ch_names if self.channel_filter(ch)]
                    sensor_xyzdir_list = []
                    sensor_types_list = []
                    for ch_name in filtered_ch_names:
                        if ch_name in self.sensor_xyzdir_dict:
                            sensor_xyzdir_list.append(self.sensor_xyzdir_dict[ch_name])
                            sensor_types_list.append(self.sensor_types_dict[ch_name])
                        else:
                            warnings.warn(f"Channel {ch_name} not found in sensor JSON, skipping")

                    rec['sensor_xyzdir'] = np.array(sensor_xyzdir_list)
                    rec['sensor_types'] = np.array(sensor_types_list)

    def _open_file_handles(self) -> None:
        """
        Open HDF5 file handles for all recordings.

        Opens either the original h5 file or the re-preprocessed cache,
        depending on the use_original flag set during preprocessing.
        """
        self.file_handles = []

        for rec in self.recordings:
            if rec["use_original"]:
                # Open original h5 file
                h5_file = h5py.File(rec["raw_path"], 'r')

                # Build sensor arrays from JSON for this recording's channels
                # (original files don't have sensor info embedded)
                ch_names = h5_file.attrs['channel_names'].split(', ')

                # Filter channels according to channel_filter
                filtered_ch_names = [ch for ch in ch_names if self.channel_filter(ch)]

                # Build sensor arrays
                sensor_xyzdir_list = []
                sensor_types_list = []

                for ch_name in filtered_ch_names:
                    if ch_name in self.sensor_xyzdir_dict:
                        sensor_xyzdir_list.append(self.sensor_xyzdir_dict[ch_name])
                        sensor_types_list.append(self.sensor_types_dict[ch_name])
                    else:
                        warnings.warn(f"Channel {ch_name} not found in sensor JSON, skipping")

                rec['sensor_xyzdir'] = np.array(sensor_xyzdir_list)
                rec['sensor_types'] = np.array(sensor_types_list)
                rec['filtered_channels'] = filtered_ch_names
            else:
                # Open re-preprocessed cache (has sensor info embedded)
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
            rec = self.recordings[rec_idx]

            if rec["use_original"]:
                # Original file - get sample info from attributes
                n_samples = h5_file['data'].shape[1]
                sfreq = h5_file.attrs.get('sample_frequency', 250.0)
            else:
                # Cached file - get sample info from attributes
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

        # Get HDF5 file handle and recording metadata
        h5_file = self.file_handles[rec_idx]
        rec = self.recordings[rec_idx]

        # Get data properties
        if rec["use_original"]:
            sfreq = h5_file.attrs.get('sample_frequency', 250.0)
        else:
            sfreq = h5_file.attrs["sample_freq"]

        samples_per_segment = int(self.segment_length * sfreq)

        # Calculate sample range
        start_sample = seg_idx * samples_per_segment
        end_sample = start_sample + samples_per_segment

        # Load segment data
        if rec["use_original"]:
            # Original file - need to apply channel filter manually
            all_data = h5_file["data"][:, start_sample:end_sample]
            ch_names = h5_file.attrs['channel_names'].split(', ')

            # Get indices of filtered channels
            filtered_indices = [i for i, ch in enumerate(ch_names) if ch in rec['filtered_channels']]
            meg_data = all_data[filtered_indices, :]

            # Get sensor info from recording dict
            sensor_xyzdir = rec['sensor_xyzdir']
            sensor_types = rec['sensor_types']
        else:
            # Cached file - data already filtered
            meg_data = h5_file["data"][:, start_sample:end_sample]
            # sensor_xyzdir = h5_file["sensor_xyzdir"][:]
            # sensor_types = h5_file["sensor_types"][:]
            # Get sensor info from recording dict
            sensor_xyzdir = rec['sensor_xyzdir']
            sensor_types = rec['sensor_types']

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

        # Calculate timing
        start_time = start_sample / sfreq
        end_time = end_sample / sfreq

        # Normalize sensor positions
        sensor_xyzdir = norm_sensor_positions(sensor_xyzdir)

        # Pad channel dimension and sensor positions if needed
        if self.max_channel_dim is not None:
            n_channels = meg_data.shape[0]
            meg_data = np.pad(meg_data, ((0, self.max_channel_dim - n_channels), (0, 0)))
            sensor_xyzdir = np.pad(sensor_xyzdir, ((0, self.max_channel_dim - n_channels), (0, 0)))
            sensor_types = np.pad(sensor_types, (0, self.max_channel_dim - n_channels))
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
            "subject": rec["subject"],
            "session": rec["session"],
            "task": rec["task"],
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
    # Example usage
    dataset = LibriBrainMEGDataset(
        data_root="/path/to/LibriBrain",
        segment_length=30.0,
        tasks=["Sherlock1"],
        l_freq=0.1,
        h_freq=125.0,
        target_sfreq=250.0,
    )
    print(f"Dataset size: {len(dataset)} segments")

    sample = dataset[0]
    print(f"MEG data shape: {sample['meg'].shape}")
    print(f"Sensor positions shape: {sample['sensor_xyzdir'].shape}")
    print(f"Subject: {sample['subject']}, Session: {sample['session']}, Task: {sample['task']}")
    breakpoint()
