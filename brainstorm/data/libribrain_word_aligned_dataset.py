"""PyTorch Dataset for word-aligned segments from the LibriBrain MEG dataset."""

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings
from .utils import norm_sensor_positions

from .preprocessing import (
    load_libribrain_sensors,
    preprocess_libribrain_h5,
    cache_preprocessed,
    load_cached,
    compute_preproc_hash,
    _process_single_chunk
)


class LibriBrainWordAlignedDataset(Dataset):
    """
    PyTorch Dataset for word-aligned 30s segments from LibriBrain MEG dataset.

    Each segment contains 10 consecutive words, where each word has a 3s
    window aligned to its onset. The 10 windows are concatenated to form
    a 30s segment. Each 3s subsegment is independently preprocessed with
    baseline correction, robust scaling, and clipping.

    Parameters
    ----------
    data_root : str
        Root directory of the LibriBrain dataset (e.g., "/path/to/LibriBrain")
    segment_length : float
        Total segment length in seconds (should equal words_per_segment x subsegment_duration)
        Default: 30.0
    subsegment_duration : float
        Duration of each word window in seconds. Default: 3.0
    words_per_segment : int
        Number of consecutive words per segment. Default: 10
    window_onset_offset : float
        Start time of window relative to word onset in seconds.
        Default: -0.5 (starts 0.5s before word onset)
    cache_dir : str, optional
        Directory for storing preprocessed cache files (default: "./data/cache")
    subjects : List[str], optional
        List of subjects to include (e.g., ["sub-0"]). If None, use all.
    sessions : List[str], optional
        List of sessions to include (e.g., ["ses-1", "ses-2"]). If None, use all.
    tasks : List[str], optional
        List of tasks to include (e.g., ["Sherlock1"]). If None, use all.
    l_freq : float
        Low frequency cutoff for band-pass filter (default: 0.1 Hz)
    h_freq : float
        High frequency cutoff for band-pass filter (default: 40.0 Hz)
    target_sfreq : float
        Target sampling frequency after resampling (default: 50.0 Hz)
    channel_filter : Callable[[str], bool]
        Filter function for channels. Channels for which this function returns True will be kept.
        Default: lambda x: x.startswith('MEG') (MEG channels only)
    max_channel_dim : int, optional
        Maximum channel dimension for padding. If specified, MEG data and sensor
        positions will be zero-padded to this dimension (default: None, no padding)
    baseline_duration : float
        Duration of baseline window for correction in seconds (default: 0.5)
    clip_range : tuple
        Min and max values for clipping after scaling (default: (-5, 5))

    Returns (from __getitem__)
    -------
    Dictionary containing:
        - meg: torch.Tensor of shape (n_channels, n_timepoints)
        - words: List[str] of length words_per_segment
        - subsegment_boundaries: List[Dict] with 'start_sample' and 'end_sample' keys
        - sensor_xyzdir: torch.Tensor of shape (n_channels, 6)
        - sensor_types: torch.Tensor of shape (n_channels,)
        - sensor_mask: torch.Tensor of shape (n_channels,)
        - subject: str
        - session: str
        - task: str
        - recording_idx: int
        - segment_idx: int
        - start_time: float (seconds)
        - end_time: float (seconds)

    Example
    -------
    >>> dataset = LibriBrainWordAlignedDataset(
    ...     data_root="/path/to/LibriBrain",
    ...     segment_length=30.0,
    ...     subsegment_duration=3.0,
    ...     words_per_segment=10,
    ...     window_onset_offset=-0.5,
    ...     tasks=["Sherlock1"],
    ... )
    >>> print(f"Dataset: {len(dataset)} segments")
    >>> sample = dataset[0]
    >>> print(f"MEG shape: {sample['meg'].shape}")
    >>> print(f"Words: {sample['words']}")
    >>> print(f"Number of subsegments: {len(sample['subsegment_boundaries'])}")
    """

    def __init__(
        self,
        data_root: str,
        segment_length: float = 30.0,
        subsegment_duration: float = 3.0,
        words_per_segment: int = 10,
        window_onset_offset: float = -0.5,
        cache_dir: str = "./data/cache",
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        l_freq: float = 0.1,
        h_freq: float = 40.0,
        target_sfreq: float = 50.0,
        channel_filter: Callable[[str], bool] = lambda x: x.startswith('MEG'),
        max_channel_dim: Optional[int] = None,
        baseline_duration: float = 0.5,
        clip_range: tuple = (-5, 5)
    ):
        self.data_root = Path(data_root)
        self.segment_length = segment_length
        self.subsegment_duration = subsegment_duration
        self.words_per_segment = words_per_segment
        self.window_onset_offset = window_onset_offset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_duration = baseline_duration
        self.clip_range = clip_range

        self.l_freq = l_freq
        self.h_freq = h_freq
        self.target_sfreq = target_sfreq
        self.channel_filter = channel_filter
        self.max_channel_dim = max_channel_dim

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

        # Preprocess and cache all recordings
        self._preprocess_all()

        # Open file handles for all cached recordings
        self.file_handles: List[h5py.File] = []
        self._open_file_handles()

        # Parse events and build word groups
        self.word_groups: List[List[List[Dict]]] = []
        self._parse_all_events()

        # Build segment index: maps global index -> (recording_idx, word_group_idx)
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
                # Check if it has the expected structure with events
                events_dir = task_dir / "derivatives" / "events"
                serialised_dir = task_dir / "derivatives" / "serialised"
                if events_dir.exists() and serialised_dir.exists():
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

    def _construct_events_filename(self, metadata: Dict[str, str], task: str) -> str:
        """
        Construct events filename from metadata.

        Parameters
        ----------
        metadata : Dict[str, str]
            Dictionary with subject, session, run keys
        task : str
            Task name (e.g., "Sherlock1")

        Returns
        -------
        filename : str
            Example: "sub-0_ses-9_task-Sherlock1_run-1_events.tsv"
        """
        return (f"{metadata['subject']}_{metadata['session']}_"
                f"task-{task}_{metadata['run']}_events.tsv")

    def _discover_recordings(self) -> List[Dict[str, Any]]:
        """
        Discover all h5 recordings with matching events files.

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
            h5_dir = self.data_root / "serialized" / task / "derivatives" / "serialised"
            events_dir = self.data_root / "serialized" / task / "derivatives" / "events"

            if not h5_dir.exists():
                warnings.warn(f"H5 directory not found: {h5_dir}")
                continue

            if not events_dir.exists():
                warnings.warn(f"Events directory not found: {events_dir}")
                continue

            # Find all h5 files for this task
            h5_files = sorted(h5_dir.glob(f"*_task-{task}_*.h5"))

            for h5_file in h5_files:
                # Parse filename to extract metadata
                metadata = self._parse_filename(h5_file.name)

                # Apply subject filter
                if self.subjects is not None and metadata.get('subject') not in self.subjects:
                    continue

                # Apply session filter
                if self.sessions is not None and metadata.get('session') not in self.sessions:
                    continue

                # Construct expected events file path
                events_filename = self._construct_events_filename(metadata, task)
                events_file = events_dir / events_filename

                if not events_file.exists():
                    warnings.warn(f"Events file not found for {h5_file.name}: {events_file}, skipping")
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
                    "raw_path": h5_file,
                    "events_path": events_file,
                    "cache_path": cache_path,
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

    def _open_file_handles(self) -> None:
        """
        Open HDF5 file handles for all recordings and build sensor arrays.
        """
        self.file_handles = []

        for rec in self.recordings:
            if rec["use_original"]:
                # Open original h5 file
                h5_file = h5py.File(rec["raw_path"], 'r')

                # Build sensor arrays from JSON for this recording's channels
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
                # Open re-preprocessed cache
                h5_file = load_cached(rec["cache_path"])

                # Build sensor arrays from cache's channel_names + JSON
                ch_names_bytes = h5_file['channel_names'][:]
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
                rec['filtered_channels'] = filtered_ch_names

            self.file_handles.append(h5_file)

    def _parse_events_file(self, events_path: Path) -> pd.DataFrame:
        """
        Parse LibriBrain events.tsv and filter to valid word events.

        LibriBrain events format columns:
        - idx, wavfile, kind, segment, sentenceidx, wordidx, phonemeidx,
          timemeg, timeds, timechapter, timesentence, duration

        Parameters
        ----------
        events_path : Path
            Path to events.tsv file

        Returns
        -------
        events_df : pd.DataFrame
            DataFrame with 'onset' and 'value' columns for word events
        """
        # Load TSV
        events_df = pd.read_csv(events_path, sep='\t')

        # Filter to word events only (exclude silence and phoneme)
        events_df = events_df[events_df['kind'] == 'word'].copy()

        # Rename columns for consistency with Armeni logic
        events_df = events_df.rename(columns={
            'timemeg': 'onset',
            'segment': 'value'
        })

        # Remove invalid entries
        events_df = events_df[events_df['value'].notna()]

        # Sort by onset time
        events_df = events_df.sort_values('onset').reset_index(drop=True)

        return events_df[['onset', 'value']]

    def _build_word_groups(self, events_df: pd.DataFrame, recording_duration: float) -> List[List[Dict]]:
        """
        Group consecutive valid words into segments.

        Parameters
        ----------
        events_df : pd.DataFrame
            DataFrame with 'onset' and 'value' columns
        recording_duration : float
            Total duration of recording in seconds

        Returns
        -------
        word_groups : List[List[Dict]]
            List of word groups, where each group contains words_per_segment word dicts
        """
        word_groups = []
        current_group = []

        for _, row in events_df.iterrows():
            word_value = str(row['value']).strip().lower()

            # Skip silence markers (LibriBrain uses "silence", Armeni uses "sp")
            if word_value == 'silence':
                continue

            word_onset = row['onset']

            # Calculate window boundaries
            window_start = word_onset + self.window_onset_offset
            window_end = window_start + self.subsegment_duration

            # Skip if window extends beyond recording boundaries
            if window_start < 0 or window_end > recording_duration:
                if len(current_group) > 0:
                    current_group = []  # Reset incomplete group
                continue

            # Add word to current group
            current_group.append({
                'word': word_value,
                'onset': word_onset,
                'window_start': window_start,
                'window_end': window_end,
                'subsegment_idx': len(current_group)
            })

            # Save complete group
            if len(current_group) == self.words_per_segment:
                word_groups.append(current_group.copy())
                current_group = []

        return word_groups

    def _parse_all_events(self) -> None:
        """
        Parse events for all recordings and build word groups.
        """
        self.word_groups = []

        for rec_idx, rec in enumerate(self.recordings):
            # Get recording duration from HDF5
            h5_file = self.file_handles[rec_idx]

            if rec["use_original"]:
                n_samples = h5_file['data'].shape[1]
                sfreq = h5_file.attrs.get('sample_frequency', 250.0)
            else:
                n_samples = h5_file.attrs["n_samples"]
                sfreq = h5_file.attrs["sample_freq"]

            recording_duration = n_samples / sfreq

            # Parse events
            events_df = self._parse_events_file(rec["events_path"])

            # Build word groups
            groups = self._build_word_groups(events_df, recording_duration)
            self.word_groups.append(groups)

            print(f"Recording {rec_idx} ({rec['subject']} {rec['session']} {rec['task']}): "
                  f"Found {len(groups)} word-aligned segments")

    def _build_segment_index(self) -> List[Tuple[int, int]]:
        """
        Build an index mapping global segment index to (recording_idx, word_group_idx).

        Returns
        -------
        segment_index : List[Tuple[int, int]]
            List of (recording_idx, word_group_idx) tuples
        """
        segment_index = []

        for rec_idx, groups in enumerate(self.word_groups):
            for group_idx in range(len(groups)):
                segment_index.append((rec_idx, group_idx))

        return segment_index

    def __len__(self) -> int:
        """Return total number of segments across all recordings."""
        return len(self.segment_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single word-aligned segment.

        Parameters
        ----------
        idx : int
            Global segment index

        Returns
        -------
        sample : Dict[str, Any]
            Dictionary containing MEG data, words, sensor info, and metadata
        """
        # Get recording and word group indices
        rec_idx, group_idx = self.segment_index[idx]

        # Get HDF5 file handle and recording metadata
        h5_file = self.file_handles[rec_idx]
        rec = self.recordings[rec_idx]

        # Get sampling frequency
        if rec["use_original"]:
            sfreq = h5_file.attrs.get('sample_frequency', 250.0)
        else:
            sfreq = h5_file.attrs["sample_freq"]

        # Get word group (list of words_per_segment word dicts)
        word_group = self.word_groups[rec_idx][group_idx]

        # Extract 3s windows for each word and concatenate
        subsegments = []
        sensor_types = rec['sensor_types']

        for word_info in word_group:
            # Convert time to samples
            start_sample = int(word_info['window_start'] * sfreq)
            end_sample = int(word_info['window_end'] * sfreq)

            # Load raw MEG data for this window
            if rec["use_original"]:
                all_data = h5_file["data"][:, start_sample:end_sample]
                ch_names = h5_file.attrs['channel_names'].split(', ')
                filtered_indices = [i for i, ch in enumerate(ch_names) if ch in rec['filtered_channels']]
                meg_subsegment = all_data[filtered_indices, :]
            else:
                meg_subsegment = h5_file["data"][:, start_sample:end_sample]

            # Apply preprocessing to this subsegment
            processed = _process_single_chunk(
                meg_subsegment,
                sensor_types,
                sfreq,
                self.baseline_duration,
                self.clip_range
            )

            subsegments.append(processed)

        # Concatenate along time axis to form 30s segment
        meg_data = np.concatenate(subsegments, axis=1)

        # Load sensor positions (same for all subsegments)
        sensor_xyzdir = rec['sensor_xyzdir'].copy()
        sensor_xyzdir = norm_sensor_positions(sensor_xyzdir)

        # Pad channel dimension if needed
        if self.max_channel_dim is not None:
            original_n_channels = meg_data.shape[0]
            meg_data = np.pad(meg_data, ((0, self.max_channel_dim - meg_data.shape[0]), (0, 0)))
            sensor_xyzdir = np.pad(sensor_xyzdir, ((0, self.max_channel_dim - sensor_xyzdir.shape[0]), (0, 0)))
            sensor_types = np.pad(sensor_types, (0, self.max_channel_dim - sensor_types.shape[0]))
            sensor_mask = np.zeros(self.max_channel_dim, dtype=np.float32)
            sensor_mask[:original_n_channels] = 1.0
        else:
            sensor_mask = np.ones(meg_data.shape[0], dtype=np.float32)

        # Extract word strings and subsegment boundaries
        words = [w['word'] for w in word_group]
        subsegment_boundaries = []
        cumulative_samples = 0
        for subseg in subsegments:
            subsegment_boundaries.append({
                'start_sample': cumulative_samples,
                'end_sample': cumulative_samples + subseg.shape[1]
            })
            cumulative_samples += subseg.shape[1]

        # Convert to tensors and return
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
            "sensor_mask": sensor_mask_tensor,
            "words": words,
            "subsegment_boundaries": subsegment_boundaries,
            "recording_idx": rec_idx,
            "segment_idx": group_idx,
            "start_time": float(word_group[0]['window_start']),
            "end_time": float(word_group[-1]['window_end']),
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
    dataset = LibriBrainWordAlignedDataset(
        data_root="/path/to/LibriBrain",
        segment_length=30.0,
        subsegment_duration=3.0,
        words_per_segment=10,
        window_onset_offset=-0.5,
        tasks=["Sherlock1"],
        l_freq=0.1,
        h_freq=125.0,
        target_sfreq=250.0,
    )

    print(f"\nDataset: {len(dataset)} segments")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nFirst sample:")
        print(f"  MEG shape: {sample['meg'].shape}")
        print(f"  Words: {sample['words']}")
        print(f"  Number of subsegments: {len(sample['subsegment_boundaries'])}")
        print(f"  Start time: {sample['start_time']:.2f}s")
        print(f"  End time: {sample['end_time']:.2f}s")
        print(f"  Duration: {sample['end_time'] - sample['start_time']:.2f}s")
        print(f"  Subject: {sample['subject']}")
        print(f"  Session: {sample['session']}")
        print(f"  Task: {sample['task']}")

        # Verify subsegment boundaries are continuous
        boundaries = sample['subsegment_boundaries']
        print(f"\n  Subsegment boundaries:")
        for i, bound in enumerate(boundaries):
            print(f"    {i}: samples {bound['start_sample']}-{bound['end_sample']} "
                  f"(duration: {bound['end_sample'] - bound['start_sample']} samples)")

        dataset.close()
