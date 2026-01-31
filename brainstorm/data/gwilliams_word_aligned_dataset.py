"""PyTorch Dataset for word-aligned segments from the Gwilliams 2022 MEG dataset."""

import ast
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings
import mne
from .utils import norm_sensor_positions

from .preprocessing import (
    cache_preprocessed,
    load_cached,
    get_cache_path,
    _process_single_chunk
)


def preprocess_gwilliams_recording(
    raw_path: str,
    l_freq: float = 0.1,
    h_freq: float = 40.0,
    target_sfreq: float = 50.0,
    channel_filter: Callable[[str], bool] = lambda _: True
) -> mne.io.Raw:
    """
    Preprocess a single Gwilliams MEG recording (KIT/Ricoh format).

    Parameters
    ----------
    raw_path : str
        Path to the raw MEG file (.con file for KIT/Ricoh format)
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
    # Load raw data (KIT/Ricoh format)
    raw = mne.io.read_raw_kit(raw_path, preload=True, verbose=False)

    # First, use MNE's pick_types to get only MEG channels (not ref_meg)
    meg_picks = mne.pick_types(raw.info, meg=True, ref_meg=False, exclude=[])
    raw.pick(meg_picks)

    # Band-pass filter
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False, n_jobs=-1)

    # Resample
    raw.resample(sfreq=target_sfreq, verbose=False, n_jobs=-1)

    # Apply additional channel filter if provided
    ch_names = [ch for ch in raw.ch_names if channel_filter(ch)]
    raw.pick(ch_names)

    return raw


class GwilliamsWordAlignedDataset(Dataset):
    """
    PyTorch Dataset for word-aligned 30s segments from Gwilliams 2022 MEG dataset.

    Each segment contains 10 consecutive words, where each word has a 3s
    window aligned to its onset. The 10 windows are concatenated to form
    a 30s segment. Each 3s subsegment is independently preprocessed with
    baseline correction, robust scaling, and clipping.

    Parameters
    ----------
    data_root : str
        Root directory of the Gwilliams dataset (e.g., "/path/to/gwilliams2022")
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
        List of subjects to include (e.g., ["sub-01", "sub-02"]). If None, use all.
    sessions : List[str], optional
        List of sessions to include (e.g., ["ses-0", "ses-1"]). If None, use all.
    tasks : List[str], optional
        List of tasks to include (e.g., ["0", "1", "2", "3"]). If None, use all.
    val_subjects : List[str], optional
        List of subjects to use for validation. If None, no validation split.
    l_freq : float
        Low frequency cutoff for band-pass filter (default: 0.1 Hz)
    h_freq : float
        High frequency cutoff for band-pass filter (default: 40.0 Hz)
    target_sfreq : float
        Target sampling frequency after resampling (default: 50.0 Hz)
    channel_filter : Callable[[str], bool]
        Filter function for channels. Channels for which this function returns True will be kept.
    max_channel_dim : int, optional
        Maximum channel dimension for padding (for multi-dataset training)
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
    >>> dataset = GwilliamsWordAlignedDataset(
    ...     data_root="/path/to/gwilliams2022",
    ...     segment_length=30.0,
    ...     subsegment_duration=3.0,
    ...     words_per_segment=10,
    ...     window_onset_offset=-0.5,
    ...     subjects=["sub-01"],
    ...     sessions=["ses-0"],
    ...     tasks=["0"]
    ... )
    >>> print(f"Dataset: {len(dataset)} segments")
    >>> sample = dataset[0]
    >>> print(f"MEG shape: {sample['meg'].shape}")
    >>> print(f"Words: {sample['words']}")
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
        val_subjects: Optional[List[str]] = None,
        l_freq: float = 0.1,
        h_freq: float = 40.0,
        target_sfreq: float = 50.0,
        channel_filter: Callable[[str], bool] = lambda x: True,
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
        self.val_subjects = val_subjects

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

            # Apply validation subject filter - skip if in val_subjects
            if self.val_subjects is not None and subject in self.val_subjects:
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

                # Find task MEG files (pattern: sub-XX_ses-X_task-X_meg.con)
                meg_files = sorted(meg_dir.glob(f"{subject}_{session}_task-*_meg.con"))

                for meg_file in meg_files:
                    # Extract task from filename
                    # Format: sub-XX_ses-X_task-X_meg.con
                    parts = meg_file.name.split("_")
                    task = None
                    for part in parts:
                        if part.startswith("task-"):
                            task = part.replace("task-", "")
                            break

                    if task is None:
                        continue

                    # Apply task filter
                    if self.tasks is not None and task not in self.tasks:
                        continue

                    # Check for corresponding events file
                    events_file = meg_dir / f"{subject}_{session}_task-{task}_events.tsv"
                    if not events_file.exists():
                        warnings.warn(f"Events file not found for {meg_file}, skipping")
                        continue

                    recordings.append({
                        "subject": subject,
                        "session": session,
                        "task": task,
                        "raw_path": meg_file,
                        "events_path": events_file,
                        "cache_path": get_cache_path(
                            self.cache_dir, subject, session, task,
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
                      f"{rec['subject']} {rec['session']} task-{rec['task']}")

                # Preprocess using Gwilliams-specific function
                raw = preprocess_gwilliams_recording(
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
                    "dataset": "gwilliams"
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
                      f"{rec['subject']} {rec['session']} task-{rec['task']}")

    def _open_file_handles(self) -> None:
        """
        Open HDF5 file handles for all cached recordings.
        """
        self.file_handles = []
        for rec in self.recordings:
            h5_file = load_cached(rec["cache_path"])
            self.file_handles.append(h5_file)

    def _parse_events_file(self, events_path: Path) -> pd.DataFrame:
        """
        Parse Gwilliams events.tsv and filter to valid word events.

        Gwilliams events format has a 'trial_type' column containing a dict-like
        string with 'kind' and 'word' keys. We parse this to extract word events.

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

        # Parse trial_type dict string and filter to word events
        word_events = []
        for _, row in events_df.iterrows():
            try:
                trial_info = ast.literal_eval(row['trial_type'])
                if trial_info.get('kind') == 'word':
                    word_events.append({
                        'onset': row['onset'],
                        'value': trial_info['word']
                    })
            except (ValueError, SyntaxError, KeyError):
                continue

        result_df = pd.DataFrame(word_events)

        if len(result_df) == 0:
            return result_df

        # Sort by onset time
        result_df = result_df.sort_values('onset').reset_index(drop=True)

        return result_df

    def _build_word_groups(self, events_df: pd.DataFrame, recording_duration: float) -> List[List[Dict]]:
        """
        Group consecutive valid words into segments.

        Unlike Armeni ('sp') or LibriBrain ('silence'), Gwilliams has no silence
        markers to skip - all entries are real words.

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
            n_samples = h5_file.attrs["n_samples"]
            sfreq = h5_file.attrs["sample_freq"]
            recording_duration = n_samples / sfreq

            # Parse events
            events_df = self._parse_events_file(rec["events_path"])

            # Build word groups
            groups = self._build_word_groups(events_df, recording_duration)
            self.word_groups.append(groups)

            print(f"Recording {rec_idx} ({rec['subject']} {rec['session']} task-{rec['task']}): "
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
        sfreq = h5_file.attrs["sample_freq"]

        # Get word group (list of words_per_segment word dicts)
        word_group = self.word_groups[rec_idx][group_idx]

        # Extract 3s windows for each word and concatenate
        subsegments = []
        sensor_types = h5_file["sensor_types"][:]

        for word_info in word_group:
            # Convert time to samples
            start_sample = int(word_info['window_start'] * sfreq)
            end_sample = int(word_info['window_end'] * sfreq)

            # Load raw MEG data for this window
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
        sensor_xyzdir = h5_file["sensor_xyzdir"][:]
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
            "subject": h5_file.attrs["subject"],
            "session": h5_file.attrs["session"],
            "task": h5_file.attrs["task"],
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
    dataset = GwilliamsWordAlignedDataset(
        data_root="/path/to/gwilliams2022",
        segment_length=30.0,
        subsegment_duration=3.0,
        words_per_segment=10,
        window_onset_offset=-0.5,
        subjects=["sub-01"],
        sessions=["ses-0"],
        tasks=["0"],
        l_freq=0.1,
        h_freq=40.0,
        target_sfreq=50.0,
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
