"""PyTorch Dataset for word-aligned windows from the Armeni 2022 MEG dataset."""

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from sklearn.preprocessing import RobustScaler
from collections import Counter

from .preprocessing import (
    preprocess_recording,
    cache_preprocessed,
    load_cached,
    get_cache_path
)


class WordWindowDataset(Dataset):
    """
    PyTorch Dataset for word-aligned windows from the Armeni 2022 MEG dataset.

    This dataset extracts variable-length windows aligned to word onsets, enabling
    classification or analysis of specific words. It reuses the preprocessing and
    caching infrastructure from ArmeniMEGDataset.

    Parameters
    ----------
    data_root : str
        Root directory of the Armeni dataset
    word_windows : List[Tuple[str, str, str, float, float, int]]
        List of (subject, session, task, start_time, end_time, label) tuples.
        Each tuple specifies a window to extract.
    cache_dir : str, optional
        Directory for storing preprocessed cache files (default: "./data/cache")
    l_freq : float
        Low frequency cutoff for band-pass filter (default: 0.1 Hz)
    h_freq : float
        High frequency cutoff for band-pass filter (default: 128.0 Hz)
    target_sfreq : float
        Target sampling frequency after resampling (default: 256.0 Hz)
    channel_filter : Callable[[str], bool]
        Filter function for channels
    max_channel_dim : int, optional
        Maximum channel dimension for padding. If specified, MEG data and sensor
        positions will be zero-padded to this dimension (default: None, no padding)

    Example
    -------
    >>> word_windows = [
    ...     ("sub-001", "ses-001", "compr", 10.5, 11.5, 0),  # Word class 0
    ...     ("sub-001", "ses-001", "compr", 25.3, 26.3, 1),  # Word class 1
    ... ]
    >>> dataset = WordWindowDataset(
    ...     data_root="/path/to/armeni2022",
    ...     word_windows=word_windows
    ... )
    >>> sample = dataset[0]
    >>> print(sample['meg'].shape)  # (n_channels, n_timepoints)
    >>> print(sample['label'])  # 0
    """

    def __init__(
        self,
        data_root: str,
        word_windows: List[Tuple[str, str, str, float, float, int]],
        cache_dir: str = "./data/cache",
        l_freq: float = 0.1,
        h_freq: float = 128.0,
        target_sfreq: float = 256.0,
        channel_filter: Callable[[str], bool] = lambda x: x.startswith('M'),
        max_channel_dim: Optional[int] = None
    ):
        self.data_root = Path(data_root)
        self.word_windows = word_windows
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.l_freq = l_freq
        self.h_freq = h_freq
        self.target_sfreq = target_sfreq
        self.channel_filter = channel_filter
        self.max_channel_dim = max_channel_dim

        # Discover unique recordings needed for the word windows
        self.recordings = self._discover_recordings()

        if len(self.recordings) == 0:
            raise ValueError(f"No recordings found for the specified word windows")

        # Preprocess and cache all recordings
        self._preprocess_all()

        # Open file handles for all cached recordings
        self.file_handles: List[h5py.File] = []
        self._open_file_handles()

        # Build word window index: maps global index -> (rec_idx, start_sample, end_sample, label)
        self.window_index = self._build_window_index()

    def _discover_recordings(self) -> List[Dict[str, Any]]:
        """
        Discover all unique recordings referenced in word_windows.

        Returns
        -------
        recordings : List[Dict[str, Any]]
            List of recording metadata dictionaries
        """
        # Get unique (subject, session, task) combinations
        unique_recordings = set()
        for subject, session, task, _, _, _ in self.word_windows:
            unique_recordings.add((subject, session, task))

        recordings = []
        for subject, session, task in sorted(unique_recordings):
            # Construct paths
            meg_file = (
                self.data_root / subject / session / "meg" /
                f"{subject}_{session}_task-{task}_meg.ds"
            )

            if not meg_file.exists():
                raise ValueError(f"MEG file not found: {meg_file}")

            cache_path = get_cache_path(
                self.cache_dir, subject, session, task,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                target_sfreq=self.target_sfreq,
                channel_filter_name="MEG_only"
            )

            recordings.append({
                "subject": subject,
                "session": session,
                "task": task,
                "raw_path": meg_file,
                "cache_path": cache_path
            })

        return recordings

    def _preprocess_all(self) -> None:
        """Preprocess all recordings that haven't been cached yet."""
        for i, rec in enumerate(self.recordings):
            if not rec["cache_path"].exists():
                print(f"Preprocessing recording {i+1}/{len(self.recordings)}: "
                      f"{rec['subject']} {rec['session']} {rec['task']}")

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
                    "dataset": "armeni2022"
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
                      f"{rec['subject']} {rec['session']} {rec['task']}")

    def _open_file_handles(self) -> None:
        """Open HDF5 file handles for all cached recordings."""
        self.file_handles = []
        for rec in self.recordings:
            h5_file = load_cached(rec["cache_path"])
            self.file_handles.append(h5_file)

    def _build_window_index(self) -> List[Tuple[int, int, int, int]]:
        """
        Build an index mapping global window index to (rec_idx, start_sample, end_sample, label).

        Returns
        -------
        window_index : List[Tuple[int, int, int, int]]
            List of (rec_idx, start_sample, end_sample, label) tuples
        """
        # Create a mapping from (subject, session, task) to rec_idx
        rec_map = {}
        for rec_idx, rec in enumerate(self.recordings):
            key = (rec["subject"], rec["session"], rec["task"])
            rec_map[key] = rec_idx

        window_index = []

        for subject, session, task, start_time, end_time, label in self.word_windows:
            # Find the recording index
            key = (subject, session, task)
            rec_idx = rec_map[key]

            # Get sampling frequency from the HDF5 file
            h5_file = self.file_handles[rec_idx]
            sfreq = h5_file.attrs["sample_freq"]
            n_samples = h5_file.attrs["n_samples"]

            # Convert times to samples
            start_sample = int(start_time * sfreq)
            end_sample = int(end_time * sfreq)

            # Validate bounds
            if start_sample < 0 or end_sample > n_samples:
                raise ValueError(
                    f"Window [{start_time:.2f}s, {end_time:.2f}s] is out of bounds "
                    f"for recording {subject} {session} {task} "
                    f"(duration: {n_samples/sfreq:.2f}s)"
                )

            window_index.append((rec_idx, start_sample, end_sample, label))

        return window_index

    def __len__(self) -> int:
        """Return total number of word windows."""
        return len(self.window_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single word-aligned window.

        Parameters
        ----------
        idx : int
            Global window index

        Returns
        -------
        sample : Dict[str, Any]
            Dictionary containing:
            - meg: torch.Tensor of shape (n_channels, n_timepoints) or (max_channel_dim, n_timepoints)
            - label: int (word class label)
            - subject: str
            - session: str
            - task: str
            - sensor_xyz: torch.Tensor of shape (n_channels, 3) or (max_channel_dim, 3)
            - sensor_mask: torch.Tensor of shape (n_channels,) or (max_channel_dim,)
            - start_time: float (seconds)
            - end_time: float (seconds)
        """
        # Get window information
        rec_idx, start_sample, end_sample, label = self.window_index[idx]

        # Get HDF5 file handle
        h5_file = self.file_handles[rec_idx]

        # Get recording metadata
        rec = self.recordings[rec_idx]

        # Get sampling frequency
        sfreq = h5_file.attrs["sample_freq"]

        # Load window data
        meg_data = h5_file["data"][:, start_sample:end_sample]

        # Apply same normalization as ArmeniMEGDataset

        # Baseline correction (subtract mean of first 0.5s or entire window if shorter)
        baseline_samples = min(round(sfreq * 0.5), meg_data.shape[1])
        meg_data = meg_data - np.mean(meg_data[:, :baseline_samples], axis=1, keepdims=True)

        # Robust scaler
        scaler = RobustScaler()
        # Transpose to (n_samples, n_channels) for sklearn, then transpose back
        meg_data = scaler.fit_transform(meg_data.T).T

        # Clamp to avoid extreme values
        meg_data = np.clip(meg_data, -5, 5)

        # Load sensor positions
        sensor_xyz = h5_file["sensor_xyz"][:]

        # Pad channel dimension and sensor positions if needed
        if self.max_channel_dim is not None:
            original_n_channels = meg_data.shape[0]
            meg_data = np.pad(meg_data, ((0, self.max_channel_dim - meg_data.shape[0]), (0, 0)))
            sensor_xyz = np.pad(sensor_xyz, ((0, self.max_channel_dim - sensor_xyz.shape[0]), (0, 0)))
            sensor_mask = np.zeros(self.max_channel_dim, dtype=np.float32)
            sensor_mask[:original_n_channels] = 1.0
        else:
            sensor_mask = np.ones(meg_data.shape[0], dtype=np.float32)

        # Calculate timing
        start_time = start_sample / sfreq
        end_time = end_sample / sfreq

        # Convert to torch tensors
        meg_tensor = torch.from_numpy(meg_data).float()
        sensor_xyz_tensor = torch.from_numpy(sensor_xyz).float()
        sensor_mask_tensor = torch.from_numpy(sensor_mask).float()

        return {
            "meg": meg_tensor,
            "label": label,
            "subject": h5_file.attrs["subject"],
            "session": h5_file.attrs["session"],
            "task": h5_file.attrs["task"],
            "sensor_xyz": sensor_xyz_tensor,
            "sensor_mask": sensor_mask_tensor,
            "start_time": float(start_time),
            "end_time": float(end_time)
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


def get_top_k_words(
    data_root: str,
    subjects: List[str],
    sessions: List[str],
    task: str = "compr",
    k: int = 10,
    min_count: int = 1
) -> List[str]:
    """
    Get the top-k most frequent words from event files.

    Parameters
    ----------
    data_root : str
        Root directory of the Armeni dataset
    subjects : List[str]
        List of subjects (e.g., ["sub-001"])
    sessions : List[str]
        List of sessions (e.g., ["ses-001", "ses-002"])
    task : str
        Task name (default: "compr")
    k : int
        Number of top words to return (default: 10)
    min_count : int
        Minimum number of occurrences for a word to be included (default: 1)

    Returns
    -------
    top_words : List[str]
        List of top-k most frequent words
    """
    data_root = Path(data_root)
    word_counts = Counter()

    # Collect word counts across all specified recordings
    for subject in subjects:
        for session in sessions:
            events_file = (
                data_root / subject / session / "meg" /
                f"{subject}_{session}_task-{task}_events.tsv"
            )

            if not events_file.exists():
                print(f"Warning: Events file not found: {events_file}")
                continue

            # Load events
            events_df = pd.read_csv(events_file, sep='\t')

            # Filter for word onset events only (word_onset_01, word_onset_02, etc.)
            events_df = events_df[events_df['type'].str.startswith('word_onset', na=False)].copy()

            # Count words
            for word in events_df['value']:
                # Skip if not a string (e.g., NaN)
                if not isinstance(word, str):
                    continue
                # Strip quotes and lowercase for consistency
                word = word.strip('"').lower()
                # Skip 'sp' (silence/pause marker)
                if word == 'sp':
                    continue
                word_counts[word] += 1

    # Filter by minimum count and get top-k
    filtered_counts = {word: count for word, count in word_counts.items() if count >= min_count}
    top_words = [word for word, _ in Counter(filtered_counts).most_common(k)]

    print(f"\nTop {k} words:")
    for i, word in enumerate(top_words):
        print(f"  {i+1}. '{word}' ({word_counts[word]} occurrences)")

    return top_words


def create_word_windows(
    events_df: pd.DataFrame,
    top_words: List[str],
    recording_duration: float,
    window_size: float = 1.0
) -> Tuple[List[float], List[float], List[int]]:
    """
    Create word-aligned windows for the top N words.

    Windows start at word onset and extend for window_size seconds.
    Words near recording boundaries are skipped.

    Parameters
    ----------
    events_df : pd.DataFrame
        DataFrame with 'onset' and 'value' columns
    top_words : List[str]
        List of words to extract windows for
    recording_duration : float
        Total duration of recording in seconds
    window_size : float
        Window size in seconds (default: 1.0)

    Returns
    -------
    start_times : List[float]
        Start times of windows (at word onset)
    end_times : List[float]
        End times of windows (onset + window_size)
    labels : List[int]
        Labels (indices into top_words)
    """
    start_times = []
    end_times = []
    labels = []

    for _, event in events_df.iterrows():
        word = event['value'].strip('"').lower()

        # Check if this word is in our top words list
        if word in top_words:
            onset = event['onset']

            # Calculate window boundaries: start at onset, extend window_size forward
            start_time = onset
            end_time = onset + window_size

            # Skip if window extends beyond recording boundaries
            if start_time < 0 or end_time > recording_duration:
                continue

            # Get label (index in top_words list)
            label = top_words.index(word)

            start_times.append(start_time)
            end_times.append(end_time)
            labels.append(label)

    return start_times, end_times, labels
