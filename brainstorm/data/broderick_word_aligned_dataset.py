"""PyTorch Dataset for word-aligned segments from the Broderick 2018 EEG dataset."""

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import scipy.io
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

from .preprocessing import _process_single_chunk
from .utils import norm_sensor_positions


class BroderickWordAlignedDataset(Dataset):
    """
    PyTorch Dataset for word-aligned segments from the Broderick 2018 EEG dataset.

    Each segment contains consecutive words, where each word has a time window
    aligned to its onset. The windows are concatenated to form a segment.
    Each subsegment is independently preprocessed with baseline correction,
    robust scaling, and clipping.

    The Broderick dataset uses pre-preprocessed H5 files stored under
    derivatives/sentences/ and word timing from MATLAB .mat files under
    Stimuli/Text/.

    Since this is EEG (not MEG), all 128 channels are treated as gradiometers
    (sensor_types=0) and sensor orientations are set to zero.

    Parameters
    ----------
    data_root : str
        Root directory of the Broderick dataset (e.g., "/path/to/broderick2018")
    segment_length : float
        Total segment length in seconds (should equal words_per_segment x subsegment_duration)
    subsegment_duration : float
        Duration of each word window in seconds. Default: 3.0
    words_per_segment : int
        Number of consecutive words per segment. Default: 10
    window_onset_offset : float
        Start time of window relative to word onset in seconds.
        Default: -0.5 (starts 0.5s before word onset)
    cache_dir : str, optional
        Directory for storing preprocessed cache files (unused for Broderick,
        kept for interface compatibility)
    subjects : List[str], optional
        List of subjects to include (e.g., ["sub-1", "sub-2"]). If None, use all.
    sessions : List[str], optional
        List of sessions to include (e.g., ["ses-1", "ses-2"]). If None, use all.
    tasks : List[str], optional
        List of tasks to include. If None, use all.
    val_subjects : List[str], optional
        List of subjects to use for validation. If None, no validation split.
    l_freq : float
        Low frequency cutoff (unused, kept for interface compatibility)
    h_freq : float
        High frequency cutoff (unused, kept for interface compatibility)
    target_sfreq : float
        Target sampling frequency (unused, kept for interface compatibility)
    max_channel_dim : int, optional
        Maximum channel dimension for padding (for multi-dataset training)
    baseline_duration : float
        Duration of baseline window for correction in seconds (default: 0.5)
    clip_range : tuple
        Min and max values for clipping after scaling (default: (-5, 5))
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
        max_channel_dim: Optional[int] = None,
        baseline_duration: float = 0.5,
        clip_range: tuple = (-5, 5),
        **kwargs,
    ):
        self.data_root = Path(data_root)
        self.segment_length = segment_length
        self.subsegment_duration = subsegment_duration
        self.words_per_segment = words_per_segment
        self.window_onset_offset = window_onset_offset
        self.baseline_duration = baseline_duration
        self.clip_range = clip_range
        self.max_channel_dim = max_channel_dim

        # Filters
        self.subjects = subjects
        self.sessions = sessions
        self.tasks = tasks
        self.val_subjects = val_subjects

        # Load sensor positions from CSV (128 channels, x/y/z in mm)
        sensor_csv_path = self.data_root / "sensor_xyz.csv"
        self.sensor_xyz = np.loadtxt(sensor_csv_path, delimiter=",", dtype=np.float64)  # (128, 3)

        # Discover all recordings
        self.recordings = self._discover_recordings()

        if len(self.recordings) == 0:
            raise ValueError(
                f"No recordings found in {self.data_root} with the specified filters. "
                f"Subjects: {subjects}, Sessions: {sessions}, Tasks: {tasks}"
            )

        # Open file handles for all H5 files
        self.file_handles: List[h5py.File] = []
        self._open_file_handles()

        # Parse events and build word groups
        self.word_groups: List[List[List[Dict]]] = []
        self._parse_all_events()

        # Build segment index: maps global index -> (recording_idx, word_group_idx)
        self.segment_index = self._build_segment_index()

    def _discover_recordings(self) -> List[Dict[str, Any]]:
        """
        Discover all pre-preprocessed H5 recordings in derivatives/sentences/.

        Returns
        -------
        recordings : List[Dict[str, Any]]
            List of recording metadata dictionaries
        """
        recordings = []
        sentences_dir = self.data_root / "derivatives" / "sentences"

        if not sentences_dir.exists():
            warnings.warn(f"Sentences directory not found: {sentences_dir}")
            return recordings

        # Pattern: sub-{X}_ses-{Y}_task-{task}_run-None.h5
        pattern = re.compile(
            r"sub-(\d+)_ses-(\d+)_task-(\w+)_run-None\.h5"
        )

        for h5_path in sorted(sentences_dir.glob("*.h5")):
            match = pattern.match(h5_path.name)
            if match is None:
                continue

            sub_num, ses_num, task = match.groups()
            subject = f"sub-{sub_num}"
            session = f"ses-{ses_num}"

            # Apply filters
            if self.subjects is not None and subject not in self.subjects:
                continue
            if self.sessions is not None and session not in self.sessions:
                continue
            if self.tasks is not None and task not in self.tasks:
                continue
            if self.val_subjects is not None and subject in self.val_subjects:
                continue

            # Word timing from Stimuli/Text/Run{ses_num}.mat
            events_path = self.data_root / "Stimuli" / "Text" / f"Run{ses_num}.mat"
            if not events_path.exists():
                warnings.warn(
                    f"Events file not found for {h5_path.name}: {events_path}, skipping"
                )
                continue

            recordings.append({
                "subject": subject,
                "session": session,
                "task": task,
                "h5_path": h5_path,
                "events_path": events_path,
            })

        return recordings

    def _open_file_handles(self) -> None:
        """Open HDF5 file handles for all recordings."""
        self.file_handles = []
        for rec in self.recordings:
            h5_file = h5py.File(rec["h5_path"], "r")
            self.file_handles.append(h5_file)

    def _parse_events_file(self, events_path: Path) -> pd.DataFrame:
        """
        Parse Broderick .mat events file and extract word onsets.

        Parameters
        ----------
        events_path : Path
            Path to Run{N}.mat file

        Returns
        -------
        events_df : pd.DataFrame
            DataFrame with 'onset' and 'value' columns for word events
        """
        mat = scipy.io.loadmat(str(events_path))

        words = [str(w[0][0]).strip().lower() for w in mat["wordVec"]]
        onsets = [float(t[0]) for t in mat["onset_time"]]

        events_df = pd.DataFrame({"onset": onsets, "value": words})
        events_df = events_df.sort_values("onset").reset_index(drop=True)

        return events_df

    def _build_word_groups(
        self, events_df: pd.DataFrame, recording_duration: float
    ) -> List[List[Dict]]:
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
            List of word groups, each containing words_per_segment word dicts
        """
        word_groups = []
        current_group = []

        for _, row in events_df.iterrows():
            word_value = str(row["value"]).strip().lower()
            word_onset = row["onset"]

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
                "word": word_value,
                "onset": word_onset,
                "window_start": window_start,
                "window_end": window_end,
                "subsegment_idx": len(current_group),
            })

            # Save complete group
            if len(current_group) == self.words_per_segment:
                word_groups.append(current_group.copy())
                current_group = []

        return word_groups

    def _parse_all_events(self) -> None:
        """Parse events for all recordings and build word groups."""
        self.word_groups = []

        for rec_idx, rec in enumerate(self.recordings):
            h5_file = self.file_handles[rec_idx]
            n_samples = int(h5_file["data"].attrs["n_samples"])
            sfreq = int(h5_file["data"].attrs["sfreq"])
            recording_duration = n_samples / sfreq

            # Parse events
            events_df = self._parse_events_file(rec["events_path"])

            # Build word groups
            groups = self._build_word_groups(events_df, recording_duration)
            self.word_groups.append(groups)

            print(
                f"Recording {rec_idx} ({rec['subject']} {rec['session']} task-{rec['task']}): "
                f"Found {len(groups)} word-aligned segments"
            )

    def _build_segment_index(self) -> List[Tuple[int, int]]:
        """
        Build an index mapping global segment index to (recording_idx, word_group_idx).
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
            Dictionary containing EEG data, words, sensor info, and metadata
        """
        rec_idx, group_idx = self.segment_index[idx]

        h5_file = self.file_handles[rec_idx]
        rec = self.recordings[rec_idx]
        sfreq = int(h5_file["data"].attrs["sfreq"])

        word_group = self.word_groups[rec_idx][group_idx]

        # EEG: all channels are gradiometers (type 0)
        n_channels = h5_file["data"].shape[0]
        sensor_types = np.zeros(n_channels, dtype=np.int64)

        # Extract subsegments for each word
        expected_subseg_samples = int(self.subsegment_duration * sfreq)
        subsegments = []
        for word_info in word_group:
            start_sample = int(word_info["window_start"] * sfreq)
            end_sample = start_sample + expected_subseg_samples

            eeg_subsegment = h5_file["data"][:, start_sample:end_sample]

            processed = _process_single_chunk(
                eeg_subsegment,
                sensor_types,
                sfreq,
                self.baseline_duration,
                self.clip_range,
            )
            subsegments.append(processed)

        # Concatenate along time axis
        eeg_data = np.concatenate(subsegments, axis=1)

        # Build sensor_xyzdir: xyz positions from CSV + zero orientations
        sensor_xyzdir = np.concatenate(
            [self.sensor_xyz, np.zeros_like(self.sensor_xyz)], axis=1
        )  # (128, 6)
        sensor_xyzdir = norm_sensor_positions(sensor_xyzdir.copy())

        # Pad channel dimension if needed
        if self.max_channel_dim is not None:
            original_n_channels = eeg_data.shape[0]
            eeg_data = np.pad(
                eeg_data,
                ((0, self.max_channel_dim - eeg_data.shape[0]), (0, 0)),
            )
            sensor_xyzdir = np.pad(
                sensor_xyzdir,
                ((0, self.max_channel_dim - sensor_xyzdir.shape[0]), (0, 0)),
            )
            sensor_types = np.pad(
                sensor_types,
                (0, self.max_channel_dim - sensor_types.shape[0]),
            )
            sensor_mask = np.zeros(self.max_channel_dim, dtype=np.float32)
            sensor_mask[:original_n_channels] = 1.0
        else:
            sensor_mask = np.ones(eeg_data.shape[0], dtype=np.float32)

        # Extract word strings and subsegment boundaries
        words = [w["word"] for w in word_group]
        subsegment_boundaries = []
        cumulative_samples = 0
        for subseg in subsegments:
            subsegment_boundaries.append({
                "start_sample": cumulative_samples,
                "end_sample": cumulative_samples + subseg.shape[1],
            })
            cumulative_samples += subseg.shape[1]

        # Convert to tensors
        eeg_tensor = torch.from_numpy(eeg_data).float()
        sensor_xyzdir_tensor = torch.from_numpy(sensor_xyzdir).float()
        sensor_mask_tensor = torch.from_numpy(sensor_mask).float()
        sensor_types_tensor = torch.from_numpy(sensor_types).int()

        return {
            "meg": eeg_tensor,  # Key is "meg" for compatibility with eval scripts
            "subject": str(h5_file["data"].attrs["subject"]),
            "session": str(h5_file["data"].attrs["session"]),
            "task": str(h5_file["data"].attrs["task"]),
            "sensor_xyzdir": sensor_xyzdir_tensor,
            "sensor_types": sensor_types_tensor,
            "sensor_mask": sensor_mask_tensor,
            "words": words,
            "subsegment_boundaries": subsegment_boundaries,
            "recording_idx": rec_idx,
            "segment_idx": group_idx,
            "start_time": float(word_group[0]["window_start"]),
            "end_time": float(word_group[-1]["window_end"]),
        }

    def __del__(self):
        """Close all file handles when the dataset is destroyed."""
        self.close()

    def close(self):
        """Explicitly close all HDF5 file handles."""
        for h5_file in self.file_handles:
            try:
                h5_file.close()
            except Exception:
                pass
        self.file_handles = []


if __name__ == "__main__":
    dataset = BroderickWordAlignedDataset(
        data_root="/data/engs-pnpl/datasets/broderick2018",
        segment_length=150.0,
        subsegment_duration=3.0,
        words_per_segment=50,
        window_onset_offset=-0.5,
        subjects=["sub-1"],
        sessions=["ses-1"],
        tasks=["natural"],
    )

    print(f"\nDataset: {len(dataset)} segments")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nFirst sample:")
        print(f"  EEG shape: {sample['meg'].shape}")
        print(f"  Words: {sample['words']}")
        print(f"  Number of subsegments: {len(sample['subsegment_boundaries'])}")
        print(f"  Start time: {sample['start_time']:.2f}s")
        print(f"  End time: {sample['end_time']:.2f}s")
        print(f"  Subject: {sample['subject']}")
        print(f"  Session: {sample['session']}")
        print(f"  Task: {sample['task']}")
        print(f"  Sensor types (unique): {sample['sensor_types'].unique()}")
        print(f"  Sensor xyzdir shape: {sample['sensor_xyzdir'].shape}")

        breakpoint()

        dataset.close()
