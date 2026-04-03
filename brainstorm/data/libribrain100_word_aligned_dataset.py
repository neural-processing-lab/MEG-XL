"""PyTorch Dataset for word-aligned segments from the LibriBrain100 MEG dataset."""

import mne
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
    cache_preprocessed,
    load_cached,
    get_sensor_positions,
    compute_preproc_hash,
    _process_single_chunk,
)


class LibriBrain100WordAlignedDataset(Dataset):
    """
    PyTorch Dataset for word-aligned segments from LibriBrain100 MEG dataset.

    Each segment contains consecutive words, where each word has a time window
    aligned to its onset. The windows are concatenated to form a segment.
    Each subsegment is independently preprocessed with baseline correction,
    robust scaling, and clipping.

    LibriBrain100 contains 30 subjects listening to 2 chapters of Sherlock Holmes.
    Data is stored as .fif files which are preprocessed (bandpass + resample)
    and cached to HDF5 on first use.

    Parameters
    ----------
    data_root : str
        Root directory of the LibriBrain100 dataset
    segment_length : float
        Total segment length in seconds (should equal words_per_segment x subsegment_duration)
    subsegment_duration : float
        Duration of each word window in seconds. Default: 3.0
    words_per_segment : int
        Number of consecutive words per segment. Default: 10
    window_onset_offset : float
        Start time of window relative to word onset in seconds.
        Default: -0.5 (starts 0.5s before word onset)
    cache_dir : str
        Directory for storing preprocessed cache files
    subjects : List[str], optional
        List of subjects to include (e.g., ["subj01", "subj02"]). If None, use all.
    sessions : List[str], optional
        List of sessions/chapters to include (e.g., ["chapter_11", "chapter_12"]).
        If None, defaults to both chapters.
    tasks : List[str], optional
        Unused, kept for interface compatibility with evaluation scripts.
    l_freq : float
        Low frequency cutoff for band-pass filter (default: 0.1 Hz)
    h_freq : float
        High frequency cutoff for band-pass filter (default: 40.0 Hz)
    target_sfreq : float
        Target sampling frequency after resampling (default: 50.0 Hz)
    channel_filter : Callable[[str], bool]
        Filter function for channels. Default: MEG channels only.
    max_channel_dim : int, optional
        Maximum channel dimension for padding.
    baseline_duration : float
        Duration of baseline window for correction in seconds (default: 0.5)
    clip_range : tuple
        Min and max values for clipping after scaling (default: (-5, 5))
    fif_variant : str
        Which .fif variant to use: "ssp_ica" for SSP+ICA cleaned files,
        "basic" for minimally preprocessed files. Default: "ssp_ica"
    """

    def __init__(
        self,
        data_root: str,
        segment_length: float = 150.0,
        subsegment_duration: float = 3.0,
        words_per_segment: int = 50,
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
        clip_range: tuple = (-5, 5),
        fif_variant: str = "ssp_ica",
        **kwargs,
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
        self.fif_variant = fif_variant

        # Filters
        self.subjects = subjects
        self.sessions = sessions if sessions is not None else ["chapter_11", "chapter_12"]

        # Discover all recordings
        self.recordings = self._discover_recordings()

        if len(self.recordings) == 0:
            raise ValueError(
                f"No recordings found in {self.data_root} with the specified filters. "
                f"Subjects: {subjects}, Sessions: {sessions}"
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

    @staticmethod
    def _subj_to_event_prefix(subj: str) -> str:
        """Convert 'subj01' -> 'sub01' for event file naming."""
        num = subj.replace("subj", "")
        return f"sub{num}"

    @staticmethod
    def _session_to_chapter_num(session: str) -> str:
        """Convert 'chapter_11' -> '11'."""
        return session.replace("chapter_", "")

    def _get_available_subjects(self) -> List[str]:
        """Scan the data root for available subject directories."""
        subjects = []
        for d in sorted(self.data_root.iterdir()):
            if d.is_dir() and d.name.startswith("subj") and d.name != "extra":
                preproc_dir = d / "preproc"
                if preproc_dir.exists():
                    subjects.append(d.name)
        return subjects

    def _discover_recordings(self) -> List[Dict[str, Any]]:
        """
        Discover all .fif recordings with matching events files.

        Returns
        -------
        recordings : List[Dict[str, Any]]
            List of recording metadata dictionaries
        """
        recordings = []

        # Get subjects to iterate through
        if self.subjects is not None:
            subjects_to_check = self.subjects
        else:
            subjects_to_check = self._get_available_subjects()

        if len(subjects_to_check) == 0:
            warnings.warn(f"No subjects found in {self.data_root}")
            return recordings

        # Build filename suffix based on variant
        if self.fif_variant == "ssp_ica":
            fif_suffix = "_ssp_ica_meg.fif"
        else:
            fif_suffix = "_meg.fif"

        for subj in subjects_to_check:
            for session in self.sessions:
                ch_num = self._session_to_chapter_num(session)

                # Build .fif path
                fif_path = (
                    self.data_root / subj / "preproc"
                    / f"{subj}_sherlock_1_chapter_{ch_num}{fif_suffix}"
                )

                if not fif_path.exists():
                    warnings.warn(f"FIF file not found: {fif_path}, skipping")
                    continue

                # Build events path (note: sub01 not subj01)
                event_prefix = self._subj_to_event_prefix(subj)
                events_path = (
                    self.data_root / "events" / subj
                    / f"{event_prefix}_sherlock1_ch{ch_num}_events.tsv"
                )

                if not events_path.exists():
                    warnings.warn(
                        f"Events file not found for {fif_path.name}: {events_path}, skipping"
                    )
                    continue

                # Generate cache path
                cache_path = self.cache_dir / (
                    f"libribrain100_{subj}_{session}_"
                    f"preproc-{compute_preproc_hash(self.l_freq, self.h_freq, self.target_sfreq, 'MEG_only')}.h5"
                )

                recordings.append({
                    "subject": subj,
                    "session": session,
                    "task": "sherlock",
                    "raw_path": fif_path,
                    "events_path": events_path,
                    "cache_path": cache_path,
                })

        return recordings

    def _preprocess_all(self) -> None:
        """
        Preprocess all .fif recordings and cache to HDF5.

        Pipeline per recording:
        1. Load .fif via MNE
        2. Band-pass filter
        3. Resample to target frequency
        4. Filter to MEG channels only
        5. Cache to HDF5 (data + sensor positions)
        """
        for i, rec in enumerate(self.recordings):
            if rec["cache_path"].exists():
                print(
                    f"Using cached recording {i+1}/{len(self.recordings)}: "
                    f"{rec['subject']} {rec['session']}"
                )
                continue

            print(
                f"Preprocessing recording {i+1}/{len(self.recordings)}: "
                f"{rec['subject']} {rec['session']}"
            )

            # Load .fif file
            raw = mne.io.read_raw_fif(str(rec["raw_path"]), preload=True, verbose=False)

            # Band-pass filter
            raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, verbose=False, n_jobs=-1)

            # Resample
            raw.resample(sfreq=self.target_sfreq, verbose=False, n_jobs=-1)

            # Filter to MEG channels
            filtered_chs = [ch for ch in raw.ch_names if self.channel_filter(ch)]
            raw.pick(filtered_chs)

            # Cache preprocessed data (also stores sensor_xyzdir and sensor_types)
            metadata = {
                "subject": rec["subject"],
                "session": rec["session"],
                "task": rec["task"],
                "dataset": "libribrain100",
            }
            cache_preprocessed(
                raw, rec["cache_path"], metadata,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                target_sfreq=self.target_sfreq,
                channel_filter_name="MEG_only",
            )

            print(f"  Cached to {rec['cache_path']}")

    def _open_file_handles(self) -> None:
        """Open HDF5 file handles for all cached recordings."""
        self.file_handles = []
        for rec in self.recordings:
            h5_file = load_cached(rec["cache_path"])
            self.file_handles.append(h5_file)

    def _parse_events_file(self, events_path: Path) -> pd.DataFrame:
        """
        Parse LibriBrain100 events.tsv and filter to valid word events.

        Parameters
        ----------
        events_path : Path
            Path to events.tsv file

        Returns
        -------
        events_df : pd.DataFrame
            DataFrame with 'onset' and 'value' columns for word events
        """
        events_df = pd.read_csv(events_path, sep='\t')

        # Filter to word events only (exclude silence and phoneme)
        events_df = events_df[events_df['kind'] == 'word'].copy()

        # Rename columns for consistency
        events_df = events_df.rename(columns={
            'timemeg': 'onset',
            'segment': 'value',
        })

        # Remove invalid entries
        events_df = events_df[events_df['value'].notna()]

        # Sort by onset time
        events_df = events_df.sort_values('onset').reset_index(drop=True)

        return events_df[['onset', 'value']]

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
            word_value = str(row['value']).strip().lower()

            # Skip silence markers
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
                'subsegment_idx': len(current_group),
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
            n_samples = h5_file.attrs["n_samples"]
            sfreq = h5_file.attrs["sample_freq"]
            recording_duration = n_samples / sfreq

            # Parse events
            events_df = self._parse_events_file(rec["events_path"])

            # Build word groups
            groups = self._build_word_groups(events_df, recording_duration)
            self.word_groups.append(groups)

            print(
                f"Recording {rec_idx} ({rec['subject']} {rec['session']}): "
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
            Dictionary containing MEG data, words, sensor info, and metadata
        """
        rec_idx, group_idx = self.segment_index[idx]

        h5_file = self.file_handles[rec_idx]
        rec = self.recordings[rec_idx]
        sfreq = h5_file.attrs["sample_freq"]

        word_group = self.word_groups[rec_idx][group_idx]

        # Load sensor info from cache
        sensor_xyzdir = h5_file['sensor_xyzdir'][:]
        sensor_types = h5_file['sensor_types'][:]

        # Extract subsegments for each word
        expected_subseg_samples = int(self.subsegment_duration * sfreq)
        subsegments = []

        for word_info in word_group:
            start_sample = int(word_info['window_start'] * sfreq)
            end_sample = start_sample + expected_subseg_samples

            meg_subsegment = h5_file["data"][:, start_sample:end_sample]

            # Apply per-subsegment preprocessing
            processed = _process_single_chunk(
                meg_subsegment,
                sensor_types,
                sfreq,
                self.baseline_duration,
                self.clip_range,
            )
            subsegments.append(processed)

        # Concatenate along time axis
        meg_data = np.concatenate(subsegments, axis=1)

        # Normalize sensor positions
        sensor_xyzdir = norm_sensor_positions(sensor_xyzdir.copy())

        # Pad channel dimension if needed
        if self.max_channel_dim is not None:
            original_n_channels = meg_data.shape[0]
            meg_data = np.pad(
                meg_data,
                ((0, self.max_channel_dim - meg_data.shape[0]), (0, 0)),
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
            sensor_mask = np.ones(meg_data.shape[0], dtype=np.float32)

        # Extract word strings and subsegment boundaries
        words = [w['word'] for w in word_group]
        subsegment_boundaries = []
        cumulative_samples = 0
        for subseg in subsegments:
            subsegment_boundaries.append({
                'start_sample': cumulative_samples,
                'end_sample': cumulative_samples + subseg.shape[1],
            })
            cumulative_samples += subseg.shape[1]

        # Convert to tensors
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
            except Exception:
                pass
        self.file_handles = []


if __name__ == "__main__":
    dataset = LibriBrain100WordAlignedDataset(
        data_root="/data/engs-asr/LibriBrain100",
        segment_length=150.0,
        subsegment_duration=3.0,
        words_per_segment=50,
        window_onset_offset=-0.5,
        subjects=["subj01"],
        sessions=["chapter_11"],
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
        print(f"  Subject: {sample['subject']}")
        print(f"  Session: {sample['session']}")
        print(f"  Sensor xyzdir shape: {sample['sensor_xyzdir'].shape}")
        print(f"  Sensor types (unique): {sample['sensor_types'].unique()}")

        dataset.close()
