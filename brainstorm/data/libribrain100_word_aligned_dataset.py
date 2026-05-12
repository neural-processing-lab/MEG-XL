"""Word-aligned dataset for the LibriBrain100 fine-tuning corpus."""

import hashlib
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import (
    _process_single_chunk,
    cache_preprocessed,
    compute_preproc_hash,
    load_cached,
    load_libribrain_sensors,
    preprocess_libribrain_h5,
)
from .utils import norm_sensor_positions


TIMIT_TEST_SPEAKERS = {
    "mdab0", "mwbt0", "felc0", "mtas1", "mwew0",
    "fpas0", "mjmp0", "mlnt0", "fpkt0", "mlll0",
    "mtls0", "fjlm0", "mbpm0", "mklt0", "fnlp0",
    "mcmj0", "mjdh0", "fmgd0", "mgrt0", "mnjm0",
    "fdhc0", "mjln0", "mpam0", "fmld0",
}

TIMIT_VALIDATION_SPEAKERS = {
    "faks0", "fdac1", "fjem0", "mgwt0", "mjar0",
    "mmdb1", "mmdm2", "mpdf0", "fcmh0", "fkms0",
    "mbdg0", "mbwm0", "mcsh0", "fadg0", "fdms0",
    "fedw0", "mgjf0", "mglb0", "mrtk0", "mtaa0",
    "mtdt0", "mthc0", "mwjg0", "fnmr0", "frew0",
    "fsem0", "mbns0", "mmjr0", "mdls0", "mdlf0",
    "mdvc0", "mers0", "fmah0", "fdrw0", "mrcs0",
    "mrjm4", "fcal1", "mmwh0", "fjsj0", "majc0",
    "mjsw0", "mreb0", "fgjd0", "fjmg0", "mroa0",
    "mteb0", "mjfc0", "mrjr0", "fmml0", "mrws1",
}


class LibriBrain100WordAlignedDataset(Dataset):
    """
    Word-aligned LibriBrain100 dataset with fixed fine-tuning splits.

    LibriBrain100 combines newly downloaded LibriBrain HF and LibriBrain2 HF
    roots. It supports task-first HF layouts and applies the project split rules
    for Sherlock, TheMoth, TIMIT, and MOCHATIMIT.
    """

    VALID_SPLITS = {"train", "val", "test"}

    def __init__(
        self,
        data_root: Any,
        split: str = "train",
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
        channel_filter: Callable[[str], bool] = lambda x: x.startswith("MEG"),
        max_channel_dim: Optional[int] = None,
        baseline_duration: float = 0.5,
        clip_range: tuple = (-5, 5),
        allow_incomplete_segments: bool = False,
        include_competition_serialised: bool = True,
    ):
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(self.VALID_SPLITS)}, got {split!r}")

        self.data_roots = self._normalize_roots(data_root)
        self.data_root = self.data_roots[0]
        self.split = split
        self.segment_length = segment_length
        self.subsegment_duration = subsegment_duration
        self.words_per_segment = words_per_segment
        self.window_onset_offset = window_onset_offset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.subjects = subjects
        self.sessions = sessions
        self.tasks = tasks
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.target_sfreq = target_sfreq
        self.channel_filter = channel_filter
        self.max_channel_dim = max_channel_dim
        self.baseline_duration = baseline_duration
        self.clip_range = clip_range
        self.allow_incomplete_segments = allow_incomplete_segments
        self.include_competition_serialised = include_competition_serialised

        self.sensor_xyzdir_dict: Dict[str, np.ndarray] = {}
        self.sensor_types_dict: Dict[str, int] = {}

        self.recordings = self._discover_recordings()
        if len(self.recordings) == 0:
            raise ValueError(
                f"No LibriBrain100 recordings found in {self.data_roots}. "
                f"Subjects: {subjects}, Sessions: {sessions}, Tasks: {tasks}"
            )

        self._load_sensor_info()
        self._validate_sensor_geometry_available()
        self._prepare_split_metadata()
        self._preprocess_all()

        self.file_handles: List[h5py.File] = []
        self._open_file_handles()

        self.word_groups: List[List[List[Dict[str, Any]]]] = []
        self._parse_all_events()
        self.segment_index = self._build_segment_index()

        if len(self.segment_index) == 0:
            warnings.warn(f"LibriBrain100 split {self.split!r} produced zero word-aligned segments")

    @staticmethod
    def _normalize_roots(data_root: Any) -> List[Path]:
        if isinstance(data_root, (str, Path)):
            roots = [data_root]
        elif isinstance(data_root, Iterable):
            roots = list(data_root)
        else:
            raise TypeError(f"data_root must be a path or sequence of paths, got {type(data_root)!r}")
        return [Path(root) for root in roots]

    def _load_sensor_info(self) -> None:
        sensor_paths: List[Path] = []
        for root in self.data_roots:
            candidates = [root / "meg_sensors_information.json"]
            candidates.extend(root.glob("*/meg_sensors_information.json"))
            candidates.extend((root / "serialized").glob("*/meg_sensors_information.json"))
            sensor_paths.extend(path for path in candidates if path.exists())

        for sensor_path in sorted(set(sensor_paths)):
            xyzdir, sensor_types = load_libribrain_sensors(str(sensor_path))
            self.sensor_xyzdir_dict.update(xyzdir)
            self.sensor_types_dict.update(sensor_types)

        if self.sensor_xyzdir_dict:
            print(f"Loaded LibriBrain100 sensor info for {len(self.sensor_xyzdir_dict)} channels")

    def _available_task_dirs(self, root: Path) -> List[Tuple[str, Path]]:
        task_dirs: List[Tuple[str, Path]] = []

        def has_derivatives(task_dir: Path) -> bool:
            derivatives = task_dir / "derivatives"
            return (
                (derivatives / "events").exists()
                and (
                    (derivatives / "serialised").exists()
                    or (derivatives / "serialised_competition").exists()
                )
            )

        if self.tasks is not None:
            for task in self.tasks:
                for task_dir in (root / task, root / "serialized" / task):
                    if task_dir.exists() and has_derivatives(task_dir):
                        task_dirs.append((task, task_dir))
            return task_dirs

        for base in (root, root / "serialized"):
            if not base.exists():
                continue
            for task_dir in sorted(base.iterdir()):
                if task_dir.is_dir() and not task_dir.name.startswith(".") and has_derivatives(task_dir):
                    task_dirs.append((task_dir.name, task_dir))
        return task_dirs

    @staticmethod
    def _parse_filename(filename: str) -> Dict[str, str]:
        metadata: Dict[str, str] = {}
        for part in filename.replace(".h5", "").split("_"):
            if part.startswith("sub-"):
                metadata["subject"] = part
            elif part.startswith("ses-"):
                metadata["session"] = part
            elif part.startswith("run-"):
                metadata["run"] = part
            elif part.startswith("desc-"):
                metadata["desc"] = part
        return metadata

    @staticmethod
    def _construct_events_filename(metadata: Dict[str, str], task: str) -> str:
        return (
            f"{metadata['subject']}_{metadata['session']}_"
            f"task-{task}_{metadata['run']}_events.tsv"
        )

    def _cache_path_for(self, root: Path, metadata: Dict[str, str], task: str, h5_file: Path) -> Path:
        root_hash = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:8]
        preproc_hash = compute_preproc_hash(self.l_freq, self.h_freq, self.target_sfreq, "MEG_only")
        desc = metadata.get("desc", "desc-none")
        stem_hash = hashlib.sha256(str(h5_file).encode("utf-8")).hexdigest()[:8]
        filename = (
            f"libribrain100_{root_hash}_{metadata.get('subject', 'sub-unknown')}_"
            f"{metadata.get('session', 'ses-unknown')}_task-{task}_"
            f"{metadata.get('run', 'run-unknown')}_{desc}_{stem_hash}_preproc-{preproc_hash}.h5"
        )
        return self.cache_dir / filename

    def _discover_recordings(self) -> List[Dict[str, Any]]:
        recordings: List[Dict[str, Any]] = []

        for root in self.data_roots:
            for task, task_dir in self._available_task_dirs(root):
                derivatives = task_dir / "derivatives"
                events_dir = derivatives / "events"
                h5_dirs = [derivatives / "serialised"]
                if self.include_competition_serialised:
                    h5_dirs.append(derivatives / "serialised_competition")

                for h5_dir in h5_dirs:
                    if not h5_dir.exists():
                        continue
                    for h5_file in sorted(h5_dir.glob(f"*_task-{task}_*.h5")):
                        metadata = self._parse_filename(h5_file.name)
                        if self.subjects is not None and metadata.get("subject") not in self.subjects:
                            continue
                        if self.sessions is not None and metadata.get("session") not in self.sessions:
                            continue
                        if not {"subject", "session", "run"} <= set(metadata):
                            warnings.warn(f"Could not parse LibriBrain100 filename metadata: {h5_file}")
                            continue

                        events_file = events_dir / self._construct_events_filename(metadata, task)
                        if not events_file.exists():
                            warnings.warn(f"Events file not found for {h5_file.name}: {events_file}, skipping")
                            continue

                        recordings.append({
                            "root": root,
                            "task": task,
                            "subject": metadata["subject"],
                            "session": metadata["session"],
                            "run": metadata["run"],
                            "desc": metadata.get("desc"),
                            "raw_path": h5_file,
                            "events_path": events_file,
                            "cache_path": self._cache_path_for(root, metadata, task, h5_file),
                            "source_serialised_dir": h5_dir.name,
                            "use_original": False,
                        })

        return recordings

    @staticmethod
    def _h5_channel_names(h5_file: h5py.File) -> List[str]:
        if "channel_names" in h5_file:
            values = h5_file["channel_names"][:]
            return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
        attr = h5_file.attrs.get("channel_names")
        if isinstance(attr, bytes):
            attr = attr.decode("utf-8")
        if isinstance(attr, str):
            return [name.strip() for name in attr.split(",")]
        if attr is not None:
            return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in attr]
        raise ValueError(f"Missing channel names in HDF5 file: {h5_file.filename}")

    @staticmethod
    def _embedded_sensor_arrays(h5_file: h5py.File) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if "sensor_xyzdir" in h5_file and "sensor_types" in h5_file:
            return np.asarray(h5_file["sensor_xyzdir"][:]), np.asarray(h5_file["sensor_types"][:])
        if "sensor_xyzdir" in h5_file.attrs and "sensor_types" in h5_file.attrs:
            return np.asarray(h5_file.attrs["sensor_xyzdir"]), np.asarray(h5_file.attrs["sensor_types"])
        return None

    def _sensor_arrays_for_channels(
        self,
        channel_names: Sequence[str],
        source_h5: Optional[h5py.File] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.sensor_xyzdir_dict:
            missing = [ch for ch in channel_names if ch not in self.sensor_xyzdir_dict]
            if not missing:
                return (
                    np.asarray([self.sensor_xyzdir_dict[ch] for ch in channel_names]),
                    np.asarray([self.sensor_types_dict[ch] for ch in channel_names]),
                )

        if source_h5 is not None:
            embedded = self._embedded_sensor_arrays(source_h5)
            if embedded is not None:
                all_channel_names = self._h5_channel_names(source_h5)
                name_to_idx = {name: idx for idx, name in enumerate(all_channel_names)}
                if all(ch in name_to_idx for ch in channel_names):
                    xyzdir, sensor_types = embedded
                    indices = [name_to_idx[ch] for ch in channel_names]
                    return xyzdir[indices], sensor_types[indices]

        raise FileNotFoundError(
            "Could not find real LibriBrain100 sensor geometry. Expected "
            "meg_sensors_information.json under a configured root/task directory "
            "or embedded sensor_xyzdir/sensor_types arrays in the HDF5 files."
        )

    def _validate_sensor_geometry_available(self) -> None:
        for rec in self.recordings:
            with h5py.File(rec["raw_path"], "r") as h5_file:
                ch_names = self._h5_channel_names(h5_file)
                filtered_ch_names = [ch for ch in ch_names if self.channel_filter(ch)]
                self._sensor_arrays_for_channels(filtered_ch_names, h5_file)

    def _prepare_split_metadata(self) -> None:
        self.timit_test_sentence_ids: set = set()
        self.timit_excluded_train_speakers: set = set()
        speaker_to_sentences: Dict[str, set] = {}

        for rec in self.recordings:
            if rec["task"] != "TIMIT":
                continue
            events_df = pd.read_csv(rec["events_path"], sep="\t")
            word_events = events_df[events_df["kind"] == "word"]
            for wavile in word_events["wavile"].dropna().unique():
                parsed = self._parse_timit_wavile(str(wavile))
                if parsed is None:
                    continue
                speaker, sentence_id = parsed
                if sentence_id.startswith("SA"):
                    continue
                speaker_to_sentences.setdefault(speaker, set()).add(sentence_id)
                if speaker in TIMIT_TEST_SPEAKERS:
                    self.timit_test_sentence_ids.add(sentence_id)

        for speaker, sentence_ids in speaker_to_sentences.items():
            if speaker in TIMIT_TEST_SPEAKERS or speaker in TIMIT_VALIDATION_SPEAKERS:
                continue
            if sentence_ids & self.timit_test_sentence_ids:
                self.timit_excluded_train_speakers.add(speaker)

        self.mocha_validation_stimuli = self._build_mocha_validation_stimuli()

    @staticmethod
    def _parse_timit_wavile(wavile: str) -> Optional[Tuple[str, str]]:
        stem = Path(wavile).stem
        parts = stem.split("_")
        if len(parts) < 2:
            return None
        return parts[0].lower(), parts[1].upper()

    def _build_mocha_validation_stimuli(self) -> set:
        stimuli: List[str] = []
        seen = set()
        for rec in self.recordings:
            if rec["task"] != "MOCHATIMIT" or rec["session"] != "ses-2":
                continue
            events_df = pd.read_csv(rec["events_path"], sep="\t")
            word_events = events_df[events_df["kind"] == "word"]
            for wavile in word_events["wavile"].dropna():
                stimulus = str(wavile)
                if stimulus not in seen:
                    seen.add(stimulus)
                    stimuli.append(stimulus)

        return set(stimuli[: len(stimuli) // 2])

    def _event_belongs_to_split(self, row: pd.Series, rec: Dict[str, Any]) -> bool:
        task = rec["task"]
        session = rec["session"]

        if task.startswith("Sherlock"):
            if task == "Sherlock1" and session == "ses-11":
                return self.split == "val"
            if task == "Sherlock1" and session == "ses-12":
                return self.split == "test"
            return self.split == "train"

        if task == "TheMoth":
            if session == "ses-29":
                return self.split == "val"
            if session == "ses-30":
                return self.split == "test"
            return self.split == "train"

        if task == "TIMIT":
            parsed = self._parse_timit_wavile(str(row.get("wavile", "")))
            if parsed is None:
                return False
            speaker, sentence_id = parsed
            if sentence_id.startswith("SA") and self.split in {"val", "test"}:
                return False
            if speaker in TIMIT_TEST_SPEAKERS:
                return self.split == "test" and not sentence_id.startswith("SA")
            if speaker in TIMIT_VALIDATION_SPEAKERS:
                return self.split == "val" and not sentence_id.startswith("SA")
            if self.split != "train":
                return False
            return speaker not in self.timit_excluded_train_speakers

        if task == "MOCHATIMIT":
            stimulus = str(row.get("wavile", ""))
            if session in {"ses-1", "ses-4"}:
                return self.split == "train"
            if session == "ses-2":
                return self.split == "val" and stimulus in self.mocha_validation_stimuli
            if session == "ses-3":
                return self.split == "test" and stimulus not in self.mocha_validation_stimuli
            return False

        return self.split == "train"

    def _needs_reprocessing(self, h5_file: h5py.File) -> bool:
        existing_sfreq = h5_file.attrs.get("sample_frequency", h5_file.attrs.get("sample_freq", 250.0))
        existing_l_freq = h5_file.attrs.get("highpass_cutoff", h5_file.attrs.get("preproc_l_freq", 0.1))
        existing_h_freq = h5_file.attrs.get("lowpass_cutoff", h5_file.attrs.get("preproc_h_freq", 125.0))

        if abs(self.target_sfreq - existing_sfreq) > 0.1:
            return True
        if self.l_freq > existing_l_freq + 0.01:
            return True
        if self.h_freq < existing_h_freq - 0.1:
            return True
        return False

    def _preprocess_all(self) -> None:
        for i, rec in enumerate(self.recordings):
            with h5py.File(rec["raw_path"], "r") as h5_orig:
                rec["use_original"] = not self._needs_reprocessing(h5_orig)

            if rec["use_original"]:
                print(
                    f"Using original LibriBrain100 h5 file {i + 1}/{len(self.recordings)}: "
                    f"{rec['subject']} {rec['session']} {rec['task']} {rec['run']}"
                )
                continue

            if rec["cache_path"].exists():
                print(
                    f"Using cached LibriBrain100 re-preprocessed recording {i + 1}/{len(self.recordings)}: "
                    f"{rec['subject']} {rec['session']} {rec['task']} {rec['run']}"
                )
                continue

            print(
                f"Re-preprocessing LibriBrain100 recording {i + 1}/{len(self.recordings)}: "
                f"{rec['subject']} {rec['session']} {rec['task']} {rec['run']}"
            )
            raw = preprocess_libribrain_h5(
                str(rec["raw_path"]),
                self.sensor_xyzdir_dict,
                self.sensor_types_dict,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                target_sfreq=self.target_sfreq,
                channel_filter=self.channel_filter,
            )
            metadata = {
                "subject": rec["subject"],
                "session": rec["session"],
                "task": rec["task"],
                "run": rec["run"],
                "dataset": "libribrain100",
                "source_root": str(rec["root"]),
                "source_serialised_dir": rec["source_serialised_dir"],
            }
            if rec.get("desc"):
                metadata["desc"] = rec["desc"]
            cache_preprocessed(
                raw,
                rec["cache_path"],
                metadata,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                target_sfreq=self.target_sfreq,
                channel_filter_name="MEG_only",
            )

    def _open_file_handles(self) -> None:
        self.file_handles = []

        for rec in self.recordings:
            if rec["use_original"]:
                h5_file = h5py.File(rec["raw_path"], "r")
                ch_names = self._h5_channel_names(h5_file)
                filtered_ch_names = [ch for ch in ch_names if self.channel_filter(ch)]
                sensor_xyzdir, sensor_types = self._sensor_arrays_for_channels(filtered_ch_names, h5_file)
                rec["filtered_channels"] = filtered_ch_names
            else:
                h5_file = load_cached(rec["cache_path"])
                ch_names = self._h5_channel_names(h5_file)
                filtered_ch_names = [ch for ch in ch_names if self.channel_filter(ch)]
                with h5py.File(rec["raw_path"], "r") as source_h5:
                    sensor_xyzdir, sensor_types = self._sensor_arrays_for_channels(filtered_ch_names, source_h5)
                rec["filtered_channels"] = filtered_ch_names

            rec["sensor_xyzdir"] = np.asarray(sensor_xyzdir)
            rec["sensor_types"] = np.asarray(sensor_types)
            self.file_handles.append(h5_file)

    def _parse_events_file(self, events_path: Path, rec: Dict[str, Any]) -> pd.DataFrame:
        events_df = pd.read_csv(events_path, sep="\t")
        events_df = events_df[events_df["kind"] == "word"].copy()
        if events_df.empty:
            return pd.DataFrame(columns=["onset", "value"])

        split_mask = events_df.apply(lambda row: self._event_belongs_to_split(row, rec), axis=1)
        events_df = events_df[split_mask].copy()
        if events_df.empty:
            return pd.DataFrame(columns=["onset", "value"])

        events_df = events_df.rename(columns={"timemeg": "onset", "segment": "value"})
        events_df = events_df[events_df["value"].notna()]
        events_df = events_df.sort_values("onset").reset_index(drop=True)
        return events_df[["onset", "value"]]

    def _build_word_groups(self, events_df: pd.DataFrame, recording_duration: float) -> List[List[Dict[str, Any]]]:
        word_groups: List[List[Dict[str, Any]]] = []
        current_group: List[Dict[str, Any]] = []

        for _, row in events_df.iterrows():
            word_value = str(row["value"]).strip().lower()
            if word_value in {"silence", "sp", "nonspeech"}:
                continue

            word_onset = float(row["onset"])
            window_start = word_onset + self.window_onset_offset
            window_end = window_start + self.subsegment_duration

            if window_start < 0 or window_end > recording_duration:
                current_group = []
                continue

            current_group.append({
                "word": word_value,
                "onset": word_onset,
                "window_start": window_start,
                "window_end": window_end,
                "subsegment_idx": len(current_group),
            })

            if len(current_group) == self.words_per_segment:
                word_groups.append(current_group.copy())
                current_group = []

        if self.allow_incomplete_segments and current_group:
            word_groups.append(current_group.copy())

        return word_groups

    def _parse_all_events(self) -> None:
        self.word_groups = []

        for rec_idx, rec in enumerate(self.recordings):
            h5_file = self.file_handles[rec_idx]
            if rec["use_original"]:
                n_samples = h5_file["data"].shape[1]
                sfreq = h5_file.attrs.get("sample_frequency", h5_file.attrs.get("sample_freq", 250.0))
            else:
                n_samples = h5_file.attrs["n_samples"]
                sfreq = h5_file.attrs["sample_freq"]

            events_df = self._parse_events_file(rec["events_path"], rec)
            groups = self._build_word_groups(events_df, n_samples / sfreq)
            self.word_groups.append(groups)

            print(
                f"Recording {rec_idx} ({rec['subject']} {rec['session']} {rec['task']} {rec['run']} "
                f"{rec['source_serialised_dir']} split={self.split}): Found {len(groups)} word-aligned segments"
            )

    def _build_segment_index(self) -> List[Tuple[int, int]]:
        segment_index = []
        for rec_idx, groups in enumerate(self.word_groups):
            for group_idx in range(len(groups)):
                segment_index.append((rec_idx, group_idx))
        return segment_index

    def __len__(self) -> int:
        return len(self.segment_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_idx, group_idx = self.segment_index[idx]
        h5_file = self.file_handles[rec_idx]
        rec = self.recordings[rec_idx]

        if rec["use_original"]:
            sfreq = h5_file.attrs.get("sample_frequency", h5_file.attrs.get("sample_freq", 250.0))
            ch_names = self._h5_channel_names(h5_file)
            filtered_indices = [i for i, ch in enumerate(ch_names) if ch in rec["filtered_channels"]]
        else:
            sfreq = h5_file.attrs["sample_freq"]
            filtered_indices = None

        word_group = self.word_groups[rec_idx][group_idx]
        subsegments = []
        sensor_types = rec["sensor_types"]

        for word_info in word_group:
            start_sample = int(word_info["window_start"] * sfreq)
            end_sample = int(word_info["window_end"] * sfreq)

            if rec["use_original"]:
                all_data = h5_file["data"][:, start_sample:end_sample]
                meg_subsegment = all_data[filtered_indices, :]
            else:
                meg_subsegment = h5_file["data"][:, start_sample:end_sample]

            processed = _process_single_chunk(
                meg_subsegment,
                sensor_types,
                sfreq,
                self.baseline_duration,
                self.clip_range,
            )
            subsegments.append(processed)

        meg_data = np.concatenate(subsegments, axis=1)
        sensor_xyzdir = norm_sensor_positions(rec["sensor_xyzdir"].copy())

        if self.max_channel_dim is not None:
            original_n_channels = meg_data.shape[0]
            if original_n_channels > self.max_channel_dim:
                raise ValueError(
                    f"Recording has {original_n_channels} channels, exceeding max_channel_dim={self.max_channel_dim}"
                )
            pad_channels = self.max_channel_dim - original_n_channels
            meg_data = np.pad(meg_data, ((0, pad_channels), (0, 0)))
            sensor_xyzdir = np.pad(sensor_xyzdir, ((0, pad_channels), (0, 0)))
            sensor_types = np.pad(sensor_types, (0, pad_channels))
            sensor_mask = np.zeros(self.max_channel_dim, dtype=np.float32)
            sensor_mask[:original_n_channels] = 1.0
        else:
            sensor_mask = np.ones(meg_data.shape[0], dtype=np.float32)

        subsegment_boundaries = []
        cumulative_samples = 0
        for subseg in subsegments:
            subsegment_boundaries.append({
                "start_sample": cumulative_samples,
                "end_sample": cumulative_samples + subseg.shape[1],
            })
            cumulative_samples += subseg.shape[1]

        return {
            "meg": torch.from_numpy(meg_data).float(),
            "subject": rec["subject"],
            "session": rec["session"],
            "task": rec["task"],
            "sensor_xyzdir": torch.from_numpy(sensor_xyzdir).float(),
            "sensor_types": torch.from_numpy(sensor_types).int(),
            "sensor_mask": torch.from_numpy(sensor_mask).float(),
            "words": [w["word"] for w in word_group],
            "subsegment_boundaries": subsegment_boundaries,
            "recording_idx": rec_idx,
            "segment_idx": group_idx,
            "start_time": float(word_group[0]["window_start"]),
            "end_time": float(word_group[-1]["window_end"]),
        }

    def __del__(self):
        self.close()

    def close(self) -> None:
        for h5_file in self.file_handles:
            try:
                h5_file.close()
            except Exception:
                pass
        self.file_handles = []
