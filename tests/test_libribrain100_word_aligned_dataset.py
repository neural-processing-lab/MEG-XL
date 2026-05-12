import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from brainstorm.data import LibriBrain100WordAlignedDataset
from brainstorm.data.libribrain100_word_aligned_dataset import (
    TIMIT_TEST_SPEAKERS,
    TIMIT_VALIDATION_SPEAKERS,
)
from brainstorm.evaluate_criss_cross_word_classification import get_dataset_class


def _write_sensor_json(root: Path) -> None:
    sensors = [
        {
            "ch_name": "MEG001",
            "coil_type": 3024,
            "loc": [0.01, 0.02, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        {
            "ch_name": "MEG002",
            "coil_type": 3012,
            "loc": [0.02, 0.03, 0.04, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    ]
    root.mkdir(parents=True, exist_ok=True)
    (root / "meg_sensors_information.json").write_text(json.dumps(sensors))


def _write_recording(
    root: Path,
    task: str,
    subject: str,
    session: str,
    rows,
    run: str = "run-1",
    serialised_dir: str = "serialised",
    desc: Optional[str] = None,
) -> None:
    derivatives = root / task / "derivatives"
    h5_dir = derivatives / serialised_dir
    events_dir = derivatives / "events"
    h5_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)

    desc_part = f"_{desc}" if desc else ""
    h5_name = f"{subject}_{session}_task-{task}_{run}_proc-bads+headpos+sss+notch+bp+ds{desc_part}_meg.h5"
    with h5py.File(h5_dir / h5_name, "w") as h5_file:
        h5_file.create_dataset("data", data=np.zeros((2, 1000), dtype=np.float32))
        h5_file.attrs["sample_frequency"] = 250.0
        h5_file.attrs["highpass_cutoff"] = 0.1
        h5_file.attrs["lowpass_cutoff"] = 125.0
        h5_file.attrs["channel_names"] = "MEG001, MEG002"
        h5_file.attrs["channel_types"] = "mag, grad"

    event_rows = []
    for idx, (wavile, word) in enumerate(rows, start=1):
        event_rows.append({
            "idx": idx,
            "wavile": wavile,
            "kind": "word",
            "segment": word,
            "sentenceidx": 0,
            "wordidx": idx - 1,
            "phonemeidx": "",
            "timemeg": 1.0 + idx * 0.2,
            "timeds": 1.0 + idx * 0.2,
            "timesentence": idx * 0.2,
            "duration": 0.1,
        })
    events_name = f"{subject}_{session}_task-{task}_{run}_events.tsv"
    pd.DataFrame(event_rows).to_csv(events_dir / events_name, sep="\t", index=False)


def _dataset(root, split, tasks=None):
    return LibriBrain100WordAlignedDataset(
        data_root=root,
        split=split,
        tasks=tasks,
        cache_dir=str(Path(root[0] if isinstance(root, list) else root) / "cache"),
        l_freq=0.1,
        h_freq=125.0,
        target_sfreq=250.0,
        words_per_segment=1,
        subsegment_duration=0.1,
        window_onset_offset=0.0,
    )


def _words(dataset):
    return [group[0]["word"] for recording_groups in dataset.word_groups for group in recording_groups]


def test_sherlock_fixed_splits_and_competition_discovery(tmp_path):
    root1 = tmp_path / "LibriBrain_hf"
    root2 = tmp_path / "LibriBrain2_hf"
    _write_sensor_json(root1)

    _write_recording(root1, "Sherlock2", "sub-0", "ses-1", [("segments/", "train_sherlock2")])
    _write_recording(root2, "Sherlock1", "sub-1", "ses-11", [("segments/", "val_sherlock1")])
    _write_recording(root2, "Sherlock1", "sub-1", "ses-12", [("segments/", "test_sherlock1")])
    _write_recording(
        root2,
        "Sherlock1",
        "sub-13",
        "ses-1",
        [("segments/", "train_competition")],
        serialised_dir="serialised_competition",
        desc="desc-firsthalf",
    )

    train = _dataset([root1, root2], "train", tasks=["Sherlock1", "Sherlock2"])
    val = _dataset([root1, root2], "val", tasks=["Sherlock1", "Sherlock2"])
    test = _dataset([root1, root2], "test", tasks=["Sherlock1", "Sherlock2"])

    assert set(_words(train)) == {"train_sherlock2", "train_competition"}
    assert _words(val) == ["val_sherlock1"]
    assert _words(test) == ["test_sherlock1"]
    assert any(rec["source_serialised_dir"] == "serialised_competition" for rec in train.recordings)
    assert set(train.get_split_subset_indices()) == {"Sherlock", "Sherlock1"}
    assert set(val.get_split_subset_indices()) == {"Sherlock1"}
    assert set(test.get_split_subset_indices()) == {"Sherlock1"}


def test_themoth_and_mocha_timit_splits(tmp_path):
    root = tmp_path / "LibriBrain2_hf"
    _write_sensor_json(root)

    _write_recording(root, "TheMoth", "sub-0", "ses-28", [("stimuli/a.wav", "moth_train")])
    _write_recording(root, "TheMoth", "sub-0", "ses-29", [("stimuli/b.wav", "moth_val")])
    _write_recording(root, "TheMoth", "sub-0", "ses-30", [("stimuli/c.wav", "moth_test")])

    _write_recording(root, "MOCHATIMIT", "sub-0", "ses-1", [("stimuli/a.wav", "mocha_train_a")])
    _write_recording(root, "MOCHATIMIT", "sub-0", "ses-4", [("stimuli/d.wav", "mocha_train_d")])
    _write_recording(
        root,
        "MOCHATIMIT",
        "sub-0",
        "ses-2",
        [
            ("stimuli/b1.wav", "mocha_val_1"),
            ("stimuli/b2.wav", "mocha_val_2"),
            ("stimuli/b3.wav", "mocha_b_second_half"),
            ("stimuli/b4.wav", "mocha_b_second_half_2"),
        ],
    )
    _write_recording(
        root,
        "MOCHATIMIT",
        "sub-0",
        "ses-3",
        [
            ("stimuli/b1.wav", "mocha_c_overlap_1"),
            ("stimuli/b2.wav", "mocha_c_overlap_2"),
            ("stimuli/b3.wav", "mocha_test_1"),
            ("stimuli/b4.wav", "mocha_test_2"),
        ],
    )

    train = _dataset(root, "train", tasks=["TheMoth", "MOCHATIMIT"])
    val = _dataset(root, "val", tasks=["TheMoth", "MOCHATIMIT"])
    test = _dataset(root, "test", tasks=["TheMoth", "MOCHATIMIT"])

    assert set(_words(train)) == {"moth_train", "mocha_train_a", "mocha_train_d"}
    assert set(_words(val)) == {"moth_val", "mocha_val_1", "mocha_val_2"}
    assert set(_words(test)) == {"moth_test", "mocha_test_1", "mocha_test_2"}
    assert set(val.get_split_subset_indices()) == {"TheMoth", "MOCHATIMIT"}
    assert set(test.get_split_subset_indices()) == {"TheMoth", "MOCHATIMIT"}


def test_timit_speaker_split_sa_exclusion_and_train_overlap_exclusion(tmp_path):
    root = tmp_path / "LibriBrain2_hf"
    _write_sensor_json(root)

    test_speaker = sorted(TIMIT_TEST_SPEAKERS)[0].upper()
    val_speaker = sorted(TIMIT_VALIDATION_SPEAKERS)[0].upper()

    _write_recording(
        root,
        "TIMIT",
        "sub-0",
        "ses-1",
        [
            (f"stimuli/{test_speaker}_SX1.wav", "test_word"),
            (f"stimuli/{test_speaker}_SA1.wav", "test_sa_excluded"),
            (f"stimuli/{val_speaker}_SX2.wav", "val_word"),
            (f"stimuli/{val_speaker}_SA1.wav", "val_sa_excluded"),
            ("stimuli/MABC0_SX1.wav", "train_overlap_excluded"),
            ("stimuli/MZZY0_SX9.wav", "train_word"),
            ("stimuli/MZZY0_SA1.wav", "train_sa_allowed"),
        ],
    )

    train = _dataset(root, "train", tasks=["TIMIT"])
    val = _dataset(root, "val", tasks=["TIMIT"])
    test = _dataset(root, "test", tasks=["TIMIT"])

    assert set(_words(train)) == {"train_word", "train_sa_allowed"}
    assert _words(val) == ["val_word"]
    assert _words(test) == ["test_word"]
    assert set(val.get_split_subset_indices()) == {"TIMIT"}
    assert set(test.get_split_subset_indices()) == {"TIMIT"}


def test_missing_sensor_geometry_fails_loudly(tmp_path):
    root = tmp_path / "LibriBrain_hf"
    _write_recording(root, "Sherlock1", "sub-0", "ses-1", [("segments/", "word")])

    with pytest.raises(FileNotFoundError, match="sensor geometry"):
        _dataset(root, "train", tasks=["Sherlock1"])


def test_libribrain100_factory_and_config_smoke():
    assert get_dataset_class("libribrain100") is LibriBrain100WordAlignedDataset

    cfg = OmegaConf.load("configs/eval_criss_cross_word_classification_libribrain100.yaml")
    assert cfg.data.dataset_type == "libribrain100"
    assert cfg.data.split_strategy == "libribrain100"
    assert list(cfg.data.root) == ["/data/engs-asr/LibriBrain_hf", "/data/engs-asr/LibriBrain2_hf"]
