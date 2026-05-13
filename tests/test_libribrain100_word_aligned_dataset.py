import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

import brainstorm.evaluate_criss_cross_word_classification as eval_word_cls
from brainstorm.data import LibriBrain100WordAlignedDataset
from brainstorm.data.libribrain100_word_aligned_dataset import (
    TIMIT_TEST_SPEAKERS,
    TIMIT_VALIDATION_SPEAKERS,
)
from brainstorm.evaluate_criss_cross_word_classification import (
    _add_ovmi_metrics,
    _build_named_retrieval_sets,
    _primary_metric_key,
    RANDOM_NOISE_MODE_MATCHED_PER_SAMPLE_CHANNEL,
    CrissCrossWordEmbeddingExtractor,
    apply_input_noise,
    build_subject_to_idx,
    create_word_level_collate_fn,
    evaluate_epoch,
    get_dataset_class,
    make_subject_key,
    save_target_embeddings_npz,
    save_prediction_embeddings_npz,
    save_subject_embeddings_npz,
    save_val_test_word_counts_npz,
    training_step,
    write_prediction_embeddings_npz,
)
from brainstorm.losses.contrastive import SigLipLoss


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


def _dataset(root, split, tasks=None, subjects=None, **kwargs):
    return LibriBrain100WordAlignedDataset(
        data_root=root,
        split=split,
        tasks=tasks,
        subjects=subjects,
        cache_dir=str(Path(root[0] if isinstance(root, list) else root) / "cache"),
        l_freq=0.1,
        h_freq=125.0,
        target_sfreq=250.0,
        words_per_segment=1,
        subsegment_duration=0.1,
        window_onset_offset=0.0,
        **kwargs,
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


def test_libribrain100_word_metadata_survives_sample_and_collate(tmp_path):
    root = tmp_path / "LibriBrain2_hf"
    _write_sensor_json(root)
    _write_recording(root, "Sherlock1", "sub-1", "ses-11", [("segments/", "val_sherlock1")])

    val = _dataset(root, "val", tasks=["Sherlock1"])
    sample = val[0]
    assert sample["words"] == ["val_sherlock1"]
    assert sample["wordidxs"] == [0]
    assert sample["sentenceidxs"] == [0]
    assert sample["source_root"] == str(root)

    collate_fn = create_word_level_collate_fn({"val_sherlock1": 7})
    batch = collate_fn([sample])
    assert batch["word_labels"].tolist() == [7]
    assert batch["subsegment_info"][0]["subject_idx"] == -1
    assert batch["word_metadata"] == [{
        "task": "Sherlock1",
        "subject": "sub-1",
        "session": "ses-11",
        "target_word": "val_sherlock1",
        "wordidx": 0,
        "sentenceidx": 0,
        "word_label": 7,
        "subject_idx": -1,
    }]


def test_sherlock1_session11_half_train_split_for_multisub_config(tmp_path):
    root = tmp_path / "LibriBrain2_hf"
    _write_sensor_json(root)
    _write_recording(
        root,
        "Sherlock1",
        "sub-1",
        "ses-11",
        [
            ("segments/", "train_early_1"),
            ("segments/", "train_early_2"),
            ("segments/", "train_early_3"),
            ("segments/", "train_early_4"),
            ("segments/", "val_late_1"),
            ("segments/", "val_late_2"),
            ("segments/", "val_late_3"),
        ],
    )
    _write_recording(root, "Sherlock1", "sub-1", "ses-12", [("segments/", "test_word")])
    _write_recording(root, "Sherlock1", "sub-0", "ses-11", [("segments/", "dropped_subject0")])

    kwargs = {
        "tasks": ["Sherlock1"],
        "subjects": ["sub-1"],
        "sherlock1_session11_half_train": True,
    }
    train = _dataset(root, "train", **kwargs)
    val = _dataset(root, "val", **kwargs)
    test = _dataset(root, "test", **kwargs)

    assert set(_words(train)) == {
        "train_early_1",
        "train_early_2",
        "train_early_3",
        "train_early_4",
    }
    assert set(_words(val)) == {"val_late_1", "val_late_2", "val_late_3"}
    assert _words(test) == ["test_word"]


def test_libribrain100_subject_keys_are_namespaced_by_root_and_task(tmp_path):
    root1 = tmp_path / "LibriBrain_hf"
    root2 = tmp_path / "LibriBrain2_hf"
    _write_sensor_json(root1)

    _write_recording(root1, "Sherlock2", "sub-0", "ses-1", [("segments/", "root1_word")])
    _write_recording(root2, "Sherlock2", "sub-0", "ses-1", [("segments/", "root2_word")])
    _write_recording(root2, "Sherlock3", "sub-0", "ses-1", [("segments/", "task_word")])

    train = _dataset([root1, root2], "train", tasks=["Sherlock2", "Sherlock3"])
    subject_to_idx = build_subject_to_idx(train)

    assert len(subject_to_idx) == 3
    assert make_subject_key(root1, "Sherlock2", "sub-0") in subject_to_idx
    assert make_subject_key(root2, "Sherlock2", "sub-0") in subject_to_idx
    assert make_subject_key(root2, "Sherlock3", "sub-0") in subject_to_idx


def test_subject_indices_known_and_unknown_in_collate(tmp_path):
    root = tmp_path / "LibriBrain_hf"
    _write_sensor_json(root)
    _write_recording(root, "Sherlock1", "sub-0", "ses-1", [("segments/", "known")])
    _write_recording(root, "Sherlock1", "sub-1", "ses-11", [("segments/", "unknown")])

    train = _dataset(root, "train", tasks=["Sherlock1"])
    val = _dataset(root, "val", tasks=["Sherlock1"])
    subject_to_idx = build_subject_to_idx(train)

    train_batch = create_word_level_collate_fn({"known": 0}, subject_to_idx=subject_to_idx)([train[0]])
    val_batch = create_word_level_collate_fn({"unknown": 1}, subject_to_idx=subject_to_idx)([val[0]])

    assert train_batch["subsegment_info"][0]["subject_idx"] >= 0
    assert train_batch["word_metadata"][0]["subject_idx"] >= 0
    assert val_batch["subsegment_info"][0]["subject_idx"] == -1
    assert val_batch["word_metadata"][0]["subject_idx"] == -1


def test_subject_film_identity_initialization_preserves_output():
    torch.manual_seed(0)
    plain = CrissCrossWordEmbeddingExtractor(
        num_channels=2,
        latent_dim=4,
        embed_dim=4,
        hidden_dim=8,
        dropout=0.0,
    )
    film = CrissCrossWordEmbeddingExtractor(
        num_channels=2,
        latent_dim=4,
        embed_dim=4,
        hidden_dim=8,
        dropout=0.0,
        use_subject_film=True,
        num_subjects=2,
        subject_embedding_dim=3,
    )
    film.mlp.load_state_dict(plain.mlp.state_dict())

    features = torch.randn(2, 5, 4)
    assert torch.allclose(plain(features), film(features, subject_idx=1))
    assert torch.allclose(plain(features), film(features, subject_idx=-1))


def test_training_step_with_subject_film_smoke():
    class DummyCrissCross(torch.nn.Module):
        latent_dim = 4

        def forward(self, meg, sensor_xyz, sensor_abc, sensor_types, sensor_mask, apply_mask=False):
            batch_size, n_channels, _ = meg.shape
            features = torch.ones(batch_size, n_channels, 10, self.latent_dim, device=meg.device)
            return {"features": features}

    word_mlp = CrissCrossWordEmbeddingExtractor(
        num_channels=2,
        latent_dim=4,
        embed_dim=4,
        hidden_dim=8,
        dropout=0.0,
        use_subject_film=True,
        num_subjects=1,
        subject_embedding_dim=3,
    )
    batch = {
        "meg": torch.zeros(1, 2, 24),
        "word_labels": torch.tensor([0]),
        "subsegment_info": [{
            "batch_idx": 0,
            "subseg_idx": 0,
            "start_sample": 0,
            "end_sample": 24,
            "subject_idx": -1,
        }],
        "sensor_xyzdir": torch.zeros(1, 2, 6),
        "sensor_types": torch.zeros(1, 2, dtype=torch.long),
        "sensor_mask": torch.ones(1, 2),
    }
    criterion = SigLipLoss(
        norm_kind="xy",
        temperature=False,
        bias=False,
        identical_candidates_threshold=None,
        reduction="sum",
    )

    loss, pred_embs, target_embs = training_step(
        batch,
        DummyCrissCross(),
        word_mlp,
        torch.ones(1, 4),
        criterion,
        "cpu",
    )

    assert torch.isfinite(loss)
    assert pred_embs.shape == (1, 4)
    assert target_embs.shape == (1, 4)


def test_random_noise_preserves_per_sample_channel_shape_mean_and_std():
    meg = torch.arange(2 * 3 * 100, dtype=torch.float32).reshape(2, 3, 100)
    generator = torch.Generator().manual_seed(123)
    noised = apply_input_noise(
        meg,
        RANDOM_NOISE_MODE_MATCHED_PER_SAMPLE_CHANNEL,
        generator=generator,
    )

    assert noised.shape == meg.shape
    assert torch.allclose(noised.mean(dim=-1), meg.mean(dim=-1), atol=1e-5)
    assert torch.allclose(
        noised.std(dim=-1, unbiased=False),
        meg.std(dim=-1, unbiased=False),
        atol=1e-4,
    )


def test_evaluate_epoch_with_random_noise_smoke():
    class DummyCrissCross(torch.nn.Module):
        latent_dim = 4

        def forward(self, meg, sensor_xyz, sensor_abc, sensor_types, sensor_mask, apply_mask=False):
            batch_size, n_channels, _ = meg.shape
            features = meg.mean(dim=-1, keepdim=True).expand(batch_size, n_channels, 10)
            features = features.unsqueeze(-1).expand(batch_size, n_channels, 10, self.latent_dim)
            return {"features": features}

    word_mlp = CrissCrossWordEmbeddingExtractor(
        num_channels=2,
        latent_dim=4,
        embed_dim=4,
        hidden_dim=8,
        dropout=0.0,
    )
    batch = {
        "meg": torch.randn(1, 2, 24),
        "word_labels": torch.tensor([0]),
        "subsegment_info": [{
            "batch_idx": 0,
            "subseg_idx": 0,
            "start_sample": 0,
            "end_sample": 24,
            "subject_idx": -1,
        }],
        "sensor_xyzdir": torch.zeros(1, 2, 6),
        "sensor_types": torch.zeros(1, 2, dtype=torch.long),
        "sensor_mask": torch.ones(1, 2),
    }
    criterion = SigLipLoss(
        norm_kind="xy",
        temperature=False,
        bias=False,
        identical_candidates_threshold=None,
        reduction="sum",
    )

    metrics = evaluate_epoch(
        DummyCrissCross(),
        word_mlp,
        [batch],
        torch.ones(1, 4),
        criterion,
        "cpu",
        retrieval_set_sizes=[1],
        k=1,
        k_values=[1],
        input_noise_mode=RANDOM_NOISE_MODE_MATCHED_PER_SAMPLE_CHANNEL,
        input_noise_seed=99,
    )

    assert "top1_accuracy_retrieval1" in metrics
    assert "loss" in metrics


def test_random_noise_prediction_export_npz(tmp_path):
    class DummyCrissCross(torch.nn.Module):
        latent_dim = 4

        def forward(self, meg, sensor_xyz, sensor_abc, sensor_types, sensor_mask, apply_mask=False):
            batch_size, n_channels, _ = meg.shape
            features = torch.ones(batch_size, n_channels, 10, self.latent_dim, device=meg.device)
            return {"features": features}

    word_mlp = CrissCrossWordEmbeddingExtractor(
        num_channels=2,
        latent_dim=4,
        embed_dim=4,
        hidden_dim=8,
        dropout=0.0,
    )
    batch = {
        "meg": torch.randn(1, 2, 24),
        "word_labels": torch.tensor([0]),
        "subsegment_info": [{
            "batch_idx": 0,
            "subseg_idx": 0,
            "start_sample": 0,
            "end_sample": 24,
            "subject_idx": -1,
        }],
        "word_metadata": [{
            "task": "Sherlock1",
            "subject": "sub-1",
            "session": "ses-12",
            "target_word": "hello",
            "wordidx": 0,
            "sentenceidx": 0,
            "word_label": 0,
            "subject_idx": -1,
        }],
        "sensor_xyzdir": torch.zeros(1, 2, 6),
        "sensor_types": torch.zeros(1, 2, dtype=torch.long),
        "sensor_mask": torch.ones(1, 2),
    }
    criterion = SigLipLoss(
        norm_kind="xy",
        temperature=False,
        bias=False,
        identical_candidates_threshold=None,
        reduction="sum",
    )

    save_prediction_embeddings_npz(
        tmp_path / "best_test_random_noise_predictions.npz",
        DummyCrissCross(),
        word_mlp,
        [batch],
        torch.ones(1, 4),
        criterion,
        "cpu",
        input_noise_mode=RANDOM_NOISE_MODE_MATCHED_PER_SAMPLE_CHANNEL,
        input_noise_seed=101,
    )

    prediction_npz = np.load(tmp_path / "best_test_random_noise_predictions.npz")
    assert prediction_npz["target_word"].tolist() == ["hello"]
    assert prediction_npz["subject_idx"].tolist() == [-1]
    assert prediction_npz["predicted_embeddings"].shape == (1, 4)


def test_subject_embedding_npz_writer(tmp_path):
    word_mlp = CrissCrossWordEmbeddingExtractor(
        num_channels=2,
        latent_dim=4,
        embed_dim=4,
        hidden_dim=8,
        dropout=0.0,
        use_subject_film=True,
        num_subjects=2,
        subject_embedding_dim=3,
    )
    subject_to_idx = {"root:task:sub-2": 1, "root:task:sub-1": 0}

    save_subject_embeddings_npz(
        tmp_path / "best_subject_embeddings.npz",
        word_mlp,
        subject_to_idx,
    )

    subject_npz = np.load(tmp_path / "best_subject_embeddings.npz")
    assert subject_npz["subject_keys"].tolist() == ["root:task:sub-1", "root:task:sub-2"]
    assert subject_npz["subject_indices"].tolist() == [0, 1]
    assert subject_npz["subject_embeddings"].shape == (2, 3)
    assert subject_npz["mean_subject_embedding"].shape == (3,)


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
    assert cfg.model.word_mlp.use_subject_film is True
    assert cfg.model.word_mlp.subject_embedding_dim == 64
    assert list(cfg.evaluation.k_values) == [1, 10]
    assert list(cfg.evaluation.ovmi_sets) == ["datafit50", "moses50"]
    assert cfg.evaluation.primary_k == 10
    assert len(cfg.evaluation.named_retrieval_sets.datafit50) == 50
    assert len(cfg.evaluation.named_retrieval_sets.moses50) == 50
    assert list(cfg.evaluation.named_retrieval_sets.moses50)[:3] == ["am", "are", "bad"]

    multisub_cfg = OmegaConf.load(
        "configs/eval_criss_cross_word_classification_libribrain100_multisub_train.yaml"
    )
    assert list(multisub_cfg.data.subjects) == [f"sub-{idx}" for idx in range(1, 33)]
    assert list(multisub_cfg.data.tasks) == ["Sherlock1"]
    assert multisub_cfg.data.sherlock1_session11_half_train is True
    assert multisub_cfg.evaluation.primary_metric == "balanced_top10_accuracy_datafit50"
    assert _primary_metric_key(multisub_cfg.evaluation) == "balanced_top10_accuracy_datafit50"
    assert multisub_cfg.evaluation.random_noise_test.enabled is True
    assert (
        multisub_cfg.evaluation.random_noise_test.mode
        == RANDOM_NOISE_MODE_MATCHED_PER_SAMPLE_CHANNEL
    )


def test_named_retrieval_set_resolution_handles_curly_apostrophe():
    word_to_idx = {"is": 0, "it's": 1, "on": 2}
    resolved = _build_named_retrieval_sets(
        {"datafit50": ["is", "it’s", "missing", "on"]},
        word_to_idx,
    )
    assert resolved == {"datafit50": [0, 1, 2]}


def test_ovmi_metrics_use_top1_balanced_accuracy(monkeypatch):
    class DummyOVMIResult:
        score = 1.5
        coverage = 0.25
        in_vocab_information = 6.0

    calls = []

    def fake_ovmi(vocabulary, accuracy, return_details):
        calls.append((vocabulary, accuracy, return_details))
        return DummyOVMIResult()

    monkeypatch.setattr(eval_word_cls, "compute_ovmi", fake_ovmi)
    metrics = {
        "balanced_top1_accuracy_datafit50": 0.42,
        "balanced_top10_accuracy_datafit50": 0.99,
    }

    _add_ovmi_metrics(
        metrics,
        {"datafit50": [1, 0], "other": [2]},
        ["alpha", "beta", "gamma"],
        ["datafit50"],
    )

    assert calls == [(["beta", "alpha"], 0.42, True)]
    assert metrics["ovmi_datafit50"] == 1.5
    assert metrics["ovmi_coverage_datafit50"] == 0.25
    assert metrics["ovmi_in_vocab_information_datafit50"] == 6.0
    assert metrics["ovmi_pc_datafit50"] == 0.42


def test_artifact_npz_writers(tmp_path):
    save_target_embeddings_npz(
        tmp_path,
        ["alpha", "beta"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    target_npz = np.load(tmp_path / "target_embeddings.npz")
    assert target_npz["words"].tolist() == ["alpha", "beta"]
    assert target_npz["target_embeddings"].shape == (2, 2)

    write_prediction_embeddings_npz(
        tmp_path / "best_val_predictions.npz",
        [{
            "task": "TIMIT",
            "subject": "sub-0",
            "session": "ses-1",
            "target_word": "hello",
            "wordidx": 3,
            "sentenceidx": 2,
            "word_label": 9,
        }],
        np.ones((1, 4), dtype=np.float32),
    )
    prediction_npz = np.load(tmp_path / "best_val_predictions.npz")
    assert prediction_npz["target_word"].tolist() == ["hello"]
    assert prediction_npz["wordidx"].tolist() == [3]
    assert prediction_npz["sentenceidx"].tolist() == [2]
    assert prediction_npz["word_label"].tolist() == [9]
    assert prediction_npz["predicted_embeddings"].shape == (1, 4)


def test_val_test_word_count_artifact_by_subset(tmp_path):
    root = tmp_path / "LibriBrain2_hf"
    _write_sensor_json(root)
    _write_recording(
        root,
        "Sherlock1",
        "sub-1",
        "ses-11",
        [("segments/", "repeat"), ("segments/", "repeat")],
    )
    _write_recording(root, "TheMoth", "sub-0", "ses-30", [("stimuli/c.wav", "moth_test")])

    val = _dataset(root, "val", tasks=["Sherlock1", "TheMoth"])
    test = _dataset(root, "test", tasks=["Sherlock1", "TheMoth"])
    save_val_test_word_counts_npz(tmp_path, val, test)

    counts_npz = np.load(tmp_path / "val_test_word_counts.npz")
    rows = {
        (split, subset, word): int(count)
        for split, subset, word, count in zip(
            counts_npz["split"],
            counts_npz["subset"],
            counts_npz["word"],
            counts_npz["count"],
        )
    }
    assert rows[("val", "Sherlock1", "repeat")] == 2
    assert rows[("test", "TheMoth", "moth_test")] == 1
