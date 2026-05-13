"""
Word classification evaluation for CrissCrossTransformer.

Evaluates on multiple subjects with session-based temporal split:
- Train: ses-001 through ses-008
- Val: ses-009
- Test: ses-010

Training: Uses ALL unique words (no vocabulary restriction, no OOV during training).

Evaluation: Reports top-10 accuracy when retrieving from subsets of most frequent words.
For a retrieval set of size K (e.g., top-50 or top-250):
- Only samples whose true label is in the top-K words are evaluated
- Retrieval is performed against the top-K word embeddings
- Samples with labels outside the top-K are skipped

Metrics:
- Top-10 retrieval accuracy for each retrieval set size
- Balanced top-10 accuracy (macro-averaged across retrieval set)
- Embedding quality metrics (cosine similarity, norms)

Usage:
    python -m brainstorm.evaluate_criss_cross_word_classification \
        model.criss_cross_checkpoint=path/to/ckpt.ckpt \
        training.batch_size=4 \
        training.num_epochs=50
"""

import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
from contextlib import nullcontext

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

torch.set_float32_matmul_precision("medium")

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
import pytorch_lightning as pl
from transformers import T5EncoderModel, T5Tokenizer
import wandb
from tqdm import tqdm
import pandas as pd
import numpy as np

from brainstorm.models.criss_cross_transformer import CrissCrossTransformerModule
from brainstorm.neuro_tokenizers.biocodec.model import BioCodecModel
from brainstorm.data.armeni_word_aligned_dataset import ArmeniWordAlignedDataset
from brainstorm.data.gwilliams_word_aligned_dataset import GwilliamsWordAlignedDataset
from brainstorm.data.libribrain_word_aligned_dataset import LibriBrainWordAlignedDataset
from brainstorm.data.libribrain100_word_aligned_dataset import LibriBrain100WordAlignedDataset
from brainstorm.losses.contrastive import SigLipLoss

logger = logging.getLogger(__name__)


def _base_model_for_state_dict(model: nn.Module) -> nn.Module:
    """Return the original module when torch.compile wraps it."""
    return getattr(model, "_orig_mod", model)


def _model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return _base_model_for_state_dict(model).state_dict()


def _load_model_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    base_model = _base_model_for_state_dict(model)
    has_compile_prefix = any(key.startswith("_orig_mod.") for key in state_dict)

    if base_model is not model:
        if has_compile_prefix:
            return model.load_state_dict(state_dict)
        return base_model.load_state_dict(state_dict)

    if has_compile_prefix:
        state_dict = {
            key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key: value
            for key, value in state_dict.items()
        }
    return model.load_state_dict(state_dict)


# ============================================================================
# Dataset Factory Functions
# ============================================================================

def get_dataset_class(dataset_type: str):
    """
    Get the appropriate dataset class based on dataset_type.

    Args:
        dataset_type: One of "armeni", "gwilliams", "libribrain", or "libribrain100"

    Returns:
        Dataset class to instantiate
    """
    dataset_classes = {
        "armeni": ArmeniWordAlignedDataset,
        "gwilliams": GwilliamsWordAlignedDataset,
        "libribrain": LibriBrainWordAlignedDataset,
        "libribrain100": LibriBrain100WordAlignedDataset,
    }

    if dataset_type not in dataset_classes:
        raise ValueError(
            f"Unknown dataset_type: {dataset_type}. "
            f"Must be one of: {list(dataset_classes.keys())}"
        )

    return dataset_classes[dataset_type]


def get_default_max_channel_dim(dataset_type: str) -> int:
    """
    Get the default max_channel_dim for each dataset type.

    - Armeni: 306 MEG channels (CTF system)
    - Gwilliams: 208 MEG channels (KIT/Ricoh system)
    - LibriBrain: 306 MEG channels (Elekta Neuromag system)
    - LibriBrain100: 306 MEG channels (Elekta Neuromag system)
    """
    defaults = {
        "armeni": 306,
        "gwilliams": 208,
        "libribrain": 306,
        "libribrain100": 306,
    }
    return defaults.get(dataset_type, 306)


def _config_list_or_none(value: Any) -> Optional[List[Any]]:
    """Convert Hydra scalar/list config values into a plain Python list or None."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if hasattr(value, "__iter__"):
        return list(value)
    return [value]


def _create_eval_subset_loaders(
    dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    collate_fn,
) -> Dict[str, DataLoader]:
    """Build evaluation loaders for datasets that expose split subset indices."""
    if not hasattr(dataset, "get_split_subset_indices"):
        return {}

    subset_loaders = {}
    for subset_name, indices in dataset.get_split_subset_indices().items():
        if len(indices) == 0:
            continue
        subset_loaders[subset_name] = DataLoader(
            torch.utils.data.Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
    return subset_loaders


def _retrieval_word_candidates(word: str) -> List[str]:
    normalized = str(word).strip().lower()
    candidates = [normalized]
    if "’" in normalized:
        candidates.append(normalized.replace("’", "'"))
    if "'" in normalized:
        candidates.append(normalized.replace("'", "’"))
    return list(dict.fromkeys(candidates))


def _build_named_retrieval_sets(
    named_retrieval_sets: Optional[Dict[str, Any]],
    word_to_idx: Dict[str, int],
) -> Dict[str, List[int]]:
    """Resolve config word lists to vocab indices, preserving list order."""
    if not named_retrieval_sets:
        return {}

    resolved = {}
    for set_name, words in named_retrieval_sets.items():
        indices = []
        missing = []
        seen_indices = set()

        for word in words:
            matched_idx = None
            for candidate in _retrieval_word_candidates(word):
                if candidate in word_to_idx:
                    matched_idx = word_to_idx[candidate]
                    break

            if matched_idx is None:
                missing.append(str(word))
                continue
            if matched_idx in seen_indices:
                continue

            seen_indices.add(matched_idx)
            indices.append(matched_idx)

        if missing:
            logger.warning(
                f"Named retrieval set '{set_name}' skipped {len(missing)} words not in vocabulary: {missing}"
            )
        if not indices:
            logger.warning(f"Named retrieval set '{set_name}' has no words present in the vocabulary")
            continue

        resolved[str(set_name)] = indices
        logger.info(f"Named retrieval set '{set_name}': {len(indices)} words resolved")

    return resolved


def _get_eval_k_values(evaluation_cfg: Any) -> List[int]:
    k_values = evaluation_cfg.get('k_values', None)
    if k_values is None:
        return [int(evaluation_cfg.k)]
    if isinstance(k_values, int):
        return [int(k_values)]
    return [int(k) for k in k_values]


# ============================================================================
# Model Components
# ============================================================================

class CrissCrossWordEmbeddingExtractor(nn.Module):
    """
    Extract word embeddings from CrissCrossTransformer features.

    Architecture per word subsegment:
    1. Extract features for subsegment time range: [C, ~62-63, 512]
    2. Mean pool over time: [C, 512]
    3. Flatten: [C * 512]
    4. MLP: [C * 512] -> [hidden_dim] -> [1024]
    """

    def __init__(
        self,
        num_channels: int,
        latent_dim: int = 512,
        embed_dim: int = 1024,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        use_subject_film: bool = False,
        num_subjects: int = 0,
        subject_embedding_dim: int = 64,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.use_subject_film = use_subject_film

        input_dim = num_channels * latent_dim
        if use_subject_film:
            if num_subjects <= 0:
                raise ValueError("num_subjects must be positive when use_subject_film=True")
            self.subject_embedding = nn.Embedding(num_subjects, subject_embedding_dim)
            self.subject_film = nn.Linear(subject_embedding_dim, 2 * input_dim)
            nn.init.zeros_(self.subject_film.weight)
            nn.init.zeros_(self.subject_film.bias)
        else:
            self.subject_embedding = None
            self.subject_film = None

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )

    def _subject_conditioning(self, subject_idx: Optional[Any], device: torch.device) -> torch.Tensor:
        if self.subject_embedding is None:
            raise RuntimeError("Subject conditioning requested but subject_embedding is not initialized")

        if subject_idx is None:
            idx_value = -1
        elif isinstance(subject_idx, torch.Tensor):
            idx_value = int(subject_idx.detach().cpu().item())
        else:
            idx_value = int(subject_idx)

        if idx_value < 0:
            return self.subject_embedding.weight.mean(dim=0)

        idx_tensor = torch.tensor(idx_value, dtype=torch.long, device=device)
        return self.subject_embedding(idx_tensor)

    def forward(self, features: torch.Tensor, subject_idx: Optional[Any] = None) -> torch.Tensor:
        """
        Args:
            features: [C, T_subseg, latent_dim] - features for one word subsegment
            subject_idx: Optional train-subject index. Negative/None uses the mean subject embedding.

        Returns:
            embedding: [embed_dim] - predicted T5 embedding
        """
        # Mean pool over time
        pooled = features.mean(dim=1)  # [C, latent_dim]

        # Flatten
        flat = pooled.reshape(-1)  # [C * latent_dim]

        if self.use_subject_film:
            subject_emb = self._subject_conditioning(subject_idx, flat.device)
            gamma_beta = self.subject_film(subject_emb)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            flat = flat * (1.0 + gamma) + beta

        # MLP projection
        embedding = self.mlp(flat)  # [embed_dim]

        return embedding


# ============================================================================
# Time Alignment Utilities
# ============================================================================

def map_raw_to_encoded_timesteps(
    start_sample: int,
    end_sample: int,
    downsample_ratio: int = 12
) -> Tuple[int, int]:
    """
    Map raw sample indices from subsegment_boundaries to encoded timesteps.

    BioCodec downsampling ratio: 12x (ratios=[3, 2, 2])
    - 30s segment: 7500 raw samples → 625 encoded timesteps
    - 3s word window: 750 raw samples → 62.5 encoded timesteps (62-63 in practice)

    Args:
        start_sample: Start index in raw samples (from subsegment_boundaries)
        end_sample: End index in raw samples
        downsample_ratio: BioCodec downsampling ratio (default: 12)

    Returns:
        (start_t, end_t): Encoded timestep range
    """
    # Convert to encoded timesteps
    start_t = start_sample // downsample_ratio
    end_t = (end_sample + downsample_ratio - 1) // downsample_ratio  # Ceiling division

    return start_t, end_t


# ============================================================================
# Vocabulary Building
# ============================================================================

def build_vocabulary_from_dataset(
    data_root: Path,
    subject: str,
    sessions: List[str],
    task: str = "compr",
    top_k: int = 50
) -> List[str]:
    """
    Parse events.tsv files for specified subject/sessions and extract top-K words.

    This builds vocabulary ONLY from training sessions to avoid data leakage.
    Words outside this vocabulary will be excluded from the loss calculation.

    Args:
        data_root: Path to armeni2022 dataset
        subject: Subject ID (e.g., "sub-001")
        sessions: List of session IDs (e.g., ["ses-001", "ses-002", ...])
        task: Task name (default: "compr")
        top_k: Number of most frequent words to keep (default: 50)

    Returns:
        vocab: List[str] of top-K words sorted by frequency
    """
    logger.info(f"Building vocabulary from {subject} sessions: {sessions}")

    word_counts = Counter()

    # Parse events files for each session
    for session in sessions:
        events_path = data_root / subject / session / "meg" / f"{subject}_{session}_task-{task}_events.tsv"

        if not events_path.exists():
            logger.warning(f"Events file not found: {events_path}")
            continue

        # Load TSV
        df = pd.read_csv(events_path, sep='\t')

        # Filter to word_onset events
        word_events = df[df['type'].str.startswith('word_onset', na=False)]

        # Count word frequencies
        for word in word_events['value']:
            clean_word = str(word).strip('"').lower()
            if clean_word != 'sp':  # Skip silence markers
                word_counts[clean_word] += 1

    # Get top-K most frequent words
    most_common = word_counts.most_common(top_k)
    vocab = [word for word, count in most_common]

    logger.info(f"  Total unique words: {len(word_counts)}")
    logger.info(f"  Selected top-{top_k} words")
    logger.info(f"  Frequency range: {most_common[0][1]} to {most_common[-1][1]}")

    return vocab


def hash_sentence_to_split(
    words: List[str],
    split_ratios: List[float],
    seed: int = 42
) -> str:
    """
    Deterministically assign a sentence to train/val/test split based on its hash.

    The sentence is the concatenation of all words in the segment.
    Same sentence across different sessions/subjects will always get the same split.

    Args:
        words: List of words forming the sentence
        split_ratios: [train_ratio, val_ratio, test_ratio] summing to 1.0
        seed: Random seed for hash salt (default: 42)

    Returns:
        split: One of "train", "val", or "test"
    """
    # Concatenate words with space separator to form sentence
    sentence = " ".join(words)

    # Add seed salt for reproducibility
    salted_sentence = f"{seed}:{sentence}"

    # Hash using SHA256 for deterministic output
    hash_obj = hashlib.sha256(salted_sentence.encode('utf-8'))
    hash_int = int(hash_obj.hexdigest(), 16)

    # Map hash to [0, 1) range
    # Use modulo with large number to get uniform distribution
    hash_float = (hash_int % 1_000_000) / 1_000_000.0

    # Assign to split based on cumulative ratios
    cumsum = 0.0
    for split_name, ratio in zip(["train", "val", "test"], split_ratios):
        cumsum += ratio
        if hash_float < cumsum:
            return split_name

    # Fallback (should never reach here if ratios sum to 1.0)
    return "test"


# ============================================================================
# T5 Embedding Generation
# ============================================================================

def generate_word_embeddings(
    vocab: List[str],
    vocab_size: Optional[int] = None,
    layer: int = 12,
    cache_dir: str = './embeddings_cache',
    device: str = 'cpu',
    verbose: bool = True,
    dataset_type: str = 'armeni'
) -> torch.Tensor:
    """
    Generate or load cached T5 embeddings for vocabulary words.

    Args:
        vocab: List of words to generate embeddings for
        vocab_size: Vocabulary size for cache filename (if None, uses len(vocab))
        layer: Which T5 layer to extract embeddings from (default: 12)
        cache_dir: Directory to store cached embeddings
        device: Device to run T5 model on ('cpu' or 'cuda')
        verbose: Whether to print progress messages
        dataset_type: Dataset type for cache filename (default: 'armeni')

    Returns:
        Tensor of shape [vocab_size, embedding_dim] containing word embeddings
    """
    # Setup cache directory
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine vocab size
    if vocab_size is None:
        vocab_size = len(vocab)

    # Create a hash of the vocabulary words to ensure cache correctness
    # This prevents loading wrong embeddings when vocab changes (e.g., train-only vs full dataset)
    vocab_hash = hashlib.sha256(" ".join(sorted(vocab)).encode()).hexdigest()[:8]

    # Check for cached embeddings
    cache_path = cache_dir / f'word_embeddings_{dataset_type}_{vocab_size}_layer{layer}_{vocab_hash}.pt'

    if cache_path.exists():
        if verbose:
            logger.info(f"Loading cached word embeddings from: {cache_path}")
        embeddings = torch.load(cache_path, map_location='cpu', weights_only=False)
        if verbose:
            logger.info(f"  Loaded embeddings shape: {embeddings.shape}")
        return embeddings

    # Generate embeddings using T5
    if verbose:
        logger.info(f"Generating T5 word embeddings for {len(vocab)} words...")
        logger.info(f"  Using T5-large, layer {layer}")
        logger.info(f"  Device: {device}")

    # Load T5 model
    t5 = T5EncoderModel.from_pretrained('t5-large')
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    t5 = t5.to(device)
    t5.eval()

    embeddings = []
    with torch.no_grad():
        for i, word in enumerate(vocab):
            if verbose and (i % 10 == 0 or i == len(vocab) - 1):
                logger.info(f"  Processing word {i+1}/{len(vocab)}: '{word}'")

            # Convert to lowercase for consistency
            word = word.lower()

            # Tokenize
            tokens = tokenizer(word, return_tensors='pt', padding=True)
            tokens = {k: v.to(device) for k, v in tokens.items()}

            # Forward pass
            outputs = t5(**tokens, output_hidden_states=True)

            # Extract hidden states from specified layer
            # Shape: (batch=1, seq_len, hidden_dim=1024)
            hidden_states = outputs.hidden_states[layer]

            # Ignore the last token (</s> end token)
            hidden_states = hidden_states[:, :-1, :]

            # Use mean pooling over token embeddings
            emb = hidden_states.mean(dim=1)  # Shape: (1, 1024)

            embeddings.append(emb.cpu())

    # Cleanup model
    del t5
    del tokenizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Stack embeddings
    embeddings = torch.cat(embeddings, dim=0)  # Shape: [vocab_size, 1024]

    if verbose:
        logger.info(f"  Generated embeddings shape: {embeddings.shape}")
        logger.info(f"  Saving to cache: {cache_path}")

    # Save to cache
    torch.save(embeddings, cache_path)

    return embeddings


# ============================================================================
# Custom Collate Function
# ============================================================================

def _path_hash(path_value: Any) -> str:
    path = Path(str(path_value)).expanduser().resolve()
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8]


def make_subject_key(source_root: Any, task: Any, subject: Any) -> str:
    subject_str = str(subject or "")
    task_str = str(task or "")
    if source_root:
        return f"{_path_hash(source_root)}:{task_str}:{subject_str}"
    if task_str:
        return f"{task_str}:{subject_str}"
    return subject_str


def _subject_key_from_sample(sample: Dict[str, Any]) -> str:
    return make_subject_key(
        sample.get("source_root", ""),
        sample.get("task", ""),
        sample.get("subject", ""),
    )


def _subject_key_from_recording(recording: Dict[str, Any]) -> str:
    return make_subject_key(
        recording.get("root", recording.get("source_root", "")),
        recording.get("task", ""),
        recording.get("subject", ""),
    )


def _resolve_subset(dataset: Any) -> Tuple[Any, Optional[List[int]]]:
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_dataset, parent_indices = _resolve_subset(dataset.dataset)
        indices = [int(idx) for idx in dataset.indices]
        if parent_indices is not None:
            indices = [parent_indices[idx] for idx in indices]
        return base_dataset, indices
    return dataset, None


def build_subject_to_idx(train_dataset: Any) -> Dict[str, int]:
    base_dataset, indices = _resolve_subset(train_dataset)
    subject_keys = set()

    if hasattr(base_dataset, "segment_index") and hasattr(base_dataset, "recordings"):
        segment_indices = indices if indices is not None else range(len(base_dataset.segment_index))
        for segment_idx in segment_indices:
            rec_idx, _group_idx = base_dataset.segment_index[int(segment_idx)]
            subject_keys.add(_subject_key_from_recording(base_dataset.recordings[rec_idx]))
    elif hasattr(base_dataset, "recordings"):
        for recording in base_dataset.recordings:
            subject_keys.add(_subject_key_from_recording(recording))

    return {subject_key: idx for idx, subject_key in enumerate(sorted(subject_keys))}


def create_word_level_collate_fn(
    word_to_idx: Dict[str, int],
    subject_to_idx: Optional[Dict[str, int]] = None,
):
    """
    Create a collate function that tracks word labels for SigLIP loss.

    Args:
        word_to_idx: Mapping from word string to vocabulary index (includes ALL words)
        subject_to_idx: Optional train-subject mapping. Unknown subjects map to -1.

    Returns:
        collate_fn: Collate function for DataLoader
    """
    def word_level_collate_fn(batch):
        """
        Collate function that expands 30s segments into individual word samples.

        All words are included since we train on the full vocabulary (no OOV filtering).
        Retrieval set filtering happens at evaluation time, not during collation.

        Input: List of dicts with keys:
            - meg: [C, 7500]
            - words: List[str] (10 words)
            - subsegment_boundaries: List[Dict] (10 boundaries)
            - sensor_xyzdir, sensor_types, sensor_mask

        Output: Dict with batched word samples:
            - meg: [B, C, 7500] - raw MEG for CrissCross
            - word_labels: [B*N] - word indices in vocabulary (N = words_per_segment * B)
            - subsegment_info: List of dicts with batch_idx, start, end, subject_idx for N words
            - word_metadata: List of per-word metadata rows for artifact export
            - sensor_xyzdir: [B, C, 6]
            - sensor_types: [B, C]
            - sensor_mask: [B, C]
        """
        batch_size = len(batch)

        # Pad MEG tensors to same length before stacking
        # (segments may have slightly different lengths after resampling)
        meg_tensors = [s['meg'] for s in batch]
        max_len = max(m.shape[-1] for m in meg_tensors)
        meg_padded = []
        for m in meg_tensors:
            if m.shape[-1] < max_len:
                pad_size = max_len - m.shape[-1]
                m = torch.nn.functional.pad(m, (0, pad_size), mode='constant', value=0)
            meg_padded.append(m)
        meg = torch.stack(meg_padded)
        sensor_xyzdir = torch.stack([s['sensor_xyzdir'] for s in batch])
        sensor_types = torch.stack([s['sensor_types'] for s in batch])
        sensor_mask = torch.stack([s['sensor_mask'] for s in batch])

        # Extract word labels and subsegment info
        word_labels = []
        subsegment_info = []
        word_metadata = []

        for batch_idx, sample in enumerate(batch):
            wordidxs = sample.get('wordidxs', [-1] * len(sample['words']))
            sentenceidxs = sample.get('sentenceidxs', [-1] * len(sample['words']))
            subject_key = _subject_key_from_sample(sample)
            subject_idx = subject_to_idx.get(subject_key, -1) if subject_to_idx is not None else -1
            for subseg_idx, (word, boundary, wordidx, sentenceidx) in enumerate(zip(
                sample['words'],
                sample['subsegment_boundaries'],
                wordidxs,
                sentenceidxs,
            )):
                # Include all words - no OOV filtering since vocab contains all words
                if word not in word_to_idx:
                    # This should never happen if vocab is built correctly
                    logger.warning(f"Word '{word}' not in vocabulary - skipping")
                    continue

                word_label = word_to_idx[word]
                word_labels.append(word_label)
                subsegment_info.append({
                    'batch_idx': batch_idx,
                    'subseg_idx': subseg_idx,
                    'start_sample': boundary['start_sample'],
                    'end_sample': boundary['end_sample'],
                    'subject_idx': int(subject_idx),
                })
                word_metadata.append({
                    'task': sample.get('task', ''),
                    'subject': sample.get('subject', ''),
                    'session': sample.get('session', ''),
                    'target_word': word,
                    'wordidx': int(wordidx) if wordidx is not None else -1,
                    'sentenceidx': int(sentenceidx) if sentenceidx is not None else -1,
                    'word_label': int(word_label),
                    'subject_idx': int(subject_idx),
                })

        return {
            'meg': meg,
            'word_labels': torch.tensor(word_labels, dtype=torch.long),
            'subsegment_info': subsegment_info,
            'word_metadata': word_metadata,
            'sensor_xyzdir': sensor_xyzdir,
            'sensor_types': sensor_types,
            'sensor_mask': sensor_mask
        }

    return word_level_collate_fn


# ============================================================================
# Checkpoint Loading
# ============================================================================

def load_tokenizer(ckpt_path: str, device: str = "cpu") -> BioCodecModel:
    """
    Load BioCodec tokenizer from checkpoint.

    Args:
        ckpt_path: Path to BioCodec checkpoint
        device: Device to load model on

    Returns:
        Loaded BioCodec tokenizer
    """
    logger.info(f"Loading BioCodec tokenizer from: {ckpt_path}")

    tokenizer = BioCodecModel._get_optimized_model()
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Handle torch.compile state dict prefix removal
    new_state_dict = {}
    for key, value in checkpoint["model_state_dict"].items():
        if key.startswith("_orig_mod."):
            new_key = key[len("_orig_mod."):]
        else:
            new_key = key
        new_state_dict[new_key] = value

    tokenizer.load_state_dict(new_state_dict)
    tokenizer.eval()

    logger.info("  BioCodec tokenizer loaded successfully")
    return tokenizer


def load_criss_cross_model(
    checkpoint_path: str,
    tokenizer: BioCodecModel,
    device: str = "cuda"
) -> CrissCrossTransformerModule:
    """
    Load CrissCrossTransformer from checkpoint.

    Args:
        checkpoint_path: Path to CrissCross checkpoint
        tokenizer: Loaded BioCodec tokenizer
        device: Device to load model on

    Returns:
        Loaded CrissCrossTransformerModule
    """
    logger.info(f"Loading CrissCross model from: {checkpoint_path}")

    # Load checkpoint manually to handle RoPE size mismatch
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract hyperparameters
    hparams = checkpoint['hyper_parameters']

    # Create model instance with saved hyperparameters
    model = CrissCrossTransformerModule(
        tokenizer=tokenizer,
        **hparams
    )

    # Handle RoPE size mismatch: Skip loading RoPE weights
    # RoPE rotation matrices are deterministically computed from position indices and frequencies:
    #   rotate = torch.polar(ones, torch.outer(positions, freqs))
    # where freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    # The model will recompute identical values when it expands to 625 positions on first forward pass.

    state_dict = checkpoint['state_dict']
    filtered_state_dict = {}
    skipped_rope_keys = []

    for key, value in state_dict.items():
        if 'rope_embedding_layer.rotate' in key:
            skipped_rope_keys.append(key)
        else:
            filtered_state_dict[key] = value

    logger.info(f"  Skipping {len(skipped_rope_keys)} RoPE rotation buffers (deterministic, will be recomputed)")

    # Load all weights except RoPE
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

    if missing_keys:
        logger.info(f"  Missing keys: {len(missing_keys)} (RoPE buffers)")
    if unexpected_keys:
        logger.warning(f"  Unexpected keys: {unexpected_keys}")

    logger.info(f"  Successfully loaded checkpoint (RoPE will auto-expand to 625 on first forward pass)")

    model.to(device)
    model.eval()  # Start in eval mode

    logger.info(f"  Latent dim: {model.latent_dim}")
    logger.info(f"  Num layers: {model.hparams.num_layers}")
    logger.info(f"  Num heads: {model.hparams.num_heads}")
    logger.info(f"  Loaded with strict=False (RoPE will auto-expand for sequence length)")

    return model


def initialize_criss_cross_from_scratch(
    checkpoint_path: str,
    tokenizer: BioCodecModel,
    device: str = "cuda"
) -> CrissCrossTransformerModule:
    """
    Initialize CrissCross with random weights using architecture from checkpoint.

    This extracts hyperparameters from a pretrained checkpoint but creates
    a fresh model with randomly initialized weights instead of loading the
    trained parameters.

    Args:
        checkpoint_path: Path to checkpoint (used only for architecture params)
        tokenizer: Loaded BioCodec tokenizer
        device: Device to load model on

    Returns:
        Randomly initialized CrissCrossTransformerModule
    """
    logger.info(f"Initializing CrissCross from scratch using architecture from: {checkpoint_path}")

    # Load checkpoint to extract hyperparameters
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = checkpoint['hyper_parameters']

    # Create model instance with saved hyperparameters but random weights
    model = CrissCrossTransformerModule(
        tokenizer=tokenizer,
        **hparams
    )

    logger.info(f"  Created model with random initialization")
    logger.info(f"  Latent dim: {model.latent_dim}")
    logger.info(f"  Num layers: {model.hparams.num_layers}")
    logger.info(f"  Num heads: {model.hparams.num_heads}")

    model.to(device)
    model.eval()  # Start in eval mode

    return model


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_top_k_accuracy(
    pred_embeddings: torch.Tensor,
    true_labels: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """
    Compute top-k retrieval accuracy.

    For each predicted embedding, find the k most similar vocabulary embeddings
    and check if the true label is among them.

    Args:
        pred_embeddings: [N, 1024] predicted word embeddings
        true_labels: [N] ground truth vocabulary indices
        vocab_embeddings: [vocab_size, 1024] T5 embeddings for all vocabulary words
        k_values: List of k values to compute accuracy for

    Returns:
        metrics: Dict with top-k accuracy for each k
    """
    # Compute cosine similarity
    pred_norm = F.normalize(pred_embeddings, p=2, dim=1)
    vocab_norm = F.normalize(vocab_embeddings, p=2, dim=1)
    similarity = torch.matmul(pred_norm, vocab_norm.T)  # [N, vocab_size]

    metrics = {}
    for k in k_values:
        _, top_k_indices = torch.topk(similarity, k=k, dim=1)  # [N, k]

        # Check if true label is in top-k
        true_labels_expanded = true_labels.unsqueeze(1).expand(-1, k)
        hits = (top_k_indices == true_labels_expanded).any(dim=1)

        accuracy = hits.float().mean().item()
        metrics[f'top{k}_accuracy'] = accuracy

    return metrics


def compute_top_k_accuracy_with_retrieval_set(
    pred_embeddings: torch.Tensor,
    true_labels: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    retrieval_set_size: int,
    k: int = 10
) -> Dict[str, Any]:
    """
    Compute top-k retrieval accuracy using a subset of most frequent words as retrieval set.

    Only samples whose true label is within the top retrieval_set_size words are evaluated.
    Retrieval is performed against only the retrieval set embeddings.

    Since vocabulary is ordered by frequency (most frequent first), the retrieval set
    consists of vocab indices 0 to retrieval_set_size-1.

    Args:
        pred_embeddings: [N, 1024] predicted word embeddings
        true_labels: [N] ground truth vocabulary indices (0 = most frequent word)
        vocab_embeddings: [vocab_size, 1024] T5 embeddings for all vocabulary words
        retrieval_set_size: Number of most frequent words to use as retrieval set (e.g., 50, 250)
        k: K value for top-k retrieval (default: 10)

    Returns:
        metrics: Dict with:
            - topk_accuracy: Top-k accuracy within the retrieval set
            - n_samples: Number of samples evaluated (with labels in retrieval set)
            - n_skipped: Number of samples skipped (labels outside retrieval set)
    """
    # Filter to samples whose true label is in the retrieval set
    # Since vocab is ordered by frequency, indices 0 to retrieval_set_size-1 are most frequent
    in_retrieval_set = true_labels < retrieval_set_size
    n_samples = in_retrieval_set.sum().item()
    n_skipped = len(true_labels) - n_samples

    if n_samples == 0:
        return {
            f'top{k}_accuracy_retrieval{retrieval_set_size}': 0.0,
            f'n_samples_retrieval{retrieval_set_size}': 0,
            f'n_skipped_retrieval{retrieval_set_size}': n_skipped
        }

    # Get filtered predictions and labels
    filtered_pred = pred_embeddings[in_retrieval_set]  # [n_samples, 1024]
    filtered_labels = true_labels[in_retrieval_set]  # [n_samples]

    # Get retrieval set embeddings (top retrieval_set_size words)
    retrieval_embeddings = vocab_embeddings[:retrieval_set_size]  # [retrieval_set_size, 1024]

    # Compute cosine similarity against retrieval set only
    pred_norm = F.normalize(filtered_pred, p=2, dim=1)
    retrieval_norm = F.normalize(retrieval_embeddings, p=2, dim=1)
    similarity = torch.matmul(pred_norm, retrieval_norm.T)  # [n_samples, retrieval_set_size]

    # Get top-k predictions (indices are now 0 to retrieval_set_size-1)
    actual_k = min(k, retrieval_set_size)
    _, top_k_indices = torch.topk(similarity, k=actual_k, dim=1)  # [n_samples, k]

    # Check if true label is in top-k
    filtered_labels_expanded = filtered_labels.unsqueeze(1).expand(-1, actual_k)
    hits = (top_k_indices == filtered_labels_expanded).any(dim=1)

    accuracy = hits.float().mean().item()

    return {
        f'top{k}_accuracy_retrieval{retrieval_set_size}': accuracy,
        f'n_samples_retrieval{retrieval_set_size}': n_samples,
        f'n_skipped_retrieval{retrieval_set_size}': n_skipped
    }


def compute_balanced_top_k_accuracy_with_retrieval_set(
    pred_embeddings: torch.Tensor,
    true_labels: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    retrieval_set_size: int,
    k: int = 10
) -> float:
    """
    Compute balanced (macro-averaged) top-k retrieval accuracy using a retrieval subset.

    Only considers classes within the retrieval set. For each class in the retrieval set,
    compute top-k accuracy and then macro-average across classes.

    Args:
        pred_embeddings: [N, 1024] predicted word embeddings
        true_labels: [N] ground truth vocabulary indices
        vocab_embeddings: [vocab_size, 1024] T5 embeddings for all vocabulary words
        retrieval_set_size: Number of most frequent words to use as retrieval set
        k: K value for top-k retrieval

    Returns:
        balanced_accuracy: Macro-averaged top-k accuracy across retrieval set classes
    """
    # Filter to samples whose true label is in the retrieval set
    in_retrieval_set = true_labels < retrieval_set_size
    n_samples = in_retrieval_set.sum().item()

    if n_samples == 0:
        return 0.0

    # Get filtered predictions and labels
    filtered_pred = pred_embeddings[in_retrieval_set]
    filtered_labels = true_labels[in_retrieval_set]

    # Get retrieval set embeddings
    retrieval_embeddings = vocab_embeddings[:retrieval_set_size]

    # Compute cosine similarity against retrieval set only
    pred_norm = F.normalize(filtered_pred, p=2, dim=1)
    retrieval_norm = F.normalize(retrieval_embeddings, p=2, dim=1)
    similarity = torch.matmul(pred_norm, retrieval_norm.T)

    actual_k = min(k, retrieval_set_size)
    _, top_k_indices = torch.topk(similarity, k=actual_k, dim=1)

    # Compute per-class accuracy for classes in retrieval set
    per_class_accuracies = []

    for class_idx in range(retrieval_set_size):
        class_mask = (filtered_labels == class_idx)
        n_class_samples = class_mask.sum().item()

        if n_class_samples == 0:
            continue

        class_top_k = top_k_indices[class_mask]
        hits = (class_top_k == class_idx).any(dim=1)
        class_acc = hits.float().mean().item()

        per_class_accuracies.append(class_acc)

    balanced_accuracy = sum(per_class_accuracies) / len(per_class_accuracies) if per_class_accuracies else 0.0

    return balanced_accuracy


def compute_top_k_accuracy_with_named_retrieval_set(
    pred_embeddings: torch.Tensor,
    true_labels: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    retrieval_indices: List[int],
    retrieval_name: str,
    k: int = 10
) -> Dict[str, Any]:
    """Compute top-k retrieval accuracy against an arbitrary named vocab subset."""
    retrieval_indices_for_labels = torch.tensor(retrieval_indices, dtype=torch.long, device=true_labels.device)
    if retrieval_indices_for_labels.numel() == 0:
        return {
            f'top{k}_accuracy_{retrieval_name}': 0.0,
            f'n_samples_{retrieval_name}': 0,
            f'n_skipped_{retrieval_name}': len(true_labels),
        }

    label_matches = true_labels.unsqueeze(1) == retrieval_indices_for_labels.unsqueeze(0)
    in_retrieval_set = label_matches.any(dim=1)
    n_samples = in_retrieval_set.sum().item()
    n_skipped = len(true_labels) - n_samples

    if n_samples == 0:
        return {
            f'top{k}_accuracy_{retrieval_name}': 0.0,
            f'n_samples_{retrieval_name}': 0,
            f'n_skipped_{retrieval_name}': n_skipped,
        }

    filtered_pred = pred_embeddings[in_retrieval_set]
    filtered_label_matches = label_matches[in_retrieval_set]
    filtered_label_positions = filtered_label_matches.int().argmax(dim=1)
    retrieval_indices_for_vocab = torch.tensor(retrieval_indices, dtype=torch.long, device=vocab_embeddings.device)
    retrieval_embeddings = vocab_embeddings[retrieval_indices_for_vocab].to(pred_embeddings.device)

    pred_norm = F.normalize(filtered_pred, p=2, dim=1)
    retrieval_norm = F.normalize(retrieval_embeddings, p=2, dim=1)
    similarity = torch.matmul(pred_norm, retrieval_norm.T)

    actual_k = min(k, len(retrieval_indices))
    _, top_k_indices = torch.topk(similarity, k=actual_k, dim=1)

    hits = (top_k_indices == filtered_label_positions.unsqueeze(1)).any(dim=1)
    accuracy = hits.float().mean().item()

    return {
        f'top{k}_accuracy_{retrieval_name}': accuracy,
        f'n_samples_{retrieval_name}': n_samples,
        f'n_skipped_{retrieval_name}': n_skipped,
    }


def compute_balanced_top_k_accuracy_with_named_retrieval_set(
    pred_embeddings: torch.Tensor,
    true_labels: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    retrieval_indices: List[int],
    k: int = 10
) -> float:
    """Compute macro-averaged top-k accuracy for an arbitrary named vocab subset."""
    retrieval_indices_for_labels = torch.tensor(retrieval_indices, dtype=torch.long, device=true_labels.device)
    if retrieval_indices_for_labels.numel() == 0:
        return 0.0

    label_matches = true_labels.unsqueeze(1) == retrieval_indices_for_labels.unsqueeze(0)
    in_retrieval_set = label_matches.any(dim=1)
    if in_retrieval_set.sum().item() == 0:
        return 0.0

    filtered_pred = pred_embeddings[in_retrieval_set]
    filtered_labels = true_labels[in_retrieval_set]
    retrieval_indices_for_vocab = torch.tensor(retrieval_indices, dtype=torch.long, device=vocab_embeddings.device)
    retrieval_embeddings = vocab_embeddings[retrieval_indices_for_vocab].to(pred_embeddings.device)

    pred_norm = F.normalize(filtered_pred, p=2, dim=1)
    retrieval_norm = F.normalize(retrieval_embeddings, p=2, dim=1)
    similarity = torch.matmul(pred_norm, retrieval_norm.T)

    actual_k = min(k, len(retrieval_indices))
    _, top_k_indices = torch.topk(similarity, k=actual_k, dim=1)

    per_class_accuracies = []
    for retrieval_position, class_idx in enumerate(retrieval_indices):
        class_mask = filtered_labels == class_idx
        if class_mask.sum().item() == 0:
            continue
        class_top_k = top_k_indices[class_mask]
        hits = (class_top_k == retrieval_position).any(dim=1)
        per_class_accuracies.append(hits.float().mean().item())

    return sum(per_class_accuracies) / len(per_class_accuracies) if per_class_accuracies else 0.0


def compute_balanced_top_k_accuracy(
    pred_embeddings: torch.Tensor,
    true_labels: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    k: int = 10
) -> float:
    """
    Compute balanced (macro-averaged) top-k retrieval accuracy.

    For each word class in the vocabulary, compute top-k accuracy and then
    macro-average across all classes. This gives equal weight to each class
    regardless of frequency.

    Args:
        pred_embeddings: [N, 1024] predicted word embeddings
        true_labels: [N] ground truth vocabulary indices
        vocab_embeddings: [vocab_size, 1024] T5 embeddings for all vocabulary words
        k: K value for top-k retrieval

    Returns:
        balanced_accuracy: Macro-averaged top-k accuracy across all vocabulary words
    """
    # Compute cosine similarity
    pred_norm = F.normalize(pred_embeddings, p=2, dim=1)
    vocab_norm = F.normalize(vocab_embeddings, p=2, dim=1)
    similarity = torch.matmul(pred_norm, vocab_norm.T)  # [N, vocab_size]

    _, top_k_indices = torch.topk(similarity, k=k, dim=1)  # [N, k]

    # Compute per-class accuracy
    vocab_size = vocab_embeddings.shape[0]
    per_class_accuracies = []

    for class_idx in range(vocab_size):
        # Get samples for this class
        class_mask = (true_labels == class_idx)
        n_samples = class_mask.sum().item()

        if n_samples == 0:
            # No samples for this class, skip
            continue

        # Check if true label is in top-k for this class
        class_top_k = top_k_indices[class_mask]  # [n_samples, k]
        hits = (class_top_k == class_idx).any(dim=1)
        class_acc = hits.float().mean().item()

        per_class_accuracies.append(class_acc)

    # Macro-average across classes
    balanced_accuracy = sum(per_class_accuracies) / len(per_class_accuracies) if per_class_accuracies else 0.0

    return balanced_accuracy


def compute_embedding_metrics(
    pred_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor
) -> Dict[str, float]:
    """
    Compute embedding quality metrics.

    Args:
        pred_embeddings: [N, 1024] predicted embeddings
        target_embeddings: [N, 1024] target T5 embeddings

    Returns:
        metrics: Dict with cosine similarity and norm statistics
    """
    # Cosine similarity with target
    cos_sim = F.cosine_similarity(pred_embeddings, target_embeddings, dim=1)

    # Embedding norms
    pred_norms = torch.norm(pred_embeddings, p=2, dim=1)
    target_norms = torch.norm(target_embeddings, p=2, dim=1)

    return {
        'mean_cosine_similarity': cos_sim.mean().item(),
        'std_cosine_similarity': cos_sim.std().item(),
        'mean_pred_norm': pred_norms.mean().item(),
        'std_pred_norm': pred_norms.std().item(),
        'mean_target_norm': target_norms.mean().item(),
    }


# ============================================================================
# Training and Evaluation
# ============================================================================

def training_step(
    batch: Dict[str, Any],
    criss_cross_model: CrissCrossTransformerModule,
    word_mlp: CrissCrossWordEmbeddingExtractor,
    vocab_embeddings: torch.Tensor,
    criterion: SigLipLoss,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Training step handling 10 words per 30s sample.

    Args:
        batch: Dictionary from word_level_collate_fn
        criss_cross_model: CrissCross transformer
        word_mlp: Word embedding MLP
        vocab_embeddings: [vocab_size, 1024] T5 embeddings
        criterion: SigLIP loss function
        device: Device to run on

    Returns:
        loss: Scalar loss value
        word_embeddings: [B*10, 1024] predicted embeddings
        target_embeddings: [B*10, 1024] target embeddings
    """
    meg = batch['meg'].to(device)  # [B, C, 7500]
    word_labels = batch['word_labels'].to(device)  # [B*10]
    subsegment_info = batch['subsegment_info']
    sensor_xyzdir = batch['sensor_xyzdir'].to(device)
    sensor_xyz = sensor_xyzdir[..., :3]
    sensor_abc = sensor_xyzdir[..., 3:]
    sensor_types = batch['sensor_types'].to(device)
    sensor_mask = batch['sensor_mask'].to(device)

    use_cuda_amp = torch.device(device).type == "cuda" and torch.cuda.is_available()
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_cuda_amp
        else nullcontext()
    )

    with autocast_context:
        # 1. Forward pass through CrissCross (no masking for evaluation)
        output = criss_cross_model(
            meg, sensor_xyz, sensor_abc, sensor_types, sensor_mask,
            apply_mask=False
        )
        features = output['features']  # [B, C, 625, 512]

        # 2. Extract word embeddings for all subsegments
        word_embeddings = []

        for info in subsegment_info:
            b_idx = info['batch_idx']
            start_sample = info['start_sample']
            end_sample = info['end_sample']

            # Map to encoded timesteps
            start_t, end_t = map_raw_to_encoded_timesteps(start_sample, end_sample)

            # Extract features for this word
            word_features = features[b_idx, :, start_t:end_t, :]  # [C, T_subseg, 512]

            # Pass through word MLP
            word_emb = word_mlp(word_features, subject_idx=info.get('subject_idx', -1))  # [1024]
            word_embeddings.append(word_emb)

        word_embeddings = torch.stack(word_embeddings)  # [B*10, 1024]

        # 3. Get target embeddings
        # Index on CPU, then move to device
        target_embeddings = vocab_embeddings[word_labels.cpu()].to(device)  # [B*10, 1024]

        # 4. Compute SigLIP loss
        loss = criterion(word_embeddings, target_embeddings, reweigh_positives=True)

    return loss, word_embeddings.float(), target_embeddings.float()


def evaluate_epoch(
    criss_cross_model: CrissCrossTransformerModule,
    word_mlp: CrissCrossWordEmbeddingExtractor,
    dataloader: DataLoader,
    vocab_embeddings: torch.Tensor,
    criterion: SigLipLoss,
    device: str,
    retrieval_set_sizes: List[int] = [50, 250],
    k: int = 10,
    k_values: Optional[List[int]] = None,
    named_retrieval_sets: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, float]:
    """
    Evaluate on validation or test set.

    Computes top-k retrieval accuracy for each retrieval set size.
    For each retrieval set size, only samples with labels in that set are evaluated,
    and retrieval is performed against those embeddings.

    Args:
        criss_cross_model: CrissCross transformer
        word_mlp: Word embedding MLP
        dataloader: DataLoader for evaluation
        vocab_embeddings: [vocab_size, 1024] T5 embeddings (ordered by frequency)
        criterion: SigLIP loss function
        device: Device to run on
        retrieval_set_sizes: List of retrieval set sizes to evaluate (e.g., [50, 250])
        k: K value for top-k accuracy when k_values is not provided
        k_values: Optional list of K values to compute simultaneously
        named_retrieval_sets: Optional named vocab-index retrieval sets

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    named_retrieval_sets = named_retrieval_sets or {}
    k_values = [int(value) for value in (k_values if k_values is not None else [k])]
    criss_cross_model.eval()
    word_mlp.eval()

    all_losses = []
    all_pred_embeddings = []
    all_target_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Skip batches with no valid words
            if len(batch['word_labels']) == 0:
                continue

            loss, pred_embs, target_embs = training_step(
                batch, criss_cross_model, word_mlp,
                vocab_embeddings, criterion, device
            )

            all_losses.append(loss.item())
            all_pred_embeddings.append(pred_embs.cpu())
            all_target_embeddings.append(target_embs.cpu())
            all_labels.append(batch['word_labels'])

    if len(all_losses) == 0:
        metrics = {'loss': 0.0}
        total_samples = 0
        for eval_k in k_values:
            for retrieval_size in retrieval_set_sizes:
                metrics[f'top{eval_k}_accuracy_retrieval{retrieval_size}'] = 0.0
                metrics[f'n_samples_retrieval{retrieval_size}'] = 0
                metrics[f'n_skipped_retrieval{retrieval_size}'] = total_samples
                metrics[f'balanced_top{eval_k}_accuracy_retrieval{retrieval_size}'] = 0.0
            for retrieval_name in named_retrieval_sets:
                metrics[f'top{eval_k}_accuracy_{retrieval_name}'] = 0.0
                metrics[f'n_samples_{retrieval_name}'] = 0
                metrics[f'n_skipped_{retrieval_name}'] = total_samples
                metrics[f'balanced_top{eval_k}_accuracy_{retrieval_name}'] = 0.0
        metrics.update({
            'mean_cosine_similarity': 0.0,
            'std_cosine_similarity': 0.0,
            'mean_pred_norm': 0.0,
            'std_pred_norm': 0.0,
            'mean_target_norm': 0.0,
            'std_target_norm': 0.0,
        })
        return metrics

    # Aggregate results
    all_pred_embeddings = torch.cat(all_pred_embeddings, dim=0)
    all_target_embeddings = torch.cat(all_target_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute metrics
    metrics = {}
    metrics['loss'] = sum(all_losses) / len(all_losses)

    # Compute top-k accuracy for each requested K and retrieval set.
    for eval_k in k_values:
        for retrieval_size in retrieval_set_sizes:
            retrieval_metrics = compute_top_k_accuracy_with_retrieval_set(
                all_pred_embeddings, all_labels, vocab_embeddings,
                retrieval_set_size=retrieval_size, k=eval_k
            )
            metrics.update(retrieval_metrics)

            balanced_acc = compute_balanced_top_k_accuracy_with_retrieval_set(
                all_pred_embeddings, all_labels, vocab_embeddings,
                retrieval_set_size=retrieval_size, k=eval_k
            )
            metrics[f'balanced_top{eval_k}_accuracy_retrieval{retrieval_size}'] = balanced_acc

        for retrieval_name, retrieval_indices in named_retrieval_sets.items():
            named_metrics = compute_top_k_accuracy_with_named_retrieval_set(
                all_pred_embeddings, all_labels, vocab_embeddings,
                retrieval_indices=retrieval_indices,
                retrieval_name=retrieval_name,
                k=eval_k,
            )
            metrics.update(named_metrics)
            balanced_acc = compute_balanced_top_k_accuracy_with_named_retrieval_set(
                all_pred_embeddings, all_labels, vocab_embeddings,
                retrieval_indices=retrieval_indices,
                k=eval_k,
            )
            metrics[f'balanced_top{eval_k}_accuracy_{retrieval_name}'] = balanced_acc

    # Embedding quality (computed on all samples)
    emb_metrics = compute_embedding_metrics(all_pred_embeddings, all_target_embeddings)
    metrics.update(emb_metrics)

    return metrics


def _subset_name_for_dataset_recording(dataset: Any, rec_idx: int) -> str:
    if hasattr(dataset, "_subset_name_for_recording") and hasattr(dataset, "recordings"):
        return dataset._subset_name_for_recording(dataset.recordings[rec_idx])
    if hasattr(dataset, "recordings"):
        return str(dataset.recordings[rec_idx].get("task", "unknown"))
    return "unknown"


def _iter_dataset_word_metadata(dataset: Any, split_name: str):
    if not hasattr(dataset, "word_groups") or not hasattr(dataset, "recordings"):
        return

    for rec_idx, groups in enumerate(dataset.word_groups):
        rec = dataset.recordings[rec_idx]
        subset_name = _subset_name_for_dataset_recording(dataset, rec_idx)
        for group in groups:
            for word_info in group:
                yield {
                    "split": split_name,
                    "subset": subset_name,
                    "task": rec.get("task", ""),
                    "subject": rec.get("subject", ""),
                    "session": rec.get("session", ""),
                    "target_word": word_info.get("word", ""),
                    "wordidx": int(word_info.get("wordidx", -1)),
                    "sentenceidx": int(word_info.get("sentenceidx", -1)),
                }


def save_target_embeddings_npz(save_dir: Path, vocab: List[str], vocab_embeddings: torch.Tensor) -> None:
    output_path = save_dir / "target_embeddings.npz"
    np.savez(
        output_path,
        words=np.asarray(vocab, dtype=str),
        target_embeddings=vocab_embeddings.detach().cpu().numpy().astype(np.float32),
    )
    logger.info(f"Saved target embeddings artifact: {output_path}")


def save_val_test_word_counts_npz(save_dir: Path, val_dataset: Any, test_dataset: Any) -> None:
    counts = Counter()
    for row in _iter_dataset_word_metadata(val_dataset, "val") or []:
        counts[(row["split"], row["subset"], row["target_word"])] += 1
    for row in _iter_dataset_word_metadata(test_dataset, "test") or []:
        counts[(row["split"], row["subset"], row["target_word"])] += 1

    rows = sorted(counts.items(), key=lambda item: item[0])
    output_path = save_dir / "val_test_word_counts.npz"
    np.savez(
        output_path,
        split=np.asarray([key[0] for key, _ in rows], dtype=str),
        subset=np.asarray([key[1] for key, _ in rows], dtype=str),
        word=np.asarray([key[2] for key, _ in rows], dtype=str),
        count=np.asarray([count for _, count in rows], dtype=np.int64),
    )
    logger.info(f"Saved val/test word-count artifact: {output_path}")


def write_prediction_embeddings_npz(
    output_path: Path,
    metadata_rows: List[Dict[str, Any]],
    predicted_embeddings: np.ndarray,
) -> None:
    if predicted_embeddings.shape[0] != len(metadata_rows):
        raise ValueError(
            f"Prediction artifact row mismatch: {predicted_embeddings.shape[0]} embeddings "
            f"for {len(metadata_rows)} metadata rows"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        task=np.asarray([row.get('task', '') for row in metadata_rows], dtype=str),
        subject=np.asarray([row.get('subject', '') for row in metadata_rows], dtype=str),
        session=np.asarray([row.get('session', '') for row in metadata_rows], dtype=str),
        target_word=np.asarray([row.get('target_word', '') for row in metadata_rows], dtype=str),
        wordidx=np.asarray([row.get('wordidx', -1) for row in metadata_rows], dtype=np.int64),
        sentenceidx=np.asarray([row.get('sentenceidx', -1) for row in metadata_rows], dtype=np.int64),
        word_label=np.asarray([row.get('word_label', -1) for row in metadata_rows], dtype=np.int64),
        subject_idx=np.asarray([row.get('subject_idx', -1) for row in metadata_rows], dtype=np.int64),
        predicted_embeddings=predicted_embeddings.astype(np.float32, copy=False),
    )


def save_prediction_embeddings_npz(
    output_path: Path,
    criss_cross_model: CrissCrossTransformerModule,
    word_mlp: CrissCrossWordEmbeddingExtractor,
    dataloader: DataLoader,
    vocab_embeddings: torch.Tensor,
    criterion: SigLipLoss,
    device: str,
) -> None:
    criss_cross_model.eval()
    word_mlp.eval()

    embeddings = []
    metadata_rows = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Exporting {output_path.name}"):
            if len(batch['word_labels']) == 0:
                continue
            _loss, pred_embs, _target_embs = training_step(
                batch, criss_cross_model, word_mlp,
                vocab_embeddings, criterion, device
            )
            embeddings.append(pred_embs.detach().cpu())
            metadata_rows.extend(batch.get('word_metadata', []))

    if embeddings:
        predicted_embeddings = torch.cat(embeddings, dim=0).numpy().astype(np.float32)
    else:
        predicted_embeddings = np.empty((0, 0), dtype=np.float32)

    write_prediction_embeddings_npz(output_path, metadata_rows, predicted_embeddings)
    logger.info(f"Saved prediction embeddings artifact: {output_path}")


def train_and_evaluate(
    criss_cross_model: CrissCrossTransformerModule,
    word_mlp: CrissCrossWordEmbeddingExtractor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    vocab_embeddings: torch.Tensor,
    cfg: DictConfig,
    device: str,
    val_subset_loaders: Optional[Dict[str, DataLoader]] = None,
    test_subset_loaders: Optional[Dict[str, DataLoader]] = None,
    named_retrieval_sets: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, float]:
    """
    Main training and evaluation loop.

    Args:
        criss_cross_model: CrissCross transformer
        word_mlp: Word embedding MLP
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        vocab_embeddings: [vocab_size, 1024] T5 embeddings
        cfg: Hydra configuration
        device: Device to run on
        val_subset_loaders: Optional validation loaders for per-subset metrics
        test_subset_loaders: Optional test loaders for per-subset metrics
        named_retrieval_sets: Optional named vocab-index retrieval sets

    Returns:
        test_metrics: Final test set metrics
    """
    val_subset_loaders = val_subset_loaders or {}
    test_subset_loaders = test_subset_loaders or {}
    named_retrieval_sets = named_retrieval_sets or {}

    # Setup optimizer with mode-appropriate learning rates
    if cfg.model.train_from_scratch:
        # From scratch: use same LR for both components
        logger.info(f"Optimizer: Using from_scratch_lr={cfg.training.from_scratch_lr} for all parameters")
        params = [
            {'params': criss_cross_model.parameters(), 'lr': cfg.training.from_scratch_lr},
            {'params': word_mlp.parameters(), 'lr': cfg.training.from_scratch_lr}
        ]
    else:
        # Fine-tuning: use differential learning rates
        logger.info(f"Optimizer: criss_cross_lr={cfg.training.criss_cross_lr}, word_mlp_lr={cfg.training.word_mlp_lr}")
        params = [
            {'params': criss_cross_model.parameters(), 'lr': cfg.training.criss_cross_lr},
            {'params': word_mlp.parameters(), 'lr': cfg.training.word_mlp_lr}
        ]

    optimizer = AdamW(params, weight_decay=cfg.training.weight_decay)

    # Setup scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
    )

    # Setup loss
    criterion = SigLipLoss(
        norm_kind=cfg.loss.norm_kind,
        temperature=cfg.loss.temperature,
        bias=cfg.loss.bias,
        reduction=cfg.loss.reduction
    ).to(device)

    # Training loop
    best_val_top10_acc = 0.0
    patience_counter = 0
    best_test_metrics_at_best_val = {}  # Track test metrics at best validation
    best_val_epoch = 0  # Track which epoch had best validation
    start_epoch = 0

    # Resume from checkpoint if specified
    resume_checkpoint = cfg.training.get('resume_checkpoint', None)
    if resume_checkpoint and Path(resume_checkpoint).exists():
        logger.info(f"\nResuming from checkpoint: {resume_checkpoint}")
        resume_ckpt = torch.load(resume_checkpoint, map_location=device)

        # Load model states
        _load_model_state_dict(criss_cross_model, resume_ckpt['criss_cross_state_dict'])
        word_mlp.load_state_dict(resume_ckpt['word_mlp_state_dict'])

        # Load optimizer state
        if 'optimizer_state_dict' in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])

        # Load scheduler state
        if 'scheduler_state_dict' in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])

        # Restore training state
        start_epoch = resume_ckpt.get('epoch', 0) + 1
        best_val_top10_acc = resume_ckpt.get('best_val_top10_acc', 0.0)
        patience_counter = resume_ckpt.get('patience_counter', 0)
        best_val_epoch = resume_ckpt.get('best_val_epoch', 0)
        best_test_metrics_at_best_val = resume_ckpt.get('best_test_metrics_at_best_val', {})

        logger.info(f"  Resumed from epoch {start_epoch}")
        logger.info(f"  Best val acc so far: {best_val_top10_acc:.4f} (epoch {best_val_epoch})")
        logger.info(f"  Patience counter: {patience_counter}")

    for epoch in range(start_epoch, cfg.training.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{cfg.training.num_epochs}")
        eval_k_values = _get_eval_k_values(cfg.evaluation)
        primary_k = int(cfg.evaluation.get('primary_k', cfg.evaluation.k))

        # Training
        criss_cross_model.train()
        word_mlp.train()

        train_losses = []
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            # Skip batches with no valid words
            if len(batch['word_labels']) == 0:
                continue

            optimizer.zero_grad()

            loss, pred_embs, target_embs = training_step(
                batch, criss_cross_model, word_mlp,
                vocab_embeddings, criterion, device
            )

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(criss_cross_model.parameters()) + list(word_mlp.parameters()),
                cfg.training.gradient_clip_val
            )

            optimizer.step()

            train_losses.append(loss.item())

            # Log every N steps
            if batch_idx % cfg.logging.log_every_n_steps == 0:
                wandb.log({
                    'train/loss_step': loss.item(),
                    'train/step': epoch * len(train_loader) + batch_idx
                })

        # Compute epoch metrics
        train_loss = sum(train_losses) / len(train_losses)
        logger.info(f"  Train loss: {train_loss:.4f}")

        # Validation
        val_metrics = evaluate_epoch(
            criss_cross_model, word_mlp, val_loader,
            vocab_embeddings, criterion, device,
            retrieval_set_sizes=cfg.evaluation.retrieval_set_sizes,
            k=cfg.evaluation.k,
            k_values=eval_k_values,
            named_retrieval_sets=named_retrieval_sets,
        )

        # Also evaluate on test set to understand dynamics (but don't use for early stopping)
        test_metrics = evaluate_epoch(
            criss_cross_model, word_mlp, test_loader,
            vocab_embeddings, criterion, device,
            retrieval_set_sizes=cfg.evaluation.retrieval_set_sizes,
            k=cfg.evaluation.k,
            k_values=eval_k_values,
            named_retrieval_sets=named_retrieval_sets,
        )

        val_subset_metrics = {}
        for subset_name, subset_loader in val_subset_loaders.items():
            val_subset_metrics[subset_name] = evaluate_epoch(
                criss_cross_model, word_mlp, subset_loader,
                vocab_embeddings, criterion, device,
                retrieval_set_sizes=cfg.evaluation.retrieval_set_sizes,
                k=cfg.evaluation.k,
                k_values=eval_k_values,
                named_retrieval_sets=named_retrieval_sets,
            )

        test_subset_metrics = {}
        for subset_name, subset_loader in test_subset_loaders.items():
            test_subset_metrics[subset_name] = evaluate_epoch(
                criss_cross_model, word_mlp, subset_loader,
                vocab_embeddings, criterion, device,
                retrieval_set_sizes=cfg.evaluation.retrieval_set_sizes,
                k=cfg.evaluation.k,
                k_values=eval_k_values,
                named_retrieval_sets=named_retrieval_sets,
            )

        # Get primary retrieval set size for early stopping (largest in list)
        primary_retrieval_size = cfg.evaluation.retrieval_set_sizes[-1]

        logger.info(f"  Val loss: {val_metrics['loss']:.4f}")
        for eval_k in eval_k_values:
            for ret_size in cfg.evaluation.retrieval_set_sizes:
                val_acc = val_metrics.get(f'top{eval_k}_accuracy_retrieval{ret_size}', 0)
                val_balanced = val_metrics.get(f'balanced_top{eval_k}_accuracy_retrieval{ret_size}', 0)
                val_n = val_metrics.get(f'n_samples_retrieval{ret_size}', 0)
                test_acc = test_metrics.get(f'top{eval_k}_accuracy_retrieval{ret_size}', 0)
                test_balanced = test_metrics.get(f'balanced_top{eval_k}_accuracy_retrieval{ret_size}', 0)
                logger.info(f"  [Retrieval {ret_size}] Val top-{eval_k}: {val_acc:.4f}, balanced: {val_balanced:.4f} (n={val_n})")
                logger.info(f"  [Retrieval {ret_size}] Test top-{eval_k}: {test_acc:.4f}, balanced: {test_balanced:.4f}")
                for subset_name, metrics in val_subset_metrics.items():
                    subset_acc = metrics.get(f'top{eval_k}_accuracy_retrieval{ret_size}', 0)
                    subset_balanced = metrics.get(f'balanced_top{eval_k}_accuracy_retrieval{ret_size}', 0)
                    subset_n = metrics.get(f'n_samples_retrieval{ret_size}', 0)
                    logger.info(
                        f"  [Retrieval {ret_size}] Val/{subset_name} top-{eval_k}: "
                        f"{subset_acc:.4f}, balanced: {subset_balanced:.4f} (n={subset_n})"
                    )
                for subset_name, metrics in test_subset_metrics.items():
                    subset_acc = metrics.get(f'top{eval_k}_accuracy_retrieval{ret_size}', 0)
                    subset_balanced = metrics.get(f'balanced_top{eval_k}_accuracy_retrieval{ret_size}', 0)
                    subset_n = metrics.get(f'n_samples_retrieval{ret_size}', 0)
                    logger.info(
                        f"  [Retrieval {ret_size}] Test/{subset_name} top-{eval_k}: "
                        f"{subset_acc:.4f}, balanced: {subset_balanced:.4f} (n={subset_n})"
                    )

            for retrieval_name in named_retrieval_sets:
                val_acc = val_metrics.get(f'top{eval_k}_accuracy_{retrieval_name}', 0)
                val_balanced = val_metrics.get(f'balanced_top{eval_k}_accuracy_{retrieval_name}', 0)
                val_n = val_metrics.get(f'n_samples_{retrieval_name}', 0)
                test_acc = test_metrics.get(f'top{eval_k}_accuracy_{retrieval_name}', 0)
                test_balanced = test_metrics.get(f'balanced_top{eval_k}_accuracy_{retrieval_name}', 0)
                test_n = test_metrics.get(f'n_samples_{retrieval_name}', 0)
                logger.info(
                    f"  [{retrieval_name}] Val top-{eval_k}: {val_acc:.4f}, "
                    f"balanced: {val_balanced:.4f} (n={val_n})"
                )
                logger.info(
                    f"  [{retrieval_name}] Test top-{eval_k}: {test_acc:.4f}, "
                    f"balanced: {test_balanced:.4f} (n={test_n})"
                )

        # Log to WandB
        log_dict = {
            'epoch': epoch + 1,
            'train/loss': train_loss,
            **{f'val/{metric_k}': v for metric_k, v in val_metrics.items()},
            **{f'test_during_train/{metric_k}': v for metric_k, v in test_metrics.items()},
            **{
                f'val_by_subset/{subset_name}/{metric_k}': v
                for subset_name, metrics in val_subset_metrics.items()
                for metric_k, v in metrics.items()
            },
            **{
                f'test_by_subset/{subset_name}/{metric_k}': v
                for subset_name, metrics in test_subset_metrics.items()
                for metric_k, v in metrics.items()
            },
        }

        # Log best test metrics at best validation (tracks test performance at best val checkpoint so far)
        if best_test_metrics_at_best_val:
            log_dict.update({f'test_at_best_val/{metric_k}': v for metric_k, v in best_test_metrics_at_best_val.items()})
            log_dict['test_at_best_val/best_val_epoch'] = best_val_epoch

        wandb.log(log_dict)

        # Use primary retrieval set's balanced accuracy for early stopping
        primary_metric_key = f'balanced_top{primary_k}_accuracy_retrieval{primary_retrieval_size}'
        val_primary_acc = val_metrics.get(primary_metric_key, 0)

        # Scheduler step
        scheduler.step(val_primary_acc)

        # Early stopping and checkpointing
        if val_primary_acc > best_val_top10_acc + cfg.training.min_delta:
            best_val_top10_acc = val_primary_acc
            patience_counter = 0
            best_test_metrics_at_best_val = test_metrics.copy()  # Update best test metrics
            best_val_epoch = epoch + 1  # Track which epoch had best validation (1-indexed)

            # Save best model (include test_metrics for comparison)
            checkpoint_path = Path(cfg.logging.save_dir) / 'checkpoint_best.pt'
            torch.save({
                'epoch': epoch,
                'criss_cross_state_dict': _model_state_dict(criss_cross_model),
                'word_mlp_state_dict': word_mlp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'test_metrics_at_best_val': test_metrics,  # Test metrics at this epoch
                'val_subset_metrics': val_subset_metrics,
                'test_subset_metrics_at_best_val': test_subset_metrics,
                'best_val_top10_acc': best_val_top10_acc,
                'patience_counter': patience_counter,
                'best_val_epoch': best_val_epoch,
                'best_test_metrics_at_best_val': best_test_metrics_at_best_val,
                'config': OmegaConf.to_container(cfg, resolve=True)
            }, checkpoint_path)
            save_dir = Path(cfg.logging.save_dir)
            save_prediction_embeddings_npz(
                save_dir / "best_val_predictions.npz",
                criss_cross_model, word_mlp, val_loader,
                vocab_embeddings, criterion, device
            )
            save_prediction_embeddings_npz(
                save_dir / "best_test_predictions.npz",
                criss_cross_model, word_mlp, test_loader,
                vocab_embeddings, criterion, device
            )
            test_primary_acc = test_metrics.get(f'top{primary_k}_accuracy_retrieval{primary_retrieval_size}', 0)
            logger.info(f"  Saved best model (val balanced: {best_val_top10_acc:.4f}, test top-{primary_k}@{primary_retrieval_size}: {test_primary_acc:.4f})")
        else:
            patience_counter += 1

        # Save latest checkpoint for resuming (every epoch)
        latest_checkpoint_path = Path(cfg.logging.save_dir) / 'checkpoint_latest.pt'
        torch.save({
            'epoch': epoch,
            'criss_cross_state_dict': _model_state_dict(criss_cross_model),
            'word_mlp_state_dict': word_mlp.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_metrics': val_metrics,
            'val_subset_metrics': val_subset_metrics,
            'test_subset_metrics': test_subset_metrics,
            'best_val_top10_acc': best_val_top10_acc,
            'patience_counter': patience_counter,
            'best_val_epoch': best_val_epoch,
            'best_test_metrics_at_best_val': best_test_metrics_at_best_val,
            'config': OmegaConf.to_container(cfg, resolve=True)
        }, latest_checkpoint_path)

        if patience_counter >= cfg.training.patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    logger.info(
        "\nTraining complete. Skipping end-of-training test evaluation; "
        "test metrics were logged during validation epochs."
    )
    return best_test_metrics_at_best_val


# ============================================================================
# Main Entry Point
# ============================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="eval_criss_cross_word_classification")
def main(cfg: DictConfig):
    """Main entry point for CrissCross word classification evaluation."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Setup output directory
    save_dir = Path(cfg.logging.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup WandB
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb_config['training_mode'] = 'from_scratch' if cfg.model.train_from_scratch else 'fine_tuning'

    wandb.init(
        project=cfg.logging.wandb_project,
        name=cfg.logging.experiment_name + '-scratch' if cfg.model.train_from_scratch else cfg.logging.experiment_name,
        config=wandb_config
    )

    # Set random seed
    pl.seed_everything(cfg.seed, workers=True)

    # 1. Load tokenizer
    tokenizer = load_tokenizer(cfg.model.tokenizer_checkpoint, cfg.device)

    # 2. Load or initialize CrissCross model
    if cfg.model.train_from_scratch:
        logger.info("Training mode: FROM SCRATCH (random initialization)")
        criss_cross_model = initialize_criss_cross_from_scratch(
            cfg.model.criss_cross_checkpoint,  # Still needed for architecture
            tokenizer,
            cfg.device
        )
    else:
        logger.info("Training mode: FINE-TUNING (pretrained checkpoint)")
        criss_cross_model = load_criss_cross_model(
            cfg.model.criss_cross_checkpoint,
            tokenizer,
            cfg.device
        )

    # 3. Build vocabulary - will be done after dataset creation for both split modes
    # Note: Vocabulary now includes ALL words (no top-K filtering)
    # For hashed split, vocabulary will be built after dataset creation
    vocab = None
    word_to_idx = None
    vocab_embeddings = None

    # 5. Create datasets
    logger.info("\nCreating datasets...")

    # Get dataset class and max_channel_dim based on config
    dataset_type = cfg.data.get('dataset_type', 'armeni')
    DatasetClass = get_dataset_class(dataset_type)
    max_channel_dim = cfg.data.get('max_channel_dim', get_default_max_channel_dim(dataset_type))

    logger.info(f"Using dataset type: {dataset_type}")
    logger.info(f"Dataset class: {DatasetClass.__name__}")
    logger.info(f"Max channel dim: {max_channel_dim}")

    tasks = _config_list_or_none(cfg.data.get('tasks', None))
    split_strategy = cfg.data.get('split_strategy', None)

    if split_strategy == "libribrain100":
        logger.info("Using LibriBrain100 fixed split strategy...")

        common_dataset_kwargs = dict(
            data_root=cfg.data.root,
            subjects=_config_list_or_none(cfg.data.get('subjects', None)),
            sessions=_config_list_or_none(cfg.data.get('sessions', None)),
            tasks=tasks,
            segment_length=cfg.data.segment_length,
            subsegment_duration=cfg.data.subsegment_duration,
            words_per_segment=cfg.data.words_per_segment,
            window_onset_offset=cfg.data.window_onset_offset,
            cache_dir=cfg.data.cache_dir,
            l_freq=cfg.data.l_freq,
            h_freq=cfg.data.h_freq,
            target_sfreq=cfg.data.target_sfreq,
            max_channel_dim=max_channel_dim,
            include_competition_serialised=cfg.data.get('include_competition_serialised', True),
        )

        train_dataset_full = DatasetClass(
            split="train",
            allow_incomplete_segments=False,
            **common_dataset_kwargs,
        )

        total_size = len(train_dataset_full)
        sample_size = int(cfg.data.train_pct * total_size)
        remaining_size = total_size - sample_size

        logger.info(f"\nSubsampling LibriBrain100 training data to {cfg.data.train_pct*100}%...")
        logger.info(f"  Original: {total_size} segments")
        logger.info(f"  Subsampled: {sample_size} segments")

        train_dataset, _ = random_split(
            train_dataset_full,
            [sample_size, remaining_size],
            generator=torch.Generator().manual_seed(cfg.seed)
        )

        val_dataset = DatasetClass(
            split="val",
            allow_incomplete_segments=True,
            **common_dataset_kwargs,
        )

        test_dataset = DatasetClass(
            split="test",
            allow_incomplete_segments=True,
            **common_dataset_kwargs,
        )

        logger.info("\nBuilding vocabulary from LibriBrain100 train/val/test datasets (ALL words)...")
        word_counter = Counter()
        for dataset in [train_dataset_full, val_dataset, test_dataset]:
            for word_groups in dataset.word_groups:
                for word_group in word_groups:
                    words = [w['word'] for w in word_group]
                    word_counter.update(words)

        vocab = [word for word, _ in word_counter.most_common()]
        if len(vocab) == 0:
            raise ValueError("LibriBrain100 vocabulary is empty; check split rules and data paths")

        logger.info(f"  Total unique words (vocabulary size): {len(vocab)}")
        logger.info(f"  Most frequent word: '{vocab[0]}' ({word_counter[vocab[0]]} occurrences)")
        logger.info(f"  Least frequent word: '{vocab[-1]}' ({word_counter[vocab[-1]]} occurrences)")

        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        vocab_embeddings = generate_word_embeddings(
            vocab,
            vocab_size=len(vocab),
            layer=cfg.t5.layer,
            cache_dir=cfg.t5.cache_dir,
            device=cfg.device,
            dataset_type=dataset_type
        )

    elif cfg.data.get('use_hashed_split', False):
        logger.info("Using hashed sentence-based split...")

        # Create single dataset with ALL sessions
        full_dataset = DatasetClass(
            data_root=cfg.data.root,
            subjects=cfg.data.subjects,
            sessions=cfg.data.all_sessions,  # Load all sessions
            tasks=tasks,
            segment_length=cfg.data.segment_length,
            subsegment_duration=cfg.data.subsegment_duration,
            words_per_segment=cfg.data.words_per_segment,
            window_onset_offset=cfg.data.window_onset_offset,
            cache_dir=cfg.data.cache_dir,
            l_freq=cfg.data.l_freq,
            h_freq=cfg.data.h_freq,
            target_sfreq=cfg.data.target_sfreq,
            max_channel_dim=max_channel_dim,
            allow_incomplete_segments=True,
        )

        logger.info(f"Total segments across all sessions: {len(full_dataset)}")

        # Compute sentence hash for each segment and assign to splits
        # Optimize: access word_groups directly instead of loading full MEG data
        train_indices = []
        val_indices = []
        test_indices = []
        skipped_incomplete_train = 0
        sentence_counts = {}  # Track sentence occurrences
        idx_to_sentence = {}  # Cache sentences for validation

        logger.info("Assigning segments to splits based on sentence hashes...")
        idx = 0
        for word_groups in full_dataset.word_groups:
            for word_group in word_groups:
                # Extract words directly from word_group without loading MEG data
                words = [w['word'] for w in word_group]
                sentence = " ".join(words)

                # Cache sentence for later validation
                idx_to_sentence[idx] = sentence

                # Count sentence occurrences for statistics
                sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1

                # Assign to split based on hash
                split = hash_sentence_to_split(words, cfg.data.split_ratios, cfg.seed)
                is_complete_group = len(word_group) == cfg.data.words_per_segment

                if split == "train":
                    if is_complete_group:
                        train_indices.append(idx)
                    else:
                        skipped_incomplete_train += 1
                elif split == "val":
                    val_indices.append(idx)
                else:  # test
                    test_indices.append(idx)

                idx += 1

        # Log statistics
        total_sentences = len(sentence_counts)
        duplicate_sentences = sum(1 for count in sentence_counts.values() if count > 1)
        logger.info(f"Unique sentences: {total_sentences}")
        logger.info(f"Sentences appearing multiple times: {duplicate_sentences}")
        logger.info(f"Split sizes before train subsampling:")
        logger.info(f"  Train: {len(train_indices)} segments ({len(train_indices)/len(full_dataset)*100:.1f}%)")
        logger.info(f"  Val: {len(val_indices)} segments ({len(val_indices)/len(full_dataset)*100:.1f}%)")
        logger.info(f"  Test: {len(test_indices)} segments ({len(test_indices)/len(full_dataset)*100:.1f}%)")
        logger.info(f"  Skipped incomplete train segments: {skipped_incomplete_train}")

        # Create subset datasets
        train_dataset_full = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

        # Subsample training data to train_pct
        total_size = len(train_dataset_full)
        sample_size = int(cfg.data.train_pct * total_size)
        remaining_size = total_size - sample_size

        logger.info(f"\nSubsampling training data to {cfg.data.train_pct*100}%...")
        logger.info(f"  Original: {total_size} segments")
        logger.info(f"  Subsampled: {sample_size} segments")

        train_dataset, _ = random_split(
            train_dataset_full,
            [sample_size, remaining_size],
            generator=torch.Generator().manual_seed(cfg.seed)
        )

        # Validate no sentence leakage across splits
        logger.info("\nValidating split integrity...")

        # Reuse cached sentences from hashing step (no need to reconstruct)
        train_sentences = {idx_to_sentence[idx] for idx in train_indices}
        val_sentences = {idx_to_sentence[idx] for idx in val_indices}
        test_sentences = {idx_to_sentence[idx] for idx in test_indices}

        # Check for overlaps
        train_val_overlap = train_sentences & val_sentences
        train_test_overlap = train_sentences & test_sentences
        val_test_overlap = val_sentences & test_sentences

        logger.info(f"Split validation results:")
        logger.info(f"  Unique train sentences: {len(train_sentences)}")
        logger.info(f"  Unique val sentences: {len(val_sentences)}")
        logger.info(f"  Unique test sentences: {len(test_sentences)}")
        logger.info(f"  Train-Val overlap: {len(train_val_overlap)} (should be 0)")
        logger.info(f"  Train-Test overlap: {len(train_test_overlap)} (should be 0)")
        logger.info(f"  Val-Test overlap: {len(val_test_overlap)} (should be 0)")

        if train_val_overlap or train_test_overlap or val_test_overlap:
            raise ValueError("Sentence leakage detected across splits!")

        # Build vocabulary from ENTIRE dataset using ALL unique words
        # Training uses all words (no OOV), evaluation filters by retrieval set
        logger.info("\nBuilding vocabulary from entire dataset (ALL words)...")
        word_counter = Counter()

        # Count words across ALL segments (train + val + test)
        for word_groups in full_dataset.word_groups:
            for word_group in word_groups:
                words = [w['word'] for w in word_group]
                word_counter.update(words)

        # Use ALL words, ordered by frequency (most frequent first for retrieval set indexing)
        vocab = [word for word, _ in word_counter.most_common()]

        logger.info(f"  Total unique words (vocabulary size): {len(vocab)}")
        logger.info(f"  Most frequent word: '{vocab[0]}' ({word_counter[vocab[0]]} occurrences)")
        logger.info(f"  Least frequent word: '{vocab[-1]}' ({word_counter[vocab[-1]]} occurrences)")

        # Create word-to-index mapping
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}

        # Generate T5 embeddings
        vocab_embeddings = generate_word_embeddings(
            vocab,
            vocab_size=len(vocab),
            layer=cfg.t5.layer,
            cache_dir=cfg.t5.cache_dir,
            device=cfg.device,
            dataset_type=dataset_type
        )

    else:
        # Use existing session-based temporal split
        logger.info("Using session-based temporal split...")

        train_dataset = DatasetClass(
            data_root=cfg.data.root,
            subjects=cfg.data.subjects,
            sessions=cfg.data.train_sessions,
            tasks=tasks,
            segment_length=cfg.data.segment_length,
            subsegment_duration=cfg.data.subsegment_duration,
            words_per_segment=cfg.data.words_per_segment,
            window_onset_offset=cfg.data.window_onset_offset,
            cache_dir=cfg.data.cache_dir,
            l_freq=cfg.data.l_freq,
            h_freq=cfg.data.h_freq,
            target_sfreq=cfg.data.target_sfreq,
            max_channel_dim=max_channel_dim,
            allow_incomplete_segments=False,
        )

        print("Original training dataset size:", len(train_dataset))
        print("Sampling", cfg.data.train_pct * 100, "% of training data...")

        total_size = len(train_dataset)
        sample_size = int(cfg.data.train_pct * total_size)
        remaining_size = total_size - sample_size

        print("New training dataset size:", sample_size)

        train_subset, _ = random_split(
            train_dataset,
            [sample_size, remaining_size],
            generator=torch.Generator().manual_seed(cfg.seed)
        )

        train_dataset = train_subset

        val_dataset = DatasetClass(
            data_root=cfg.data.root,
            subjects=cfg.data.subjects,
            sessions=cfg.data.val_sessions,
            tasks=tasks,
            segment_length=cfg.data.segment_length,
            subsegment_duration=cfg.data.subsegment_duration,
            words_per_segment=cfg.data.words_per_segment,
            window_onset_offset=cfg.data.window_onset_offset,
            cache_dir=cfg.data.cache_dir,
            l_freq=cfg.data.l_freq,
            h_freq=cfg.data.h_freq,
            target_sfreq=cfg.data.target_sfreq,
            max_channel_dim=max_channel_dim,
            allow_incomplete_segments=True,
        )

        test_dataset = DatasetClass(
            data_root=cfg.data.root,
            subjects=cfg.data.subjects,
            sessions=cfg.data.test_sessions,
            tasks=tasks,
            segment_length=cfg.data.segment_length,
            subsegment_duration=cfg.data.subsegment_duration,
            words_per_segment=cfg.data.words_per_segment,
            window_onset_offset=cfg.data.window_onset_offset,
            cache_dir=cfg.data.cache_dir,
            l_freq=cfg.data.l_freq,
            h_freq=cfg.data.h_freq,
            target_sfreq=cfg.data.target_sfreq,
            max_channel_dim=max_channel_dim,
            allow_incomplete_segments=True,
        )

        # Build vocabulary from ALL datasets using ALL unique words
        # Training uses all words (no OOV), evaluation filters by retrieval set
        logger.info("\nBuilding vocabulary from all datasets (ALL words)...")
        word_counter = Counter()

        # Get underlying dataset from train_dataset (which is a Subset)
        train_base = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset

        # Count words from all datasets
        for dataset in [train_base, val_dataset, test_dataset]:
            for word_groups in dataset.word_groups:
                for word_group in word_groups:
                    words = [w['word'] for w in word_group]
                    word_counter.update(words)

        # Use ALL words, ordered by frequency (most frequent first for retrieval set indexing)
        vocab = [word for word, _ in word_counter.most_common()]

        logger.info(f"  Total unique words (vocabulary size): {len(vocab)}")
        logger.info(f"  Most frequent word: '{vocab[0]}' ({word_counter[vocab[0]]} occurrences)")
        logger.info(f"  Least frequent word: '{vocab[-1]}' ({word_counter[vocab[-1]]} occurrences)")

        # Create word-to-index mapping
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}

        # Generate T5 embeddings
        vocab_embeddings = generate_word_embeddings(
            vocab,
            vocab_size=len(vocab),
            layer=cfg.t5.layer,
            cache_dir=cfg.t5.cache_dir,
            device=cfg.device,
            dataset_type=dataset_type
        )

    logger.info(f"\nFinal dataset sizes:")
    logger.info(f"  Train: {len(train_dataset)} segments")
    logger.info(f"  Val: {len(val_dataset)} segments")
    logger.info(f"  Test: {len(test_dataset)} segments")

    named_retrieval_sets = _build_named_retrieval_sets(
        cfg.evaluation.get('named_retrieval_sets', None),
        word_to_idx,
    )
    save_target_embeddings_npz(save_dir, vocab, vocab_embeddings)
    save_val_test_word_counts_npz(save_dir, val_dataset, test_dataset)

    use_subject_film = bool(cfg.model.word_mlp.get('use_subject_film', False))
    subject_embedding_dim = int(cfg.model.word_mlp.get('subject_embedding_dim', 64))
    subject_to_idx = build_subject_to_idx(train_dataset) if use_subject_film else None
    if use_subject_film:
        logger.info(f"\nSubject FiLM:")
        logger.info(f"  Train subjects: {len(subject_to_idx)}")
        logger.info(f"  Subject embedding dim: {subject_embedding_dim}")

    # Create collate function
    collate_fn = create_word_level_collate_fn(word_to_idx, subject_to_idx=subject_to_idx)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        collate_fn=collate_fn
    )

    val_subset_loaders = _create_eval_subset_loaders(
        val_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        collate_fn=collate_fn,
    )
    test_subset_loaders = _create_eval_subset_loaders(
        test_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        collate_fn=collate_fn,
    )

    if val_subset_loaders:
        logger.info("  Val subsets:")
        for subset_name, subset_loader in val_subset_loaders.items():
            logger.info(f"    {subset_name}: {len(subset_loader.dataset)} segments")
    if test_subset_loaders:
        logger.info("  Test subsets:")
        for subset_name, subset_loader in test_subset_loaders.items():
            logger.info(f"    {subset_name}: {len(subset_loader.dataset)} segments")

    # 6. Create word embedding MLP
    # Use the max_channel_dim computed earlier (based on dataset_type or config override)
    num_channels = max_channel_dim

    word_mlp = CrissCrossWordEmbeddingExtractor(
        num_channels=num_channels,
        latent_dim=criss_cross_model.latent_dim,
        embed_dim=cfg.model.word_mlp.embed_dim,
        hidden_dim=cfg.model.word_mlp.hidden_dim,
        dropout=cfg.model.word_mlp.dropout,
        use_subject_film=use_subject_film,
        num_subjects=len(subject_to_idx) if subject_to_idx is not None else 0,
        subject_embedding_dim=subject_embedding_dim,
    ).to(cfg.device)

    logger.info(f"\nWord MLP:")
    logger.info(f"  Input dim: {num_channels * criss_cross_model.latent_dim}")
    logger.info(f"  Hidden dim: {cfg.model.word_mlp.hidden_dim}")
    logger.info(f"  Output dim: {cfg.model.word_mlp.embed_dim}")
    logger.info(f"  Subject FiLM: {use_subject_film}")

    # Configure optional memory/speed tradeoffs before optimizer construction.
    if cfg.model.get('use_gradient_checkpointing', False):
        criss_cross_model.enable_gradient_checkpointing()
        logger.info("Enabled gradient checkpointing on criss_cross_model")

    if cfg.training.get('use_torch_compile', False):
        logger.info("Compiling criss_cross_model with torch.compile (first step will be slow)")
        criss_cross_model = torch.compile(criss_cross_model)

    # 7. Train and evaluate
    test_metrics = train_and_evaluate(
        criss_cross_model, word_mlp,
        train_loader, val_loader, test_loader,
        vocab_embeddings, cfg, cfg.device,
        val_subset_loaders=val_subset_loaders,
        test_subset_loaders=test_subset_loaders,
        named_retrieval_sets=named_retrieval_sets,
    )

    # 8. Save the test metrics captured during training at the best validation epoch.
    results_path = save_dir / 'final_results.txt'
    with open(results_path, 'w') as f:
        f.write("CrissCross Word Classification - Test Results at Best Validation Epoch\n")
        f.write("=" * 60 + "\n\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    logger.info(f"\nResults saved to: {results_path}")

    wandb.finish()

    return test_metrics


if __name__ == "__main__":
    main()
