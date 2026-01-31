"""
Word classification evaluation for CrissCrossTransformer using LINEAR PROBE.

This is a linear probe variant of the evaluation script:
- CrissCross backbone is FROZEN (no gradients)
- Only a single linear layer is trained on top of the frozen features

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
    python -m brainstorm.evaluate_criss_cross_word_classification_linear_probe \
        model.criss_cross_checkpoint=path/to/ckpt.ckpt \
        training.batch_size=4 \
        training.num_epochs=50
"""

import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
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

from brainstorm.models.criss_cross_transformer import CrissCrossTransformerModule
from brainstorm.neuro_tokenizers.biocodec.model import BioCodecModel
from brainstorm.data.armeni_word_aligned_dataset import ArmeniWordAlignedDataset
from brainstorm.data.gwilliams_word_aligned_dataset import GwilliamsWordAlignedDataset
from brainstorm.data.libribrain_word_aligned_dataset import LibriBrainWordAlignedDataset
from brainstorm.losses.contrastive import SigLipLoss

logger = logging.getLogger(__name__)


# ============================================================================
# Dataset Factory Functions
# ============================================================================

def get_dataset_class(dataset_type: str):
    """
    Get the appropriate dataset class based on dataset_type.

    Args:
        dataset_type: One of "armeni", "gwilliams", or "libribrain"

    Returns:
        Dataset class to instantiate
    """
    dataset_classes = {
        "armeni": ArmeniWordAlignedDataset,
        "gwilliams": GwilliamsWordAlignedDataset,
        "libribrain": LibriBrainWordAlignedDataset,
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
    """
    defaults = {
        "armeni": 306,
        "gwilliams": 208,
        "libribrain": 306,
    }
    return defaults.get(dataset_type, 306)


# ============================================================================
# Model Components - LINEAR PROBE
# ============================================================================

class CrissCrossLinearProbe(nn.Module):
    """
    Linear probe for extracting word embeddings from CrissCross features.

    Architecture per word subsegment:
    1. Extract features for subsegment time range: [C, ~62-63, 512]
    2. Mean pool over time: [C, 512]
    3. Flatten: [C * 512]
    4. Single linear layer: [C * 512] -> [1024]

    This is a simpler alternative to CrissCrossWordEmbeddingExtractor (MLP).
    """

    def __init__(
        self,
        num_channels: int,
        latent_dim: int = 512,
        embed_dim: int = 1024
    ):
        super().__init__()
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        input_dim = num_channels * latent_dim
        self.linear = nn.Linear(input_dim, embed_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [C, T_subseg, latent_dim] - features for one word subsegment

        Returns:
            embedding: [embed_dim] - predicted T5 embedding
        """
        # Mean pool over time
        pooled = features.mean(dim=1)  # [C, latent_dim]

        # Flatten
        flat = pooled.reshape(-1)  # [C * latent_dim]

        # Linear projection
        embedding = self.linear(flat)  # [embed_dim]

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

def create_word_level_collate_fn(word_to_idx: Dict[str, int]):
    """
    Create a collate function that tracks word labels for SigLIP loss.

    Args:
        word_to_idx: Mapping from word string to vocabulary index (includes ALL words)

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
            - subsegment_info: List of dicts with batch_idx, start, end for N words
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

        for batch_idx, sample in enumerate(batch):
            for subseg_idx, (word, boundary) in enumerate(zip(sample['words'], sample['subsegment_boundaries'])):
                # Include all words - no OOV filtering since vocab contains all words
                if word not in word_to_idx:
                    # This should never happen if vocab is built correctly
                    logger.warning(f"Word '{word}' not in vocabulary - skipping")
                    continue

                word_labels.append(word_to_idx[word])
                subsegment_info.append({
                    'batch_idx': batch_idx,
                    'subseg_idx': subseg_idx,
                    'start_sample': boundary['start_sample'],
                    'end_sample': boundary['end_sample']
                })

        return {
            'meg': meg,
            'word_labels': torch.tensor(word_labels, dtype=torch.long),
            'subsegment_info': subsegment_info,
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
    linear_probe: CrissCrossLinearProbe,
    vocab_embeddings: torch.Tensor,
    criterion: SigLipLoss,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Training step handling 10 words per 30s sample.

    Note: CrissCross is frozen, only linear_probe gradients are computed.

    Args:
        batch: Dictionary from word_level_collate_fn
        criss_cross_model: CrissCross transformer (FROZEN)
        linear_probe: Linear probe layer (TRAINABLE)
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

    # 1. Forward pass through CrissCross (no grad since frozen)
    with torch.no_grad():
        output = criss_cross_model(
            meg, sensor_xyz, sensor_abc, sensor_types, sensor_mask,
            apply_mask=False
        )
        features = output['features']  # [B, C, 625, 512]

    # 2. Extract word embeddings for all subsegments (with grad for linear probe)
    word_embeddings = []

    for info in subsegment_info:
        b_idx = info['batch_idx']
        start_sample = info['start_sample']
        end_sample = info['end_sample']

        # Map to encoded timesteps
        start_t, end_t = map_raw_to_encoded_timesteps(start_sample, end_sample)

        # Extract features for this word
        word_features = features[b_idx, :, start_t:end_t, :]  # [C, T_subseg, 512]

        # Pass through linear probe
        word_emb = linear_probe(word_features)  # [1024]
        word_embeddings.append(word_emb)

    word_embeddings = torch.stack(word_embeddings)  # [B*10, 1024]

    # 3. Get target embeddings
    # Index on CPU, then move to device
    target_embeddings = vocab_embeddings[word_labels.cpu()].to(device)  # [B*10, 1024]

    # 4. Compute SigLIP loss
    loss = criterion(word_embeddings, target_embeddings, reweigh_positives=True)

    return loss, word_embeddings, target_embeddings


def evaluate_epoch(
    criss_cross_model: CrissCrossTransformerModule,
    linear_probe: CrissCrossLinearProbe,
    dataloader: DataLoader,
    vocab_embeddings: torch.Tensor,
    criterion: SigLipLoss,
    device: str,
    retrieval_set_sizes: List[int] = [50, 250],
    k: int = 10
) -> Dict[str, float]:
    """
    Evaluate on validation or test set.

    Computes top-k retrieval accuracy for each retrieval set size.
    For each retrieval set size, only samples with labels in that set are evaluated,
    and retrieval is performed against those embeddings.

    Args:
        criss_cross_model: CrissCross transformer (frozen)
        linear_probe: Linear probe layer
        dataloader: DataLoader for evaluation
        vocab_embeddings: [vocab_size, 1024] T5 embeddings (ordered by frequency)
        criterion: SigLIP loss function
        device: Device to run on
        retrieval_set_sizes: List of retrieval set sizes to evaluate (e.g., [50, 250])
        k: K value for top-k accuracy (default: 10)

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    criss_cross_model.eval()
    linear_probe.eval()

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
                batch, criss_cross_model, linear_probe,
                vocab_embeddings, criterion, device
            )

            all_losses.append(loss.item())
            all_pred_embeddings.append(pred_embs.cpu())
            all_target_embeddings.append(target_embs.cpu())
            all_labels.append(batch['word_labels'])

    # Aggregate results
    all_pred_embeddings = torch.cat(all_pred_embeddings, dim=0)
    all_target_embeddings = torch.cat(all_target_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute metrics
    metrics = {}
    metrics['loss'] = sum(all_losses) / len(all_losses)

    # Compute top-k accuracy for each retrieval set size
    for retrieval_size in retrieval_set_sizes:
        # Top-k accuracy with retrieval set
        retrieval_metrics = compute_top_k_accuracy_with_retrieval_set(
            all_pred_embeddings, all_labels, vocab_embeddings,
            retrieval_set_size=retrieval_size, k=k
        )
        metrics.update(retrieval_metrics)

        # Balanced top-k accuracy with retrieval set
        balanced_acc = compute_balanced_top_k_accuracy_with_retrieval_set(
            all_pred_embeddings, all_labels, vocab_embeddings,
            retrieval_set_size=retrieval_size, k=k
        )
        metrics[f'balanced_top{k}_accuracy_retrieval{retrieval_size}'] = balanced_acc

    # Embedding quality (computed on all samples)
    emb_metrics = compute_embedding_metrics(all_pred_embeddings, all_target_embeddings)
    metrics.update(emb_metrics)

    return metrics


def train_and_evaluate(
    criss_cross_model: CrissCrossTransformerModule,
    linear_probe: CrissCrossLinearProbe,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    vocab_embeddings: torch.Tensor,
    cfg: DictConfig,
    device: str
) -> Dict[str, float]:
    """
    Main training and evaluation loop.

    Note: Only the linear probe is trained. CrissCross backbone is frozen.

    Args:
        criss_cross_model: CrissCross transformer (FROZEN)
        linear_probe: Linear probe layer (TRAINABLE)
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        vocab_embeddings: [vocab_size, 1024] T5 embeddings
        cfg: Hydra configuration
        device: Device to run on

    Returns:
        test_metrics: Final test set metrics
    """
    # Setup optimizer - ONLY linear probe parameters
    logger.info(f"Optimizer: Training only linear probe with lr={cfg.training.probe_lr}")
    optimizer = AdamW(
        linear_probe.parameters(),
        lr=cfg.training.probe_lr,
        weight_decay=cfg.training.weight_decay
    )

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

    for epoch in range(cfg.training.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{cfg.training.num_epochs}")

        # Training - only linear probe, CrissCross stays in eval mode
        linear_probe.train()
        # CrissCross stays in eval mode (frozen)

        train_losses = []
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            # Skip batches with no valid words
            if len(batch['word_labels']) == 0:
                continue

            optimizer.zero_grad()

            loss, pred_embs, target_embs = training_step(
                batch, criss_cross_model, linear_probe,
                vocab_embeddings, criterion, device
            )

            loss.backward()

            # Gradient clipping - only linear probe
            torch.nn.utils.clip_grad_norm_(
                linear_probe.parameters(),
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
            criss_cross_model, linear_probe, val_loader,
            vocab_embeddings, criterion, device,
            retrieval_set_sizes=cfg.evaluation.retrieval_set_sizes,
            k=cfg.evaluation.k
        )

        # Also evaluate on test set to understand dynamics (but don't use for early stopping)
        test_metrics = evaluate_epoch(
            criss_cross_model, linear_probe, test_loader,
            vocab_embeddings, criterion, device,
            retrieval_set_sizes=cfg.evaluation.retrieval_set_sizes,
            k=cfg.evaluation.k
        )

        # Get primary retrieval set size for early stopping (largest in list)
        primary_retrieval_size = cfg.evaluation.retrieval_set_sizes[-1]
        k = cfg.evaluation.k

        logger.info(f"  Val loss: {val_metrics['loss']:.4f}")
        for ret_size in cfg.evaluation.retrieval_set_sizes:
            val_acc = val_metrics.get(f'top{k}_accuracy_retrieval{ret_size}', 0)
            val_balanced = val_metrics.get(f'balanced_top{k}_accuracy_retrieval{ret_size}', 0)
            val_n = val_metrics.get(f'n_samples_retrieval{ret_size}', 0)
            test_acc = test_metrics.get(f'top{k}_accuracy_retrieval{ret_size}', 0)
            test_balanced = test_metrics.get(f'balanced_top{k}_accuracy_retrieval{ret_size}', 0)
            logger.info(f"  [Retrieval {ret_size}] Val top-{k}: {val_acc:.4f}, balanced: {val_balanced:.4f} (n={val_n})")
            logger.info(f"  [Retrieval {ret_size}] Test top-{k}: {test_acc:.4f}, balanced: {test_balanced:.4f}")

        # Log to WandB
        log_dict = {
            'epoch': epoch + 1,
            'train/loss': train_loss,
            **{f'val/{metric_k}': v for metric_k, v in val_metrics.items()},
            **{f'test_during_train/{metric_k}': v for metric_k, v in test_metrics.items()}
        }

        # Log best test metrics at best validation (tracks test performance at best val checkpoint so far)
        if best_test_metrics_at_best_val:
            log_dict.update({f'test_at_best_val/{metric_k}': v for metric_k, v in best_test_metrics_at_best_val.items()})
            log_dict['test_at_best_val/best_val_epoch'] = best_val_epoch

        wandb.log(log_dict)

        # Use primary retrieval set's balanced accuracy for early stopping
        primary_metric_key = f'balanced_top{k}_accuracy_retrieval{primary_retrieval_size}'
        val_primary_acc = val_metrics.get(primary_metric_key, 0)

        # Scheduler step
        scheduler.step(val_primary_acc)

        # Early stopping and checkpointing
        if val_primary_acc > best_val_top10_acc + cfg.training.min_delta:
            best_val_top10_acc = val_primary_acc
            patience_counter = 0
            best_test_metrics_at_best_val = test_metrics.copy()  # Update best test metrics
            best_val_epoch = epoch + 1  # Track which epoch had best validation (1-indexed)

            # Save best model (only linear probe since CrissCross is frozen)
            checkpoint_path = Path(cfg.logging.save_dir) / 'checkpoint_best.pt'
            torch.save({
                'epoch': epoch,
                'linear_probe_state_dict': linear_probe.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'test_metrics_at_best_val': test_metrics,  # Test metrics at this epoch
                'config': OmegaConf.to_container(cfg, resolve=True)
            }, checkpoint_path)
            test_primary_acc = test_metrics.get(f'top{k}_accuracy_retrieval{primary_retrieval_size}', 0)
            logger.info(f"  Saved best model (val balanced: {best_val_top10_acc:.4f}, test top-{k}@{primary_retrieval_size}: {test_primary_acc:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= cfg.training.patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model and test
    logger.info("\nLoading best model for final evaluation...")
    checkpoint_path = Path(cfg.logging.save_dir) / 'checkpoint_best.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    linear_probe.load_state_dict(checkpoint['linear_probe_state_dict'])

    # Get test metrics from when checkpoint was saved (in-memory evaluation)
    test_metrics_at_best_val = checkpoint.get('test_metrics_at_best_val', {})

    # Evaluate again after loading (to detect save/load issues)
    test_metrics_after_load = evaluate_epoch(
        criss_cross_model, linear_probe, test_loader,
        vocab_embeddings, criterion, device,
        retrieval_set_sizes=cfg.evaluation.retrieval_set_sizes,
        k=cfg.evaluation.k
    )

    logger.info("\n=== Final Test Results Comparison ===")
    logger.info("Test metrics at best val epoch (in-memory):")
    for metric_k, v in test_metrics_at_best_val.items():
        logger.info(f"  {metric_k}: {v:.4f}")

    logger.info("\nTest metrics after checkpoint load:")
    for metric_k, v in test_metrics_after_load.items():
        logger.info(f"  {metric_k}: {v:.4f}")

    # Check for discrepancy using primary retrieval set (largest)
    primary_retrieval_size = cfg.evaluation.retrieval_set_sizes[-1]
    eval_k = cfg.evaluation.k
    if test_metrics_at_best_val:
        primary_key = f'top{eval_k}_accuracy_retrieval{primary_retrieval_size}'
        in_mem_acc = test_metrics_at_best_val.get(primary_key, 0)
        loaded_acc = test_metrics_after_load.get(primary_key, 0)
        diff = abs(in_mem_acc - loaded_acc)
        if diff > 0.01:
            logger.warning(f"\n  Discrepancy detected! In-memory: {in_mem_acc:.4f}, After load: {loaded_acc:.4f}, Diff: {diff:.4f}")
        else:
            logger.info(f"\n  No significant discrepancy (diff: {diff:.4f})")

    wandb.log({
        **{f'test_in_memory/{k}': v for k, v in test_metrics_at_best_val.items()},
        **{f'test_after_load/{k}': v for k, v in test_metrics_after_load.items()}
    })

    return test_metrics_after_load


# ============================================================================
# Main Entry Point
# ============================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="eval_criss_cross_word_classification_linear_probe")
def main(cfg: DictConfig):
    """Main entry point for CrissCross word classification evaluation with LINEAR PROBE."""

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
    wandb_config['training_mode'] = 'linear_probe'

    wandb.init(
        project=cfg.logging.wandb_project,
        name=cfg.logging.experiment_name,
        config=wandb_config
    )

    # Set random seed
    pl.seed_everything(cfg.seed, workers=True)

    # 1. Load tokenizer
    tokenizer = load_tokenizer(cfg.model.tokenizer_checkpoint, cfg.device)

    # 2. Load CrissCross model and FREEZE it
    logger.info("Training mode: LINEAR PROBE (frozen backbone)")
    criss_cross_model = load_criss_cross_model(
        cfg.model.criss_cross_checkpoint,
        tokenizer,
        cfg.device
    )

    # Freeze ALL CrissCross parameters
    logger.info("Freezing CrissCross backbone...")
    for param in criss_cross_model.parameters():
        param.requires_grad = False
    criss_cross_model.eval()  # Keep in eval mode throughout

    # Count frozen parameters
    total_params = sum(p.numel() for p in criss_cross_model.parameters())
    logger.info(f"  Frozen CrissCross parameters: {total_params:,}")

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

    if cfg.data.get('use_hashed_split', False):
        logger.info("Using hashed sentence-based split...")

        # Create single dataset with ALL sessions
        full_dataset = DatasetClass(
            data_root=cfg.data.root,
            subjects=cfg.data.subjects,
            sessions=cfg.data.all_sessions,  # Load all sessions
            tasks=list(cfg.data.tasks) if cfg.data.tasks is not None and hasattr(cfg.data.tasks, '__iter__') and not isinstance(cfg.data.tasks, str) else ([cfg.data.tasks] if cfg.data.tasks is not None else None),
            segment_length=cfg.data.segment_length,
            subsegment_duration=cfg.data.subsegment_duration,
            words_per_segment=cfg.data.words_per_segment,
            window_onset_offset=cfg.data.window_onset_offset,
            cache_dir=cfg.data.cache_dir,
            l_freq=cfg.data.l_freq,
            h_freq=cfg.data.h_freq,
            target_sfreq=cfg.data.target_sfreq,
            max_channel_dim=max_channel_dim
        )

        logger.info(f"Total segments across all sessions: {len(full_dataset)}")

        # Compute sentence hash for each segment and assign to splits
        # Optimize: access word_groups directly instead of loading full MEG data
        train_indices = []
        val_indices = []
        test_indices = []
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

                if split == "train":
                    train_indices.append(idx)
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
            tasks=list(cfg.data.tasks) if cfg.data.tasks is not None and hasattr(cfg.data.tasks, '__iter__') and not isinstance(cfg.data.tasks, str) else ([cfg.data.tasks] if cfg.data.tasks is not None else None),
            segment_length=cfg.data.segment_length,
            subsegment_duration=cfg.data.subsegment_duration,
            words_per_segment=cfg.data.words_per_segment,
            window_onset_offset=cfg.data.window_onset_offset,
            cache_dir=cfg.data.cache_dir,
            l_freq=cfg.data.l_freq,
            h_freq=cfg.data.h_freq,
            target_sfreq=cfg.data.target_sfreq,
            max_channel_dim=max_channel_dim
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
            tasks=list(cfg.data.tasks) if cfg.data.tasks is not None and hasattr(cfg.data.tasks, '__iter__') and not isinstance(cfg.data.tasks, str) else ([cfg.data.tasks] if cfg.data.tasks is not None else None),
            segment_length=cfg.data.segment_length,
            subsegment_duration=cfg.data.subsegment_duration,
            words_per_segment=cfg.data.words_per_segment,
            window_onset_offset=cfg.data.window_onset_offset,
            cache_dir=cfg.data.cache_dir,
            l_freq=cfg.data.l_freq,
            h_freq=cfg.data.h_freq,
            target_sfreq=cfg.data.target_sfreq,
            max_channel_dim=max_channel_dim
        )

        test_dataset = DatasetClass(
            data_root=cfg.data.root,
            subjects=cfg.data.subjects,
            sessions=cfg.data.test_sessions,
            tasks=list(cfg.data.tasks) if cfg.data.tasks is not None and hasattr(cfg.data.tasks, '__iter__') and not isinstance(cfg.data.tasks, str) else ([cfg.data.tasks] if cfg.data.tasks is not None else None),
            segment_length=cfg.data.segment_length,
            subsegment_duration=cfg.data.subsegment_duration,
            words_per_segment=cfg.data.words_per_segment,
            window_onset_offset=cfg.data.window_onset_offset,
            cache_dir=cfg.data.cache_dir,
            l_freq=cfg.data.l_freq,
            h_freq=cfg.data.h_freq,
            target_sfreq=cfg.data.target_sfreq,
            max_channel_dim=max_channel_dim
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

    # Create collate function
    collate_fn = create_word_level_collate_fn(word_to_idx)

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

    # 6. Create linear probe
    # Use the max_channel_dim computed earlier (based on dataset_type or config override)
    num_channels = max_channel_dim

    linear_probe = CrissCrossLinearProbe(
        num_channels=num_channels,
        latent_dim=criss_cross_model.latent_dim,
        embed_dim=cfg.model.linear_probe.embed_dim
    ).to(cfg.device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in linear_probe.parameters() if p.requires_grad)
    logger.info(f"\nLinear Probe:")
    logger.info(f"  Input dim: {num_channels * criss_cross_model.latent_dim}")
    logger.info(f"  Output dim: {cfg.model.linear_probe.embed_dim}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    # 7. Train and evaluate
    test_metrics = train_and_evaluate(
        criss_cross_model, linear_probe,
        train_loader, val_loader, test_loader,
        vocab_embeddings, cfg, cfg.device
    )

    # 8. Save final results
    results_path = save_dir / 'final_results.txt'
    with open(results_path, 'w') as f:
        f.write("CrissCross Word Classification (LINEAR PROBE) - Final Test Results\n")
        f.write("=" * 60 + "\n\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    logger.info(f"\nResults saved to: {results_path}")

    wandb.finish()

    return test_metrics


if __name__ == "__main__":
    main()
