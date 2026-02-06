"""MEG dataset wrapper for BioCodec training."""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from brainstorm.data.armeni_dataset import ArmeniMEGDataset


class MEGDATASET(Dataset):
    """
    MEG dataset wrapper for BioCodec training.

    Wraps ArmeniMEGDataset to provide data in BioCodec-compatible format.
    Returns (signal, sample_rate) tuples where signal has shape (C, T).

    Parameters
    ----------
    config : OmegaConf
        Configuration object containing dataset parameters
    mode : str
        Either "train" or "test" to determine which sessions to use
    """

    def __init__(self, config, mode="train"):
        self.mode = mode
        self.sample_rate = config.model.sample_rate

        # Determine which sessions to use based on mode
        # Using the val_session from config as the test set
        val_session = config.datasets.get("val_session", "ses-010")

        if mode == "test":
            # Use only the validation session for testing
            sessions = [val_session]
        elif mode == "train":
            # Use all sessions except the validation session
            # Note: ArmeniMEGDataset will discover all sessions if None is passed,
            # so we need to explicitly exclude the val_session
            sessions = None  # Will filter after initialization
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'train' or 'test'.")

        # Initialize the Armeni MEG dataset
        self.dataset = ArmeniMEGDataset(
            data_root=config.datasets.data_root,
            segment_length=config.datasets.segment_length,
            cache_dir=config.datasets.cache_dir,
            subjects=config.datasets.get("subjects", None),
            sessions=sessions if mode == "test" else None,  # Will filter train sessions below
            tasks=config.datasets.get("tasks", None),
            l_freq=config.datasets.l_freq,
            h_freq=config.datasets.h_freq,
            target_sfreq=config.datasets.target_sfreq,
            channel_filter=lambda x: x.startswith('M')  # MEG channels only
        )

        # For training mode, filter out the validation session efficiently
        # by checking recording metadata instead of loading actual data
        if mode == "train":
            # Build index of samples that are NOT from the val_session
            # Use segment_index and recordings to avoid loading data
            self.valid_indices = []
            for idx, (rec_idx, seg_idx) in enumerate(self.dataset.segment_index):
                session = self.dataset.recordings[rec_idx]['session']
                if session != val_session:
                    self.valid_indices.append(idx)

            if len(self.valid_indices) == 0:
                raise ValueError(
                    f"No training samples found after excluding validation session {val_session}"
                )

            print(f"Training dataset: {len(self.valid_indices)} samples "
                  f"(excluded {len(self.dataset.segment_index) - len(self.valid_indices)} validation samples)")
        else:
            # For test mode, use all samples (which are already filtered to val_session)
            self.valid_indices = list(range(len(self.dataset)))
            print(f"Test dataset: {len(self.valid_indices)} samples")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns
        -------
        signal : torch.Tensor
            MEG signal with shape (C, T) where C=269 channels
        sample_rate : int
            Sampling rate (should be 256 Hz)
        """
        # Map to actual dataset index
        actual_idx = self.valid_indices[idx]

        # Get sample from ArmeniMEGDataset
        sample = self.dataset[actual_idx]

        # Extract MEG signal (shape: C, T)
        signal = sample['meg']

        # Ensure it's a torch tensor
        if not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal, dtype=torch.float32)

        # Optional: clip extreme values (following EEG dataset pattern)
        signal = torch.clamp(signal, min=-10, max=10)

        return signal, self.sample_rate
