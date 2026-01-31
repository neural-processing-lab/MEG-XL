"""Custom samplers for efficient MEG data loading."""

import torch
from torch.utils.data import Sampler
from typing import Iterator, Optional, List
import numpy as np


class RecordingShuffleSampler(Sampler[int]):
    """
    Sampler that shuffles recordings but yields segments sequentially within each recording.

    This sampler improves I/O throughput by ensuring that all segments from the same
    recording are loaded sequentially, taking advantage of locality in HDF5 files.
    While this sacrifices perfect IID shuffling, it provides a good trade-off between
    randomization and data loading efficiency.

    The shuffling happens at the recording level once per epoch. Within each recording,
    segments are yielded in their original sequential order.

    Parameters
    ----------
    dataset : ArmeniMEGDataset
        The dataset to sample from. Must have a `segment_index` attribute that maps
        global indices to (recording_idx, segment_idx) tuples.
    seed : int, optional
        Random seed for reproducible shuffling. If None, uses random shuffling.
    generator : torch.Generator, optional
        PyTorch random number generator for reproducible shuffling. If provided,
        this takes precedence over `seed`.

    Example
    -------
    >>> dataset = ArmeniMEGDataset(...)
    >>> sampler = RecordingShuffleSampler(dataset, seed=42)
    >>> dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4)
    >>>
    >>> # For reproducible training across epochs
    >>> for epoch in range(num_epochs):
    ...     sampler.set_epoch(epoch)  # Updates the random seed
    ...     for batch in dataloader:
    ...         train_step(batch)
    """

    def __init__(
        self,
        dataset,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ):
        self.dataset = dataset
        self.seed = seed
        self.generator = generator
        self.epoch = 0

        # Build recording-to-segments mapping
        self._build_recording_map()

    def _build_recording_map(self) -> None:
        """
        Build a mapping from recording_idx to list of global segment indices.

        This allows us to shuffle recordings and then yield all segments from
        each recording sequentially.
        """
        self.recording_to_segments: dict[int, List[int]] = {}

        for global_idx, (rec_idx, seg_idx) in enumerate(self.dataset.segment_index):
            if rec_idx not in self.recording_to_segments:
                self.recording_to_segments[rec_idx] = []
            self.recording_to_segments[rec_idx].append(global_idx)

        # Get list of recording indices
        self.recording_indices = sorted(self.recording_to_segments.keys())

    def __iter__(self) -> Iterator[int]:
        """
        Generate indices by shuffling recordings and yielding segments sequentially.

        Yields
        ------
        int
            Global segment indices in recording-shuffled order
        """
        # Create random number generator for this epoch
        if self.generator is not None:
            # Use provided generator
            g = self.generator
        elif self.seed is not None:
            # Create generator from seed + epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
        else:
            # Use default random state
            g = None

        # Shuffle recording indices
        recording_order = self.recording_indices.copy()
        if g is not None:
            # Use PyTorch for reproducible shuffling
            perm = torch.randperm(len(recording_order), generator=g).tolist()
            recording_order = [recording_order[i] for i in perm]
        else:
            # Use numpy for random shuffling
            np.random.shuffle(recording_order)

        # Yield all segments from each recording in shuffled recording order
        for rec_idx in recording_order:
            segment_indices = self.recording_to_segments[rec_idx]
            for global_idx in segment_indices:
                yield global_idx

    def __len__(self) -> int:
        """Return total number of segments."""
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler.

        This is used to ensure different shuffling across epochs when using
        a fixed seed for reproducibility.

        Parameters
        ----------
        epoch : int
            Current epoch number
        """
        self.epoch = epoch
