"""Wrapper dataset for subsampling recordings while maintaining sampler compatibility."""

from typing import Set, List, Tuple
from torch.utils.data import Dataset


class SubsampledRecordingDataset(Dataset):
    """
    Wrapper that subsamples by recording while maintaining sampler compatibility.

    This wrapper selects a subset of recordings from the base dataset and remaps
    recording indices to be contiguous (0, 1, 2, ...). This is required for
    compatibility with RecordingShuffleSampler.

    Parameters
    ----------
    base_dataset : Dataset
        The underlying dataset with a segment_index attribute
    selected_recording_indices : Set[int]
        Set of recording indices to keep

    Attributes
    ----------
    segment_index : List[Tuple[int, int]]
        Remapped segment index for RecordingShuffleSampler compatibility
    """

    def __init__(self, base_dataset, selected_recording_indices: Set[int]):
        self.base_dataset = base_dataset
        self.selected_recordings = selected_recording_indices

        # Remap recording indices to be contiguous (0, 1, 2, ...)
        self.rec_idx_mapping = {
            old: new for new, old in enumerate(sorted(selected_recording_indices))
        }

        # Build new segment_index with remapped recording indices
        self._segment_index: List[Tuple[int, int]] = []
        self.global_idx_mapping: List[int] = []  # new_idx -> original_idx

        for orig_idx, (rec_idx, seg_idx) in enumerate(base_dataset.segment_index):
            if rec_idx in selected_recording_indices:
                new_rec_idx = self.rec_idx_mapping[rec_idx]
                self._segment_index.append((new_rec_idx, seg_idx))
                self.global_idx_mapping.append(orig_idx)

    @property
    def segment_index(self) -> List[Tuple[int, int]]:
        """Return remapped segment_index for RecordingShuffleSampler."""
        return self._segment_index

    def __len__(self) -> int:
        return len(self._segment_index)

    def __getitem__(self, idx: int):
        return self.base_dataset[self.global_idx_mapping[idx]]

    def close(self):
        """Close underlying dataset file handles."""
        self.base_dataset.close()
