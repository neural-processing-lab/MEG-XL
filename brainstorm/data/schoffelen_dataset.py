"""PyTorch Dataset for the Schoffelen 2019 MEG dataset."""

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings
from .utils import norm_sensor_positions

from .preprocessing import (
    preprocess_recording,
    cache_preprocessed,
    load_cached,
    get_cache_path,
    preprocess_segment_with_subsegments,
    shuffle_temporal_segments
)

SCHOFFELEN_VALID_CHANNELS = ['MLC62', 'MRF25', 'MRF11', 'MLF44', 'MRT41', 'MRF35', 'MRF22', 'MLT53', 'MRC51', 'MRF56', 'MLT33', 'MRC17', 'MRF61', 'MRC32', 'MLF54', 'MLC22', 'MLF64', 'MLP12', 'MLC16', 'MRF52', 'MRP31', 'MRP34', 'MLF14', 'MRC63', 'MLF56', 'MZF02', 'MRC31', 'MLC25', 'MLF55', 'MZC03', 'MLO23', 'MLT42', 'MLC55', 'MRC53', 'MRO14', 'MRO44', 'MLF52', 'MLP44', 'MLC23', 'MLT13', 'MRF65', 'MLP51', 'MLF43', 'MRC42', 'MLF13', 'MRC41', 'MRT46', 'MRP21', 'MLF41', 'MRC23', 'MRF51', 'MLF35', 'MLF33', 'MLO41', 'MRP43', 'MLO33', 'MRP32', 'MLP21', 'MRC11', 'MLF65', 'MRP42', 'MRC21', 'MRT34', 'MRC54', 'MRO21', 'MLT27', 'MRT13', 'MRO12', 'MLF67', 'MRF12', 'MLT34', 'MLC41', 'MLP43', 'MRO42', 'MRT51', 'MLT47', 'MRC61', 'MRF23', 'MRC14', 'MRO31', 'MZO01', 'MLT43', 'MLF45', 'MLC15', 'MLP35', 'MRP11', 'MRP22', 'MLF34', 'MLT14', 'MLO12', 'MRF33', 'MRC25', 'MZF01', 'MLC52', 'MLC53', 'MLC54', 'MRC52', 'MRO34', 'MRC13', 'MRP54', 'MLT16', 'MLT52', 'MLT55', 'MRF54', 'MLF46', 'MLP31', 'MRO24', 'MRP57', 'MLO22', 'MRP55', 'MRF14', 'MLT35', 'MRP56', 'MRT11', 'MRF64', 'MRT57', 'MLF22', 'MLO13', 'MLP45', 'MLC61', 'MLT45', 'MRC62', 'MZF03', 'MRF43', 'MLC14', 'MRT16', 'MRT22', 'MLT24', 'MRF42', 'MZC01', 'MLF32', 'MLP41', 'MLP22', 'MLF23', 'MRC55', 'MLC17', 'MLP56', 'MRF46', 'MLT25', 'MRF31', 'MLC13', 'MRF55', 'MRT32', 'MRT26', 'MLF63', 'MRO11', 'MRO33', 'MLT26', 'MLF51', 'MLO11', 'MLC63', 'MLC31', 'MRF44', 'MRT21', 'MRP45', 'MLP54', 'MLT12', 'MLF11', 'MLP32', 'MLT51', 'MRF13', 'MLC12', 'MLP33', 'MRT36', 'MLO34', 'MRT15', 'MLC51', 'MRT54', 'MLC32', 'MZO03', 'MLO43', 'MRC15', 'MLC21', 'MLO24', 'MLT57', 'MLF42', 'MLT44', 'MRT42', 'MLF25', 'MLP53', 'MRT35', 'MLF21', 'MLT31', 'MRC22', 'MRF32', 'MRP51', 'MRO43', 'MRO23', 'MLT36', 'MRF53', 'MRP52', 'MLO21', 'MRO22', 'MLC42', 'MRF34', 'MRF63', 'MRP33', 'MRT25', 'MRC12', 'MRP53', 'MRT37', 'MLP57', 'MRP23', 'MRP35', 'MRF21', 'MLF12', 'MRO32', 'MRP44', 'MLO31', 'MZC04', 'MRT27', 'MRO53', 'MLP11', 'MRT43', 'MLT11', 'MRF67', 'MLT15', 'MLT22', 'MZO02', 'MLO44', 'MLO14', 'MLO51', 'MLT56', 'MLT23', 'MRT23', 'MRT47', 'MRT44', 'MRF45', 'MRP12', 'MLT21', 'MLT54', 'MRO13', 'MLF24', 'MRO41', 'MLO42', 'MLT46', 'MRT52', 'MRF24', 'MRP41', 'MRT55', 'MRC16', 'MRT12', 'MLP55', 'MRT53', 'MRF62', 'MLP52', 'MRF41', 'MLC24', 'MLF66', 'MLT32', 'MLP34', 'MRT24', 'MRO51', 'MLO52', 'MLO32', 'MLO53', 'MRC24', 'MLF53', 'MRT14', 'MRT31', 'MLP23', 'MLT41', 'MLP42', 'MZP01', 'MRT45', 'MLF61', 'MRT56', 'MLF31', 'MRT33', 'MZC02']


class SchoffelenMEGDataset(Dataset):
    """
    PyTorch Dataset for the Schoffelen 2019 MEG dataset.

    This dataset handles:
    - Discovery of MEG recordings from the Schoffelen dataset
    - Lazy preprocessing with caching (band-pass filter, resample, channel selection)
    - Segmentation of continuous recordings into fixed-length windows
    - Efficient loading using persistent HDF5 file handles
    - Support for subjects with multiple runs

    Parameters
    ----------
    data_root : str
        Root directory of the Schoffelen dataset (e.g., "/path/to/schoffelen2019")
    segment_length : float
        Length of each segment in seconds
    cache_dir : str, optional
        Directory for storing preprocessed cache files (default: "./data/cache")
    subjects : List[str], optional
        List of subjects to include (e.g., ["sub-A2002", "sub-A2003"]). If None, use all 'A' subjects.
    tasks : List[str], optional
        List of tasks to include (e.g., ["auditory"]). If None, use all (excluding "rest").
    l_freq : float
        Low frequency cutoff for band-pass filter (default: 0.1 Hz)
    h_freq : float
        High frequency cutoff for band-pass filter (default: 40.0 Hz)
    target_sfreq : float
        Target sampling frequency after resampling (default: 50.0 Hz)
    channel_filter : Callable[[str], bool]
        Filter function for channels. Channels for which this function returns True will be kept.
        Default accepts all channels (to be replaced with specific channel list later).

    Example
    -------
    >>> dataset = SchoffelenMEGDataset(
    ...     data_root="/path/to/schoffelen2019",
    ...     segment_length=10.0,
    ...     subjects=["sub-A2002"],
    ...     tasks=["auditory"]
    ... )
    >>> sample = dataset[0]
    >>> print(sample['meg'].shape)  # (n_channels, n_timepoints)
    """

    def __init__(
        self,
        data_root: str,
        segment_length: float,
        cache_dir: str = "./data/cache",
        subjects: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        l_freq: float = 0.1,
        h_freq: float = 40.0,
        target_sfreq: float = 50.0,
        channel_filter: Callable[[str], bool] = lambda x: x.split('-')[0] in SCHOFFELEN_VALID_CHANNELS,
        max_channel_dim: Optional[int] = None,
        shuffle_segments: bool = False,
        shuffle_segment_duration: float = 3.0,
    ):
        self.data_root = Path(data_root)
        self.segment_length = segment_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.l_freq = l_freq
        self.h_freq = h_freq
        self.target_sfreq = target_sfreq
        self.channel_filter = channel_filter
        self.max_channel_dim = max_channel_dim
        self.shuffle_segments = shuffle_segments
        self.shuffle_segment_duration = shuffle_segment_duration

        # Filters
        self.subjects = subjects
        self.tasks = tasks if tasks is not None else ["auditory"]  # Default to auditory only

        # Discover all recordings
        self.recordings = self._discover_recordings()

        if len(self.recordings) == 0:
            raise ValueError(
                f"No recordings found in {self.data_root} with the specified filters. "
                f"Subjects: {subjects}, Tasks: {tasks}"
            )

        # Preprocess and cache all recordings
        self._preprocess_all()

        # Open file handles for all cached recordings
        self.file_handles: List[h5py.File] = []
        self._open_file_handles()

        # Build segment index: maps global index -> (recording_idx, segment_idx)
        self.segment_index = self._build_segment_index()

    def _discover_recordings(self) -> List[Dict[str, Any]]:
        """
        Discover all MEG recordings matching the specified filters.

        Returns
        -------
        recordings : List[Dict[str, Any]]
            List of recording metadata dictionaries
        """
        recordings = []

        # Iterate through subject directories (only 'A' subjects for auditory)
        subject_dirs = sorted(self.data_root.glob("sub-A*"))

        for subject_dir in subject_dirs:
            subject = subject_dir.name

            # Apply subject filter
            if self.subjects is not None and subject not in self.subjects:
                continue

            # Look for MEG data (no session subdirectory in Schoffelen dataset)
            meg_dir = subject_dir / "meg"
            if not meg_dir.exists():
                continue

            # Find task MEG files
            # Pattern 1: Files without runs: sub-AXXX_task-TASK_meg.ds
            # Pattern 2: Files with runs: sub-AXXX_task-TASK_run-X_meg.ds
            meg_files = sorted(meg_dir.glob(f"{subject}_task-*_meg.ds"))

            for meg_file in meg_files:
                # Extract task and run from filename
                # Format: sub-AXXX_task-TASK_meg.ds or sub-AXXX_task-TASK_run-X_meg.ds
                parts = meg_file.name.split("_")
                task = None
                run = "none"  # Default: no run

                for i, part in enumerate(parts):
                    if part.startswith("task-"):
                        task = part.replace("task-", "")
                    elif part.startswith("run-"):
                        run = part.replace("run-", "")

                if task is None:
                    continue

                # Skip rest tasks (we only want auditory)
                if task == "rest":
                    continue

                # Apply task filter
                if self.tasks is not None and task not in self.tasks:
                    continue

                # Map run to session for cache compatibility
                # run "none" -> session "none"
                # run "1" -> session "run1"
                # run "2" -> session "run2"
                if run == "none":
                    session = "none"
                else:
                    session = f"run{run}"

                recordings.append({
                    "subject": subject,
                    "session": session,  # Using session field to store run info
                    "task": task,
                    "run": run,  # Store actual run for metadata
                    "raw_path": meg_file,
                    "cache_path": get_cache_path(
                        self.cache_dir, subject, session, task,
                        l_freq=self.l_freq,
                        h_freq=self.h_freq,
                        target_sfreq=self.target_sfreq,
                        channel_filter_name="all_channels"  # Will be updated when specific channel list is provided
                    )
                })

        return recordings

    def _preprocess_all(self) -> None:
        """
        Preprocess all recordings that haven't been cached yet.
        """
        for i, rec in enumerate(self.recordings):
            if not rec["cache_path"].exists():
                print(f"Preprocessing recording {i+1}/{len(self.recordings)}: "
                      f"{rec['subject']} run={rec['run']} {rec['task']}")

                # Preprocess
                raw = preprocess_recording(
                    str(rec["raw_path"]),
                    l_freq=self.l_freq,
                    h_freq=self.h_freq,
                    target_sfreq=self.target_sfreq,
                    channel_filter=self.channel_filter
                )

                # Cache
                metadata = {
                    "subject": rec["subject"],
                    "session": rec["session"],
                    "task": rec["task"],
                    "run": rec["run"],
                    "dataset": "schoffelen2019"
                }
                cache_preprocessed(
                    raw, rec["cache_path"], metadata,
                    l_freq=self.l_freq,
                    h_freq=self.h_freq,
                    target_sfreq=self.target_sfreq,
                    channel_filter_name="all_channels"
                )

                print(f"  Cached to {rec['cache_path']}")
            else:
                print(f"Using cached recording {i+1}/{len(self.recordings)}: "
                      f"{rec['subject']} run={rec['run']} {rec['task']}")

    def _open_file_handles(self) -> None:
        """
        Open HDF5 file handles for all cached recordings.
        """
        self.file_handles = []
        for rec in self.recordings:
            h5_file = load_cached(rec["cache_path"])
            self.file_handles.append(h5_file)

    def _build_segment_index(self) -> List[Tuple[int, int]]:
        """
        Build an index mapping global segment index to (recording_idx, segment_idx).

        Returns
        -------
        segment_index : List[Tuple[int, int]]
            List of (recording_idx, segment_idx_within_recording) tuples
        """
        segment_index = []

        for rec_idx, h5_file in enumerate(self.file_handles):
            n_samples = h5_file.attrs["n_samples"]
            sfreq = h5_file.attrs["sample_freq"]

            # Calculate number of samples per segment
            samples_per_segment = int(self.segment_length * sfreq)

            # Calculate number of complete segments (skip partial segments)
            n_segments = n_samples // samples_per_segment

            # Add all segments from this recording
            for seg_idx in range(n_segments):
                segment_index.append((rec_idx, seg_idx))

        return segment_index

    def __len__(self) -> int:
        """Return total number of segments across all recordings."""
        return len(self.segment_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single segment.

        Parameters
        ----------
        idx : int
            Global segment index

        Returns
        -------
        sample : Dict[str, Any]
            Dictionary containing:
            - meg: torch.Tensor of shape (n_channels, n_timepoints)
            - subject: str
            - session: str
            - task: str
            - run: str
            - sensor_xyzdir: torch.Tensor of shape (n_channels, 6)
            - sensor_types: torch.Tensor of shape (n_channels,)
            - start_time: float (seconds)
            - end_time: float (seconds)
            - recording_idx: int
            - segment_idx: int
            - sensor_mask: torch.Tensor of shape (n_channels,)
        """
        # Get recording and segment indices
        rec_idx, seg_idx = self.segment_index[idx]

        # Get HDF5 file handle
        h5_file = self.file_handles[rec_idx]

        # Get recording metadata
        rec = self.recordings[rec_idx]

        # Get data properties
        sfreq = h5_file.attrs["sample_freq"]
        samples_per_segment = int(self.segment_length * sfreq)

        # Calculate sample range
        start_sample = seg_idx * samples_per_segment
        end_sample = start_sample + samples_per_segment

        # Load segment data
        meg_data = h5_file["data"][:, start_sample:end_sample]

        # Load sensor types (needed for separate scaling)
        sensor_types = h5_file["sensor_types"][:]

        # Preprocess with sub-segmentation
        meg_data = preprocess_segment_with_subsegments(
            meg_data=meg_data,
            sensor_types=sensor_types,
            sfreq=sfreq,
            subsegment_duration=3.0,
            baseline_duration=0.5,
            clip_range=(-5, 5)
        )

        # Optionally shuffle temporal segments (for ablation experiments)
        if self.shuffle_segments:
            meg_data = shuffle_temporal_segments(
                meg_data, self.shuffle_segment_duration, sfreq
            )

        # Load sensor positions (same for all segments in this recording)
        sensor_xyzdir = h5_file["sensor_xyzdir"][:]

        # Calculate timing
        start_time = start_sample / sfreq
        end_time = end_sample / sfreq

        sensor_xyzdir = norm_sensor_positions(sensor_xyzdir)

        # Pad channel dimension and sensor positions if needed
        if self.max_channel_dim is not None:
            meg_data = np.pad(meg_data, ((0, self.max_channel_dim - meg_data.shape[0]), (0, 0)))
            sensor_xyzdir = np.pad(sensor_xyzdir, ((0, self.max_channel_dim - sensor_xyzdir.shape[0]), (0, 0)))
            sensor_types = np.pad(sensor_types, (0, self.max_channel_dim - sensor_types.shape[0]))
            sensor_mask = np.zeros(self.max_channel_dim, dtype=np.float32)
            sensor_mask[:h5_file["data"].shape[0]] = 1.0
        else:
            sensor_mask = np.ones(meg_data.shape[0], dtype=np.float32)

        # Convert to torch tensors
        meg_tensor = torch.from_numpy(meg_data).float()
        sensor_xyzdir_tensor = torch.from_numpy(sensor_xyzdir).float()
        sensor_mask_tensor = torch.from_numpy(sensor_mask).float()
        sensor_types_tensor = torch.from_numpy(sensor_types).int()

        return {
            "meg": meg_tensor,
            "subject": h5_file.attrs["subject"],
            "session": h5_file.attrs["session"],
            "task": h5_file.attrs["task"],
            "run": h5_file.attrs.get("run", "none"),  # Get run from metadata
            "sensor_xyzdir": sensor_xyzdir_tensor,
            "sensor_types": sensor_types_tensor,
            "start_time": float(start_time),
            "end_time": float(end_time),
            "recording_idx": rec_idx,
            "segment_idx": seg_idx,
            "sensor_mask": sensor_mask_tensor
        }

    def __del__(self):
        """Close all file handles when the dataset is destroyed."""
        self.close()

    def close(self):
        """Explicitly close all HDF5 file handles."""
        for h5_file in self.file_handles:
            try:
                h5_file.close()
            except:
                pass
        self.file_handles = []


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from brainstorm.neuro_tokenizers.biocodec.model import BioCodecModel

    # Create dataset
    dataset = SchoffelenMEGDataset(
        data_root="/path/to/schoffelen2019",
        segment_length=150.0,
        subjects=["sub-A2002"],
        tasks=["auditory"],
        l_freq=0.1,
        h_freq=40.0,
        target_sfreq=50.0,
    )
    sample = dataset[0]
    print(sample['meg'].shape)  # (n_channels, n_timepoints)
    print(f"Subject: {sample['subject']}, Run: {sample['run']}, Task: {sample['task']}")
    print(f"MEG shape: {sample['meg'].shape}")  # (n_channels, n_timepoints)

    # Load BioCodec tokenizer (same as in train_criss_cross_multi.py)
    print("\n=== Loading BioCodec Tokenizer ===")
    tokenizer_path = Path("./brainstorm/neuro_tokenizers/biocodec_ckpt.pt")

    if not tokenizer_path.exists():
        print(f"Error: Tokenizer checkpoint not found at {tokenizer_path}")
        print("Skipping reconstruction test.")
    else:
        # Create model
        tokenizer = BioCodecModel._get_optimized_model()

        # Load checkpoint
        checkpoint = torch.load(tokenizer_path, map_location="cpu")

        # Remove _orig_mod prefix from state dict keys
        new_state_dict = {}
        for key, value in checkpoint["model_state_dict"].items():
            if key.startswith("_orig_mod."):
                new_key = key[len("_orig_mod."):]
            else:
                new_key = key
            new_state_dict[new_key] = value

        tokenizer.load_state_dict(new_state_dict)
        tokenizer.eval()

        print(f"✓ Tokenizer loaded successfully")
        print(f"  RVQ levels: {tokenizer.quantizer.n_q}")
        print(f"  Codebook size: {tokenizer.quantizer.bins}")

        # Reconstruct the sample
        print("\n=== Reconstructing Sample ===")
        with torch.no_grad():
            meg_input = sample['meg'].unsqueeze(1)  # Add batch dimension: (n_channels, 1, n_timepoints)
            print(f"Input shape: {meg_input.shape}")

            # Encode and decode through tokenizer
            encoded = tokenizer.encode(meg_input)
            reconstructed = tokenizer.decode(encoded)

            print(f"Encoded shape: {encoded}")
            print(f"Reconstructed shape: {reconstructed.shape}")

            # Remove batch dimension
            meg_input = meg_input.squeeze(1).numpy()
            reconstructed = reconstructed.squeeze(1).numpy()

        # Plot 3 channels
        print("\n=== Plotting Reconstruction ===")
        n_channels_to_plot = 3
        fig, axes = plt.subplots(n_channels_to_plot, 1, figsize=(15, 8))

        time_axis = np.arange(meg_input.shape[1]) / 50.0  # Assuming 50 Hz sampling rate

        for i in range(n_channels_to_plot):
            axes[i].plot(time_axis, meg_input[i], label='Original', alpha=0.7, linewidth=0.8)
            axes[i].plot(time_axis, reconstructed[i], label='Reconstructed', alpha=0.7, linewidth=0.8)
            axes[i].set_ylabel(f'Channel {i}')
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)

            if i == 0:
                axes[i].set_title('BioCodec Reconstruction - Original vs Reconstructed')
            if i == n_channels_to_plot - 1:
                axes[i].set_xlabel('Time (s)')

        plt.tight_layout()

        # Save plot
        output_path = Path("./schoffelen_reconstruction_test.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {output_path}")

        # Calculate reconstruction error
        mse = np.mean((meg_input[:n_channels_to_plot] - reconstructed[:n_channels_to_plot])**2)
        print(f"✓ Mean squared error (first {n_channels_to_plot} channels): {mse:.6f}")

    print("\n=== Done ===")
    breakpoint()
