"""MEG preprocessing utilities for the Armeni dataset."""

import h5py
import mne
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple
import hashlib
import json


def load_libribrain_sensors(
    json_path: str
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Load sensor information from LibriBrain JSON file.

    LibriBrain stores sensor information in a single JSON file at the dataset root,
    rather than embedding it in each recording file. This function parses the JSON
    format and converts it to the standard sensor_xyzdir and sensor_types format.

    Parameters
    ----------
    json_path : str
        Path to meg_sensors_information.json file

    Returns
    -------
    sensor_xyzdir_dict : Dict[str, np.ndarray]
        Dictionary mapping channel names to position+direction arrays (6 elements)
        Format: first 3 elements = XYZ position, last 3 = orientation vector
    sensor_types_dict : Dict[str, int]
        Dictionary mapping channel names to sensor type
        1 = magnetometer (coil_type 3024), 0 = gradiometer (coil_type 3012)

    Examples
    --------
    >>> sensor_xyzdir, sensor_types = load_libribrain_sensors('/path/to/meg_sensors_information.json')
    >>> sensor_xyzdir['MEG0111']  # Position + direction for channel MEG0111
    array([-0.1066, 0.0464, -0.0604, -0.0195, 0.0070, -0.9998])
    >>> sensor_types['MEG0111']  # Type: 1=mag, 0=grad
    1
    """
    with open(json_path, 'r') as f:
        sensors = json.load(f)

    sensor_xyzdir_dict = {}
    sensor_types_dict = {}

    for sensor in sensors:
        ch_name = sensor['ch_name']
        loc = np.array(sensor['loc'])
        coil_type = sensor['coil_type']

        # Position is first 3 elements of loc array
        pos = loc[:3]

        # Direction vector is elements 3-5 (first direction vector)

        dir_idx = 3
        if coil_type == 3012:
            # Planar gradiometer: use first direction vector
            dir_idx = 1
        dir_vec = loc[3 * dir_idx : 3 * (dir_idx + 1)]

        # Combine position + direction (6 elements total)
        sensor_xyzdir_dict[ch_name] = np.concatenate([pos, dir_vec])

        # Map coil type to sensor type
        # 3024 = magnetometer, 3012 = planar gradiometer
        if coil_type == 3024:
            sensor_types_dict[ch_name] = 1  # Magnetometer
        else:
            sensor_types_dict[ch_name] = 0  # Gradiometer

    return sensor_xyzdir_dict, sensor_types_dict


def preprocess_libribrain_h5(
    h5_path: str,
    sensor_xyzdir_dict: Dict[str, np.ndarray],
    sensor_types_dict: Dict[str, int],
    l_freq: float,
    h_freq: float,
    target_sfreq: float,
    channel_filter: Callable[[str], bool]
) -> mne.io.Raw:
    """
    Re-preprocess a LibriBrain h5 file with new parameters.

    Unlike other datasets that start from raw MEG files, LibriBrain data is already
    preprocessed and stored in h5 format. This function loads the h5 data, creates
    an MNE RawArray object, and applies additional filtering/resampling if needed.

    Parameters
    ----------
    h5_path : str
        Path to LibriBrain h5 file
    sensor_xyzdir_dict : Dict[str, np.ndarray]
        Sensor position+direction dictionary from load_libribrain_sensors()
    sensor_types_dict : Dict[str, int]
        Sensor type dictionary from load_libribrain_sensors()
    l_freq : float
        Low frequency cutoff for band-pass filter (Hz)
    h_freq : float
        High frequency cutoff for band-pass filter (Hz)
    target_sfreq : float
        Target sampling frequency after resampling (Hz)
    channel_filter : Callable[[str], bool]
        Filter function for channels

    Returns
    -------
    raw : mne.io.Raw
        Preprocessed raw MEG data (MNE RawArray)

    Examples
    --------
    >>> sensor_xyzdir, sensor_types = load_libribrain_sensors('/path/to/sensors.json')
    >>> raw = preprocess_libribrain_h5(
    ...     '/path/to/sub-0_ses-1_task-Sherlock1_run-1_proc-*.h5',
    ...     sensor_xyzdir, sensor_types,
    ...     l_freq=0.1, h_freq=40.0, target_sfreq=50.0,
    ...     channel_filter=lambda x: x.startswith('MEG')
    ... )
    >>> raw.info['sfreq']
    50.0
    """
    # Load data from h5 file
    with h5py.File(h5_path, 'r') as f:
        data = f['data'][:]  # Shape: (n_channels, n_samples)
        orig_sfreq = f.attrs['sample_frequency']
        ch_names = f.attrs['channel_names'].split(', ')
        ch_types_str = f.attrs['channel_types'].split(', ')

    # Build MNE info structure
    info = mne.create_info(ch_names=ch_names, sfreq=orig_sfreq, ch_types=ch_types_str)

    # Create RawArray from data
    raw = mne.io.RawArray(data, info, verbose=False)

    # Apply bandpass filter
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False, n_jobs=-1)

    # Resample if needed
    if abs(target_sfreq - orig_sfreq) > 0.1:
        raw.resample(sfreq=target_sfreq, verbose=False, n_jobs=-1)

    # Apply channel filter
    filtered_chs = [ch for ch in raw.ch_names if channel_filter(ch)]
    raw.pick(filtered_chs)

    return raw


def preprocess_recording(
    raw_path: str,
    l_freq: float = 0.1,
    h_freq: float = 40.0,
    target_sfreq: float = 50.0,
    channel_filter: Callable[[str], bool] = lambda _: True
) -> mne.io.Raw:
    """
    Preprocess a single MEG recording.

    Pipeline:
    1. Load raw data
    2. Band-pass filter [l_freq, h_freq] Hz
    3. Resample to target_sfreq Hz
    4. Keep only channels where channel_filter returns True
    5. Apply robust scaling (median=0, Q1=-1, Q3=1) per channel

    Parameters
    ----------
    raw_path : str
        Path to the raw MEG file (.ds directory for CTF format)
    l_freq : float
        Low frequency cutoff for band-pass filter (default: 0.1 Hz)
    h_freq : float
        High frequency cutoff for band-pass filter (default: 40.0 Hz)
    target_sfreq : float
        Target sampling frequency after resampling (default: 50.0 Hz)
    channel_filter : Callable[[str], bool]
        Filter function for channels

    Returns
    -------
    raw : mne.io.Raw
        Preprocessed raw MEG data
    """
    # Load raw data
    raw = mne.io.read_raw_ctf(raw_path, preload=True, verbose=False)

    # Band-pass filter
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False, n_jobs=-1)

    # Resample
    raw.resample(sfreq=target_sfreq, verbose=False, n_jobs=-1)

    # Select channels starting with the specified prefix
    ch_names = [ch for ch in raw.ch_names if channel_filter(ch)]
    raw.pick(ch_names)

    return raw


def get_sensor_positions(raw: mne.io.Raw) -> np.ndarray:
    """
    Extract 3D sensor positions and orientations from MNE Raw object.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw MEG data with channel information

    Returns
    -------
    sensor_xyzdir : np.ndarray
        Array of shape (n_channels, 6) containing sensor positions (first 3)
        and orientations (last 3) in meters
    sensor_types : np.ndarray
        Array of shape (n_channels,) containing sensor types (1 for magnetometers, 0 for gradiometers)
    """
    sensor_positions = []
    sensor_types = []
    for ch in raw.info["chs"]:
        pos = ch["loc"][:3]
        pos_list = pos.tolist()

        coil_type = str(ch["coil_type"])
        dir_idx = 3
        if "PLANAR" in coil_type:
            dir_idx = 1
        dir = ch["loc"][3 * dir_idx : 3 * (dir_idx + 1)].tolist()

        # Append combined position + direction (6 elements total)
        sensor_positions.append(pos_list + dir)

        if "MAG" in coil_type:
            sensor_types.append(1)
        else:
            sensor_types.append(0)

    return np.array(sensor_positions), np.array(sensor_types)


def cache_preprocessed(
    raw: mne.io.Raw,
    cache_path: Path,
    metadata: Dict[str, Any],
    l_freq: float = 0.1,
    h_freq: float = 40.0,
    target_sfreq: float = 50.0,
    channel_filter_name: str = "default"
) -> None:
    """
    Cache preprocessed MEG data to HDF5 file.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed raw MEG data
    cache_path : Path
        Path where the HDF5 cache file will be saved
    metadata : Dict[str, Any]
        Metadata to store in the HDF5 file (subject, session, task, etc.)
    l_freq : float
        Low frequency cutoff used for band-pass filter
    h_freq : float
        High frequency cutoff used for band-pass filter
    target_sfreq : float
        Target sampling frequency used for resampling
    channel_filter_name : str
        Name/identifier for the channel filter function used
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Get data and sensor positions
    data = raw.get_data()  # Shape: (n_channels, n_samples)
    sensor_xyzdir, sensor_types = get_sensor_positions(raw)

    chunks = (data.shape[0], round(raw.info['sfreq'])) # Chunk by channels and 1 second of data

    # Save to HDF5
    with h5py.File(cache_path, 'w') as f:
        # Store MEG data
        f.create_dataset('data', data=data, chunks=chunks, compression=None)

        # Store sensor positions
        f.create_dataset('sensor_xyzdir', data=sensor_xyzdir, compression=None)
        f.create_dataset('sensor_types', data=sensor_types, compression=None)

        # Store channel names
        ch_names_bytes = [name.encode('utf-8') for name in raw.ch_names]
        f.create_dataset('channel_names', data=ch_names_bytes, compression=None)

        # Store metadata as attributes
        f.attrs['sample_freq'] = raw.info['sfreq']
        f.attrs['n_channels'] = len(raw.ch_names)
        f.attrs['n_samples'] = data.shape[1]

        # Store preprocessing parameters for verification
        f.attrs['preproc_l_freq'] = l_freq
        f.attrs['preproc_h_freq'] = h_freq
        f.attrs['preproc_target_sfreq'] = target_sfreq
        f.attrs['preproc_channel_filter'] = channel_filter_name
        f.attrs['preproc_hash'] = compute_preproc_hash(l_freq, h_freq, target_sfreq, channel_filter_name)

        for key, value in metadata.items():
            f.attrs[key] = value

        # Ensure data is flushed to disk (important for NFS/cluster filesystems)
        f.flush()


def load_cached(cache_path: Path) -> h5py.File:
    """
    Load cached preprocessed data with an open file handle.

    Parameters
    ----------
    cache_path : Path
        Path to the HDF5 cache file

    Returns
    -------
    h5_file : h5py.File
        Open HDF5 file handle (caller is responsible for closing)
    """
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    return h5py.File(cache_path, 'r')


def compute_preproc_hash(
    l_freq: float,
    h_freq: float,
    target_sfreq: float,
    channel_filter_name: str = "default"
) -> str:
    """
    Compute a hash of preprocessing parameters for cache identification.

    Parameters
    ----------
    l_freq : float
        Low frequency cutoff for band-pass filter
    h_freq : float
        High frequency cutoff for band-pass filter
    target_sfreq : float
        Target sampling frequency after resampling
    channel_filter_name : str
        Name/identifier for the channel filter function

    Returns
    -------
    hash_str : str
        8-character hash of the preprocessing configuration
    """
    config = {
        "l_freq": l_freq,
        "h_freq": h_freq,
        "target_sfreq": target_sfreq,
        "channel_filter": channel_filter_name,
        "version": "with_orientations_v2"
    }
    # Create deterministic JSON string (sorted keys)
    config_str = json.dumps(config, sort_keys=True)
    # Compute SHA256 hash and take first 8 characters
    hash_obj = hashlib.sha256(config_str.encode('utf-8'))
    return hash_obj.hexdigest()[:8]


def get_cache_path(
    cache_dir: Path,
    subject: str,
    session: str,
    task: str,
    l_freq: float = 0.1,
    h_freq: float = 40.0,
    target_sfreq: float = 50.0,
    channel_filter_name: str = "default"
) -> Path:
    """
    Generate cache file path for a given recording with preprocessing parameters.

    The filename includes a hash of the preprocessing parameters to ensure
    different preprocessing configurations don't accidentally load wrong cache files.

    Parameters
    ----------
    cache_dir : Path
        Base directory for cache files
    subject : str
        Subject identifier (e.g., "sub-001")
    session : str
        Session identifier (e.g., "ses-001")
    task : str
        Task identifier (e.g., "compr")
    l_freq : float
        Low frequency cutoff for band-pass filter
    h_freq : float
        High frequency cutoff for band-pass filter
    target_sfreq : float
        Target sampling frequency after resampling
    channel_filter_name : str
        Name/identifier for the channel filter function

    Returns
    -------
    cache_path : Path
        Full path to the cache file including preprocessing hash
    """
    preproc_hash = compute_preproc_hash(l_freq, h_freq, target_sfreq, channel_filter_name)
    filename = f"{subject}_{session}_task-{task}_preproc-{preproc_hash}.h5"
    return cache_dir / filename


def _apply_baseline_correction(
    chunk: np.ndarray,
    sfreq: float,
    baseline_duration: float
) -> np.ndarray:
    """
    Apply baseline correction to a chunk using the first baseline_duration seconds.

    Parameters
    ----------
    chunk : np.ndarray
        MEG data chunk of shape (n_channels, n_samples)
    sfreq : float
        Sampling frequency in Hz
    baseline_duration : float
        Duration of baseline window in seconds

    Returns
    -------
    corrected_chunk : np.ndarray
        Baseline-corrected chunk of same shape as input
    """
    baseline_samples = min(int(baseline_duration * sfreq), chunk.shape[1])
    baseline_mean = np.mean(chunk[:, :baseline_samples], axis=1, keepdims=True)
    return chunk - baseline_mean


def _apply_robust_scaling(
    chunk: np.ndarray,
    sensor_types: np.ndarray
) -> np.ndarray:
    """
    Apply RobustScaler separately to magnetometers and gradiometers.

    Parameters
    ----------
    chunk : np.ndarray
        MEG data chunk of shape (n_channels, n_samples)
    sensor_types : np.ndarray
        Sensor types of shape (n_channels,) where 1=magnetometer, 0=gradiometer

    Returns
    -------
    scaled_chunk : np.ndarray
        Scaled chunk of same shape as input
    """
    from sklearn.preprocessing import RobustScaler

    mag_mask = sensor_types == 1
    grad_mask = sensor_types == 0

    if np.any(mag_mask):
        mag_scaler = RobustScaler()
        chunk[mag_mask, :] = mag_scaler.fit_transform(chunk[mag_mask, :].T).T

    if np.any(grad_mask):
        grad_scaler = RobustScaler()
        chunk[grad_mask, :] = grad_scaler.fit_transform(chunk[grad_mask, :].T).T

    return chunk


def _process_single_chunk(
    chunk: np.ndarray,
    sensor_types: np.ndarray,
    sfreq: float,
    baseline_duration: float,
    clip_range: tuple
) -> np.ndarray:
    """
    Process a single chunk with baseline correction, robust scaling, and clipping.

    Parameters
    ----------
    chunk : np.ndarray
        MEG data chunk of shape (n_channels, n_samples)
    sensor_types : np.ndarray
        Sensor types of shape (n_channels,) where 1=magnetometer, 0=gradiometer
    sfreq : float
        Sampling frequency in Hz
    baseline_duration : float
        Duration of baseline window in seconds
    clip_range : tuple
        Min and max values for clipping (min_val, max_val)

    Returns
    -------
    processed_chunk : np.ndarray
        Fully processed chunk of same shape as input
    """
    # Baseline correction
    chunk = _apply_baseline_correction(chunk, sfreq, baseline_duration)

    # Robust scaling
    chunk = _apply_robust_scaling(chunk, sensor_types)

    # Clipping
    chunk = np.clip(chunk, clip_range[0], clip_range[1])

    return chunk


def preprocess_segment_with_subsegments(
    meg_data: np.ndarray,
    sensor_types: np.ndarray,
    sfreq: float,
    subsegment_duration: float = 3.0,
    baseline_duration: float = 0.5,
    clip_range: tuple = (-5, 5)
) -> np.ndarray:
    """
    Preprocess MEG segment by splitting into sub-segments, applying baseline
    correction and RobustScaler to each sub-segment, then concatenating.

    This function splits the input MEG segment into fixed-duration sub-segments,
    applies baseline correction and RobustScaler independently to each sub-segment,
    then concatenates them back together. This ensures preprocessing operates on
    consistent temporal windows regardless of the overall segment length.

    For segments shorter than subsegment_duration, the entire segment is processed
    as a single chunk without splitting.

    Parameters
    ----------
    meg_data : np.ndarray
        MEG data of shape (n_channels, n_samples)
    sensor_types : np.ndarray
        Sensor types of shape (n_channels,) where 1=magnetometer, 0=gradiometer
    sfreq : float
        Sampling frequency in Hz
    subsegment_duration : float
        Duration of sub-segments in seconds (default: 3.0)
    baseline_duration : float
        Duration of baseline window in seconds (default: 0.5)
    clip_range : tuple
        Min and max values for clipping (default: (-5, 5))

    Returns
    -------
    processed_data : np.ndarray
        Preprocessed data of same shape as input (n_channels, n_samples)

    Examples
    --------
    >>> # 30s segment at 50Hz -> 10 chunks of 3s each
    >>> meg_data = np.random.randn(270, 1500)  # 270 channels, 30s at 50Hz
    >>> sensor_types = np.zeros(270)
    >>> processed = preprocess_segment_with_subsegments(meg_data, sensor_types, 50.0)
    >>> processed.shape
    (270, 1500)

    >>> # 1s segment -> processed as single chunk (no splitting)
    >>> meg_data = np.random.randn(270, 50)  # 270 channels, 1s at 50Hz
    >>> processed = preprocess_segment_with_subsegments(meg_data, sensor_types, 50.0)
    >>> processed.shape
    (270, 50)
    """
    n_samples = meg_data.shape[1]
    subsegment_samples = int(subsegment_duration * sfreq)

    # If segment is shorter than subsegment_duration, process as single chunk
    if n_samples <= subsegment_samples:
        return _process_single_chunk(
            meg_data, sensor_types, sfreq, baseline_duration, clip_range
        )

    # Calculate number of complete chunks
    n_complete_chunks = n_samples // subsegment_samples
    has_partial_chunk = (n_samples % subsegment_samples) > 0

    # Process complete chunks
    chunks = []
    for i in range(n_complete_chunks):
        start = i * subsegment_samples
        end = (i + 1) * subsegment_samples
        chunk = meg_data[:, start:end]
        processed = _process_single_chunk(
            chunk, sensor_types, sfreq, baseline_duration, clip_range
        )
        chunks.append(processed)

    # Process partial chunk if exists
    if has_partial_chunk:
        start = n_complete_chunks * subsegment_samples
        chunk = meg_data[:, start:]
        processed = _process_single_chunk(
            chunk, sensor_types, sfreq, baseline_duration, clip_range
        )
        chunks.append(processed)

    # Concatenate along time axis
    return np.concatenate(chunks, axis=1)


def shuffle_temporal_segments(
    meg_data: np.ndarray,
    segment_duration: float,
    sfreq: float
) -> np.ndarray:
    """
    Randomly shuffle temporal segments within MEG data.

    This function splits the input MEG data into fixed-duration segments,
    randomly shuffles their order, and concatenates them back together.
    Useful for ablation experiments to test whether temporal order matters.

    Parameters
    ----------
    meg_data : np.ndarray
        MEG data of shape (n_channels, n_samples)
    segment_duration : float
        Duration of each segment in seconds (e.g., 3.0)
    sfreq : float
        Sampling frequency in Hz

    Returns
    -------
    shuffled_data : np.ndarray
        Shuffled data with same shape as input (n_channels, n_samples)

    Examples
    --------
    >>> # 150s segment at 50Hz -> 50 chunks of 3s each, shuffled
    >>> meg_data = np.random.randn(270, 7500)  # 270 channels, 150s at 50Hz
    >>> shuffled = shuffle_temporal_segments(meg_data, 3.0, 50.0)
    >>> shuffled.shape
    (270, 7500)
    """
    n_samples = meg_data.shape[1]
    samples_per_segment = int(segment_duration * sfreq)

    # If segment is shorter than segment_duration, return as-is
    if n_samples <= samples_per_segment:
        return meg_data

    # Calculate number of complete segments
    n_complete_segments = n_samples // samples_per_segment

    # Split into segments
    segments = [
        meg_data[:, i * samples_per_segment:(i + 1) * samples_per_segment]
        for i in range(n_complete_segments)
    ]

    # Handle remainder if any
    remainder_start = n_complete_segments * samples_per_segment
    if remainder_start < n_samples:
        remainder = meg_data[:, remainder_start:]
    else:
        remainder = None

    # Shuffle segments randomly
    np.random.shuffle(segments)

    # Concatenate back (remainder stays at end)
    if remainder is not None:
        segments.append(remainder)

    return np.concatenate(segments, axis=1)


if __name__ == "__main__":
    """Test the preprocessing hash system."""
    print("Testing preprocessing hash system...\n")

    # Test 1: Same parameters should produce same hash
    print("Test 1: Deterministic hashing")
    hash1 = compute_preproc_hash(0.1, 40.0, 50.0, 'MEG_only')
    hash2 = compute_preproc_hash(0.1, 40.0, 50.0, 'MEG_only')
    print(f"  Hash 1: {hash1}")
    print(f"  Hash 2: {hash2}")
    print(f"  ✓ Same params produce same hash: {hash1 == hash2}")

    # Test 2: Different parameters should produce different hash
    print("\nTest 2: Different parameters produce different hashes")
    hash3 = compute_preproc_hash(0.1, 128.0, 256.0, 'MEG_only')
    print(f"  Config 1 (0.1-40Hz, 50Hz): {hash1}")
    print(f"  Config 2 (0.1-128Hz, 256Hz): {hash3}")
    print(f"  ✓ Different params produce different hashes: {hash1 != hash3}")

    # Test 3: Show example cache paths
    print("\nTest 3: Cache path generation")
    cache_dir = Path('./data/cache')
    path1 = get_cache_path(
        cache_dir, 'sub-001', 'ses-001', 'compr',
        l_freq=0.1, h_freq=40.0, target_sfreq=50.0,
        channel_filter_name='MEG_only'
    )
    path2 = get_cache_path(
        cache_dir, 'sub-001', 'ses-001', 'compr',
        l_freq=0.1, h_freq=128.0, target_sfreq=256.0,
        channel_filter_name='MEG_only'
    )

    print(f"  Config 1 path: {path1.name}")
    print(f"  Config 2 path: {path2.name}")
    print(f"  ✓ Different configs produce different paths: {path1 != path2}")

    # Test 4: Channel filter names matter
    print("\nTest 4: Channel filter names are included in hash")
    hash4 = compute_preproc_hash(0.1, 40.0, 50.0, 'MEG_only')
    hash5 = compute_preproc_hash(0.1, 40.0, 50.0, 'ALL_channels')
    print(f"  MEG_only filter: {hash4}")
    print(f"  ALL_channels filter: {hash5}")
    print(f"  ✓ Different filter names produce different hashes: {hash4 != hash5}")

    print("\n✅ All tests passed!")

