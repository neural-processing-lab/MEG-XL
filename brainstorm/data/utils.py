import numpy as np

def norm_sensor_positions(sensor_xyzdir: np.ndarray) -> np.ndarray:
    """Normalize sensor positions by centering and scaling.

    Args:
        sensor_xyzdir (np.ndarray): Array of shape (n_sensors, 6) with sensor positions in first 3.
    Returns:
        np.ndarray: Normalized sensor positions.
    """
    sensor_mean = np.mean(sensor_xyzdir[:, :3], axis=0, keepdims=True)
    sensor_xyzdir[:, :3] -= sensor_mean
    sensor_scale = np.sqrt(3 * np.mean(np.sum(sensor_xyzdir[:, :3] ** 2, axis=1)))
    sensor_xyzdir[:, :3] /= sensor_scale
    return sensor_xyzdir