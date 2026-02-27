import numpy as np


def rgb_to_cmyk(image_rgb):
    """Convert 0-255 RGB image to 0.0-1.0 CMYK channels."""
    rgb = image_rgb.astype(np.float64) / 255.0
    K = 1.0 - np.max(rgb, axis=2)
    K = np.clip(K, 0.0, 0.9999)  # prevent division by zero
    C = (1.0 - rgb[:, :, 0] - K) / (1.0 - K)
    M = (1.0 - rgb[:, :, 1] - K) / (1.0 - K)
    Y = (1.0 - rgb[:, :, 2] - K) / (1.0 - K)
    return np.clip(C, 0, 1), np.clip(M, 0, 1), np.clip(Y, 0, 1), np.clip(K, 0, 1)
