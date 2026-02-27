import numpy as np
import cv2


def prepare_density_field(channel, gamma=1.2, smooth_sigma=1.5):
    """Convert raw ink channel to a clean density field for path generation."""
    ch_u8 = (channel * 255).astype(np.uint8)
    for _ in range(2):
        ch_u8 = cv2.bilateralFilter(ch_u8, d=9, sigmaColor=50, sigmaSpace=50)

    density = ch_u8.astype(np.float64) / 255.0
    density = np.power(density, gamma)
    density = cv2.GaussianBlur(density, (0, 0), smooth_sigma)
    return np.clip(density, 0.0, 1.0)
