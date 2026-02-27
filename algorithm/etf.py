import numpy as np
import cv2


def compute_etf(image_rgb, sigma_outer=3.0, iterations=3):
    """Compute Edge Tangent Flow from an RGB color image."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0

    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    Jxx = cv2.GaussianBlur(Ix * Ix, (0, 0), sigma_outer)
    Jxy = cv2.GaussianBlur(Ix * Iy, (0, 0), sigma_outer)
    Jyy = cv2.GaussianBlur(Iy * Iy, (0, 0), sigma_outer)

    # Tangent direction: perpendicular to gradient (follows edges)
    orientation = 0.5 * np.arctan2(2.0 * Jxy, Jyy - Jxx)
    magnitude = np.sqrt(Ix ** 2 + Iy ** 2)

    tx = np.cos(orientation)
    ty = np.sin(orientation)

    for _ in range(iterations):
        tx, ty = _refine_etf(tx, ty, magnitude, kernel_radius=5)

    return tx, ty, magnitude


def _refine_etf(tx, ty, mag, kernel_radius=5):
    """One refinement pass of the ETF (Kang et al., 2007)."""
    tx_new = np.zeros_like(tx)
    ty_new = np.zeros_like(ty)

    for dy in range(-kernel_radius, kernel_radius + 1):
        for dx in range(-kernel_radius, kernel_radius + 1):
            if dx == 0 and dy == 0:
                continue
            tx_s = np.roll(np.roll(tx, -dy, axis=0), -dx, axis=1)
            ty_s = np.roll(np.roll(ty, -dy, axis=0), -dx, axis=1)
            mag_s = np.roll(np.roll(mag, -dy, axis=0), -dx, axis=1)

            wm = 0.5 * (1.0 + np.tanh(5.0 * (mag_s - mag)))
            dot = tx * tx_s + ty * ty_s
            wd = np.abs(dot)
            phi = np.sign(dot)

            weight = wm * wd
            tx_new += weight * phi * tx_s
            ty_new += weight * phi * ty_s

    norm = np.sqrt(tx_new ** 2 + ty_new ** 2) + 1e-10
    return tx_new / norm, ty_new / norm
