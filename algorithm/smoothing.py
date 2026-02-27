import numpy as np


def chaikin_smooth(points, iterations=4):
    """Chaikin's corner-cutting algorithm: converges to quadratic B-spline."""
    pts = np.asarray(points, dtype=np.float64)
    for _ in range(iterations):
        if len(pts) < 3:
            break
        Q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        R = 0.25 * pts[:-1] + 0.75 * pts[1:]
        smooth = np.empty((2 * len(Q), 2))
        smooth[0::2] = Q
        smooth[1::2] = R
        pts = smooth
    return pts


def resample_polyline(points, step_size):
    """Resample polyline to uniform arc-length spacing (vectorized)."""
    if len(points) < 2:
        return points
    diffs = np.diff(points, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_length = cumulative[-1]

    if total_length < 1e-10:
        return points

    num_samples = max(2, int(total_length / step_size))
    target_dists = np.linspace(0.0, total_length, num_samples)

    idx = np.searchsorted(cumulative, target_dists, side='right') - 1
    idx = np.clip(idx, 0, len(points) - 2)

    t = np.where(
        seg_lens[idx] > 1e-10,
        (target_dists - cumulative[idx]) / seg_lens[idx],
        0.0
    )[:, np.newaxis]

    return points[idx] * (1.0 - t) + points[idx + 1] * t
