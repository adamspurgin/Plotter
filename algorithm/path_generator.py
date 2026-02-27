import numpy as np

from .hilbert import d2xy
from .occupancy import OccupancyGrid
from .smoothing import chaikin_smooth, resample_polyline
from .color_decompose import rgb_to_cmyk
from .density_field import prepare_density_field
from .etf import compute_etf

CHANNEL_ANGLES = {'C': 15, 'M': 75, 'Y': 0, 'K': 45}

CHANNEL_COLORS = {
    'C': '#00BFFF',
    'M': '#FF1493',
    'Y': '#DAA520',
    'K': '#000000',
}

# Process order: most visually important first (K carries detail)
CHANNEL_ORDER = ['K', 'C', 'M', 'Y']
CHANNEL_INDICES = {'K': 3, 'C': 0, 'M': 1, 'Y': 2}


def rotate_flow_field(tx, ty, angle_deg):
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return tx * cos_t - ty * sin_t, tx * sin_t + ty * cos_t


def _sample_density(density, cx, cy, cell_size):
    h, w = density.shape
    x0 = max(0, int(cx - cell_size / 2))
    x1 = min(w, int(cx + cell_size / 2) + 1)
    y0 = max(0, int(cy - cell_size / 2))
    y1 = min(h, int(cy + cell_size / 2) + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    return float(density[y0:y1, x0:x1].mean())


def _nudge_toward_flow(backbone, tx, ty, alpha=0.4, step=1.0):
    """Vectorised: nudge each backbone point toward the local ETF direction."""
    h, w = tx.shape
    result = backbone.copy()

    ix = np.clip(backbone[:, 0].astype(int), 0, w - 1)
    iy = np.clip(backbone[:, 1].astype(int), 0, h - 1)
    etf_t = np.stack([tx[iy, ix], ty[iy, ix]], axis=1)

    tangents = np.zeros_like(backbone)
    tangents[:-1] = backbone[1:] - backbone[:-1]
    tangents[-1] = tangents[-2] if len(backbone) > 1 else np.array([1.0, 0.0])
    norms = np.sqrt((tangents ** 2).sum(axis=1, keepdims=True)) + 1e-10
    tangents /= norms

    valid = (backbone[:, 0] >= 0) & (backbone[:, 0] < w) & \
            (backbone[:, 1] >= 0) & (backbone[:, 1] < h)
    result[valid] = backbone[valid] + alpha * (etf_t[valid] - tangents[valid]) * step
    return result


def generate_channel_path(density, tx, ty, pen_width, params,
                           progress_callback=None):
    """
    Generate a single continuous non-self-intersecting path for one channel.
    Returns an Nx2 numpy array of (x, y) points.
    """
    h, w = density.shape
    base_spacing = params['base_spacing']
    min_spacing = max(pen_width * 1.2, params.get('min_spacing', pen_width * 1.2))
    max_amp = params['max_amplitude']
    step = params['step_size']
    base_freq = params.get('base_freq', 0.02)
    max_freq = params.get('max_freq', 0.15)
    chaikin_iters = params.get('chaikin_iterations', 3)

    # ── Step 1: Hilbert-ordered cell visitation ───────────────────────────────
    cell_size = base_spacing
    grid_w = max(1, int(np.ceil(w / cell_size)))
    grid_h = max(1, int(np.ceil(h / cell_size)))
    hilbert_n = 1
    while hilbert_n < max(grid_w, grid_h):
        hilbert_n *= 2

    waypoints = []
    for d in range(hilbert_n * hilbert_n):
        gx, gy = d2xy(hilbert_n, d)
        if gx < grid_w and gy < grid_h:
            cx = (gx + 0.5) * cell_size
            cy = (gy + 0.5) * cell_size
            if cx < w and cy < h and _sample_density(density, cx, cy, cell_size) > 0.02:
                waypoints.append([cx, cy])

    if len(waypoints) < 2:
        return np.array([])

    waypoints = np.array(waypoints, dtype=np.float64)

    # ── Step 2: Smooth backbone ───────────────────────────────────────────────
    backbone = chaikin_smooth(waypoints, iterations=chaikin_iters)

    # ── Step 3: Resample and align to flow field ──────────────────────────────
    backbone = resample_polyline(backbone, step)
    if len(backbone) < 2:
        return np.array([])
    backbone = _nudge_toward_flow(backbone, tx, ty, alpha=0.4, step=step)

    # ── Pre-compute tortuosity modulation (vectorised) ────────────────────────
    n_pts = len(backbone)
    px = backbone[:, 0]
    py = backbone[:, 1]

    ix_arr = np.clip(px.astype(int), 0, w - 1)
    iy_arr = np.clip(py.astype(int), 0, h - 1)
    d_vals = density[iy_arr, ix_arr]

    tangents = np.zeros((n_pts, 2))
    tangents[:-1] = backbone[1:] - backbone[:-1]
    tangents[-1] = tangents[-2] if n_pts > 1 else np.array([1.0, 0.0])
    norms = np.sqrt((tangents ** 2).sum(axis=1, keepdims=True)) + 1e-10
    tangents /= norms
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    amps = max_amp * (d_vals ** 1.2)
    freqs = base_freq + d_vals * (max_freq - base_freq)
    phases = np.cumsum(2.0 * np.pi * freqs * step)
    offsets = amps * np.sin(phases)

    fx_all = px + offsets * normals[:, 0]
    fy_all = py + offsets * normals[:, 1]

    # ── Step 5: Sequential non-intersection check ─────────────────────────────
    occupancy = OccupancyGrid(w, h, cell_size=min_spacing)
    final_path = []
    min_spacing_sq = min_spacing * min_spacing

    for i in range(n_pts):
        if progress_callback and i % 2000 == 0:
            progress_callback(i / n_pts)

        fx, fy = fx_all[i], fy_all[i]
        if not (0.0 <= fx < w and 0.0 <= fy < h):
            continue

        if occupancy.is_clear(fx, fy, min_spacing):
            final_path.append([fx, fy])
            occupancy.insert(fx, fy)
        else:
            # Deflect: try reduced amplitudes toward the unmodulated position
            orig_x, orig_y = px[i], py[i]
            orig_off = offsets[i]
            nx, ny = normals[i]
            for scale in (0.5, 0.25, 0.0):
                x2 = orig_x + orig_off * scale * nx
                y2 = orig_y + orig_off * scale * ny
                if 0.0 <= x2 < w and 0.0 <= y2 < h and occupancy.is_clear(x2, y2, min_spacing):
                    final_path.append([x2, y2])
                    occupancy.insert(x2, y2)
                    break

    return np.array(final_path, dtype=np.float64) if final_path else np.array([])


def generate_all_channels(image_rgb, params, progress_callback=None):
    """
    Convert an RGB image to CMYK channel paths.

    progress_callback(fraction: float, status: str) -> None
    """
    cmyk = rgb_to_cmyk(image_rgb)
    h, w = image_rgb.shape[:2]

    if progress_callback:
        progress_callback(0.0, "Computing edge tangent flow…")

    tx, ty, _ = compute_etf(image_rgb, iterations=params.get('etf_iterations', 3))

    paths = {}

    for i, name in enumerate(CHANNEL_ORDER):
        base_frac = 0.15 + i * 0.21

        if progress_callback:
            progress_callback(base_frac, f"Preparing {name} density field…")

        density = prepare_density_field(cmyk[CHANNEL_INDICES[name]])
        tx_r, ty_r = rotate_flow_field(tx, ty, CHANNEL_ANGLES[name])

        if progress_callback:
            progress_callback(base_frac + 0.04, f"Tracing {name} channel…")

        def make_cb(base, label=name):
            def cb(frac):
                if progress_callback:
                    progress_callback(base + frac * 0.17, f"Tracing {label}… {int(frac * 100)}%")
            return cb

        paths[name] = generate_channel_path(
            density, tx_r, ty_r,
            params['pen_width'], params,
            progress_callback=make_cb(base_frac + 0.04),
        )

    if progress_callback:
        progress_callback(1.0, "Done.")

    return paths
