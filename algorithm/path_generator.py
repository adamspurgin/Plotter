import numpy as np
from scipy.spatial import KDTree

from .smoothing import chaikin_smooth
from .color_decompose import rgb_to_cmyk
from .density_field import prepare_density_field

CHANNEL_COLORS = {
    'C': '#00BFFF',
    'M': '#FF1493',
    'Y': '#DAA520',
    'K': '#000000',
}

# Process order: most visually important first (K carries detail)
CHANNEL_ORDER = ['K', 'C', 'M', 'Y']
CHANNEL_INDICES = {'K': 3, 'C': 0, 'M': 1, 'Y': 2}


def _stochastic_sample(density, base_spacing, rng):
    """
    Distribute points stochastically across the density field.

    Uses stratified sampling: the image is divided into cells of size
    base_spacing x base_spacing.  Each cell is accepted with probability
    equal to its mean density, then a point is placed at a random sub-pixel
    position within the accepted cell.  This keeps the distribution locally
    uniform while matching global tonal density.
    """
    h, w = density.shape
    step = max(1, int(round(base_spacing)))

    gy_starts = np.arange(0, h, step)
    gx_starts = np.arange(0, w, step)

    pts = []
    for gy in gy_starts:
        for gx in gx_starts:
            patch = density[gy:min(h, gy + step), gx:min(w, gx + step)]
            mean_d = float(patch.mean())
            if mean_d < 0.02 or rng.random() > mean_d:
                continue
            ly = rng.integers(0, patch.shape[0])
            lx = rng.integers(0, patch.shape[1])
            x = gx + lx + rng.uniform(-0.45, 0.45)
            y = gy + ly + rng.uniform(-0.45, 0.45)
            pts.append([float(np.clip(x, 0, w - 1)),
                        float(np.clip(y, 0, h - 1))])

    return np.array(pts, dtype=np.float64) if pts else np.array([]).reshape(0, 2)


def _nn_tsp(pts, direction_weight, progress_callback=None):
    """
    Nearest-neighbor TSP heuristic with directional weighting.

    At each step the algorithm considers the k nearest unvisited candidates
    and picks the one with the lowest weighted cost:

        cost = euclidean_distance * (1 + direction_weight * forward_dot)

    where forward_dot = max(0, dot(current_dir, direction_to_candidate)).
    This penalises continuing in the same direction, naturally breaking up
    long linear runs without preventing the path from being continuous.

    direction_weight == 0  → pure nearest-neighbour
    direction_weight ~1-2  → moderate-to-strong angular variety
    """
    n = len(pts)
    if n < 2:
        return pts

    tree = KDTree(pts)
    k = min(32, n)

    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=int)
    order[0] = 0
    visited[0] = True
    cur_dir = np.array([1.0, 0.0])

    for step in range(1, n):
        if progress_callback and step % 500 == 0:
            progress_callback(step / n)

        cur_pos = pts[order[step - 1]]
        _, candidates = tree.query(cur_pos, k=k + 1)

        best_idx = -1
        best_cost = np.inf
        best_dir = cur_dir

        for idx in candidates[1:]:          # candidates[0] is the point itself
            if visited[idx]:
                continue
            vec = pts[idx] - cur_pos
            dist = float(np.sqrt(vec[0] ** 2 + vec[1] ** 2)) + 1e-10
            vec_n = vec / dist
            forward = max(0.0, float(cur_dir[0] * vec_n[0] + cur_dir[1] * vec_n[1]))
            cost = dist * (1.0 + direction_weight * forward)
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
                best_dir = vec_n

        if best_idx == -1:
            # All k candidates already visited — fall back to global linear scan.
            # This only triggers near the very end of the tour.
            unvisited = np.where(~visited)[0]
            dists = np.linalg.norm(pts[unvisited] - cur_pos, axis=1)
            best_idx = int(unvisited[np.argmin(dists)])
            d = pts[best_idx] - cur_pos
            best_dir = d / (float(np.linalg.norm(d)) + 1e-10)

        order[step] = best_idx
        visited[best_idx] = True
        cur_dir = best_dir

    return pts[order]


def generate_channel_path(density, params, rng, progress_callback=None):
    """
    Generate a single continuous path for one color channel.

    1. Points are distributed stochastically, weighted by the density field.
    2. A nearest-neighbor TSP tour with directional weighting connects them.
    3. Chaikin smoothing rounds the result for the plotter.

    Returns an Nx2 numpy array of (x, y) points.
    """
    if progress_callback:
        progress_callback(0.0)

    pts = _stochastic_sample(density, params['base_spacing'], rng)

    if len(pts) < 2:
        return np.array([])

    if progress_callback:
        progress_callback(0.15)

    def tsp_cb(frac):
        if progress_callback:
            progress_callback(0.15 + frac * 0.75)

    path = _nn_tsp(
        pts,
        direction_weight=params.get('direction_weight', 1.0),
        progress_callback=tsp_cb,
    )

    chaikin_iters = params.get('chaikin_iterations', 3)
    path = chaikin_smooth(path, iterations=chaikin_iters)

    if progress_callback:
        progress_callback(1.0)

    return path


def generate_all_channels(image_rgb, params, progress_callback=None):
    """
    Convert an RGB image to CMYK channel paths.

    progress_callback(fraction: float, status: str) -> None
    """
    cmyk = rgb_to_cmyk(image_rgb)
    rng = np.random.default_rng(params.get('seed', None))

    paths = {}

    for i, name in enumerate(CHANNEL_ORDER):
        base_frac = i * 0.25

        if progress_callback:
            progress_callback(base_frac, f"Preparing {name} density field…")

        density = prepare_density_field(cmyk[CHANNEL_INDICES[name]])

        if progress_callback:
            progress_callback(base_frac + 0.03, f"Sampling & routing {name} channel…")

        def make_cb(base, label=name):
            def cb(frac):
                if progress_callback:
                    progress_callback(base + frac * 0.22,
                                      f"Routing {label}… {int(frac * 100)}%")
            return cb

        paths[name] = generate_channel_path(
            density, params, rng,
            progress_callback=make_cb(base_frac + 0.03),
        )

    if progress_callback:
        progress_callback(1.0, "Done.")

    return paths
