# Non-Intersecting Contiguous Color Line Art for Pen Plotters

## Problem Statement

Generate a color reproduction of an input image using a small set of colored pens on a plotter, where each pen traces a single continuous, non-self-intersecting path, and the combined visual effect of all paths reproduces the full-color image at viewing distance.

---

## Core Insight

Human vision integrates ink density over an area at viewing distance. A color image can be decomposed into independent ink channels — analogous to CMYK printing — where each channel's local darkness is encoded by the local density and tortuosity of a single continuous line. The key constraint beyond monochrome scribble art is that **multiple line layers coexist in the same physical space**, so inter-layer collision management and channel separation strategy become first-class design concerns.

---

## 1. Color Decomposition Strategy

### 1a. CMYK Separation (4 pens)

The natural first choice mirrors print technology. Convert the input image from sRGB to CMYK:

```python
import numpy as np
import cv2

def rgb_to_cmyk(image_rgb):
    """Convert 0-255 RGB image to 0.0-1.0 CMYK channels."""
    rgb = image_rgb.astype(np.float64) / 255.0
    K = 1.0 - np.max(rgb, axis=2)
    K = np.clip(K, 0, 0.9999)  # prevent division by zero
    C = (1.0 - rgb[:,:,0] - K) / (1.0 - K)
    M = (1.0 - rgb[:,:,1] - K) / (1.0 - K)
    Y = (1.0 - rgb[:,:,2] - K) / (1.0 - K)
    return np.clip(C, 0, 1), np.clip(M, 0, 1), np.clip(Y, 0, 1), np.clip(K, 0, 1)
```

Each channel (C, M, Y, K) becomes an independent density field driving one pen's path. The K (black) channel carries structural detail and contrast; CMY carry chroma.

### 1b. Custom Palette Separation (N pens)

Most plotters use a fixed pen set (e.g., 6-8 Staedtler triplus fineliners). For arbitrary pen colors, solve a **non-negative least squares** color decomposition per pixel:

```python
from scipy.optimize import nnls

def decompose_to_palette(image_rgb, pen_colors_rgb):
    """
    pen_colors_rgb: Nx3 array of pen RGB values (0-255).
    Returns: H x W x N array of ink densities per pen.
    """
    h, w, _ = image_rgb.shape
    N = len(pen_colors_rgb)

    # Work in linear light for physically accurate mixing
    img_lin = srgb_to_linear(image_rgb.astype(np.float64) / 255.0)
    pens_lin = srgb_to_linear(pen_colors_rgb.astype(np.float64) / 255.0)

    # Subtractive mixing model: paper_white * product(1 - ink_i * color_i)
    # Linearize via log: approximate as additive in optical density
    paper_white = np.array([1.0, 1.0, 1.0])
    target_od = -np.log(img_lin + 1e-6)  # optical density of target
    pen_od = -np.log(1.0 - pens_lin / 255.0 + 1e-6)  # OD per unit ink

    channels = np.zeros((h, w, N))
    for y in range(h):
        for x in range(w):
            channels[y, x], _ = nnls(pen_od.T, target_od[y, x])

    return np.clip(channels, 0, 1)
```

For speed, precompute a lookup table over quantized RGB values (e.g., 64³ = 262,144 entries) rather than solving NNLS per pixel.

### 1c. Perceptual Refinement with Dithering

After decomposition, apply **error diffusion** (Floyd-Steinberg or Stucki) in the continuous density domain to sharpen edges at the tonal resolution limit of the line art:

```python
def diffuse_error_continuous(channel, target_levels=32):
    """Quantize channel to target_levels and diffuse error."""
    h, w = channel.shape
    result = channel.copy()
    for y in range(h):
        for x in range(w):
            old = result[y, x]
            new = round(old * target_levels) / target_levels
            result[y, x] = new
            err = old - new
            if x + 1 < w:     result[y, x+1]     += err * 7/16
            if y + 1 < h:
                if x > 0:     result[y+1, x-1]   += err * 3/16
                result[y+1, x]                     += err * 5/16
                if x + 1 < w: result[y+1, x+1]    += err * 1/16
    return np.clip(result, 0, 1)
```

---

## 2. Per-Channel Density Field Preparation

Each channel's raw ink density (0.0 = no ink, 1.0 = full coverage) passes through the same preprocessing pipeline adapted from monochrome scribble art:

```python
def prepare_density_field(channel, gamma=1.2, smooth_sigma=1.5):
    """Convert raw ink channel to a clean density field for path generation."""
    # 1. Bilateral filter: smooth noise, preserve edges
    ch_u8 = (channel * 255).astype(np.uint8)
    for _ in range(2):
        ch_u8 = cv2.bilateralFilter(ch_u8, d=9, sigmaColor=50, sigmaSpace=50)

    density = ch_u8.astype(np.float64) / 255.0

    # 2. Apply gamma for perceptual linearization
    density = np.power(density, gamma)

    # 3. Final smooth for gradient stability during path generation
    density = cv2.GaussianBlur(density, (0, 0), smooth_sigma)

    return np.clip(density, 0, 1)
```

---

## 3. Shared Orientation Field (Edge Tangent Flow)

All pen channels share a single orientation field computed from the original color image. This ensures lines from different pens flow in harmonious directions, producing a coherent visual result rather than competing patterns.

```python
def compute_etf(image_rgb, sigma_inner=1.0, sigma_outer=3.0, iterations=3):
    """Compute Edge Tangent Flow from color image."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Structure tensor
    Jxx = cv2.GaussianBlur(Ix * Ix, (0, 0), sigma_outer)
    Jxy = cv2.GaussianBlur(Ix * Iy, (0, 0), sigma_outer)
    Jyy = cv2.GaussianBlur(Iy * Iy, (0, 0), sigma_outer)

    # Initial tangent field: perpendicular to gradient (edge-following)
    orientation = 0.5 * np.arctan2(2 * Jxy, Jyy - Jxx)
    magnitude = np.sqrt(Ix**2 + Iy**2)

    tx = np.cos(orientation)
    ty = np.sin(orientation)

    # Iterative ETF refinement (Kang et al., 2007)
    for _ in range(iterations):
        tx, ty = _refine_etf(tx, ty, magnitude, kernel_radius=5)

    return tx, ty, magnitude

def _refine_etf(tx, ty, mag, kernel_radius=5):
    h, w = tx.shape
    tx_new = np.zeros_like(tx)
    ty_new = np.zeros_like(ty)

    for dy in range(-kernel_radius, kernel_radius + 1):
        for dx in range(-kernel_radius, kernel_radius + 1):
            if dx == 0 and dy == 0:
                continue
            # Shifted arrays (with border handling)
            tx_s = np.roll(np.roll(tx, -dy, axis=0), -dx, axis=1)
            ty_s = np.roll(np.roll(ty, -dy, axis=0), -dx, axis=1)
            mag_s = np.roll(np.roll(mag, -dy, axis=0), -dx, axis=1)

            # Spatial weight (box filter)
            ws = 1.0

            # Magnitude weight: stronger gradients contribute more
            wm = 0.5 * (1.0 + np.tanh(5.0 * (mag_s - mag)))

            # Direction weight: aligned neighbors contribute more
            dot = tx * tx_s + ty * ty_s
            wd = np.abs(dot)

            # Sign flip for anti-aligned vectors
            phi = np.sign(dot)

            weight = ws * wm * wd
            tx_new += weight * phi * tx_s
            ty_new += weight * phi * ty_s

    # Normalize
    norm = np.sqrt(tx_new**2 + ty_new**2) + 1e-10
    return tx_new / norm, ty_new / norm
```

---

## 4. Path Generation: Flow-Guided Density-Modulated Streamlines

This is the heart of the algorithm. For each color channel, generate a single continuous non-self-intersecting path whose local density and tortuosity encode that channel's ink density.

### 4a. The Non-Intersection Strategy

Self-intersection avoidance uses a **spatial occupancy grid**. Every point deposited by the current pen is registered in a grid. Before extending the path, check that the proposed segment doesn't come within a minimum clearance distance of any previously deposited point from the same pen.

```python
class OccupancyGrid:
    """Spatial grid for fast proximity queries to enforce non-intersection."""

    def __init__(self, width, height, cell_size):
        self.cell_size = cell_size
        self.cols = int(np.ceil(width / cell_size))
        self.rows = int(np.ceil(height / cell_size))
        self.grid = [[[] for _ in range(self.cols)] for _ in range(self.rows)]

    def insert(self, x, y):
        c, r = int(x / self.cell_size), int(y / self.cell_size)
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.grid[r][c].append((x, y))

    def min_distance(self, x, y, radius):
        """Return minimum distance to any registered point within search radius."""
        c0 = max(0, int((x - radius) / self.cell_size))
        c1 = min(self.cols - 1, int((x + radius) / self.cell_size))
        r0 = max(0, int((y - radius) / self.cell_size))
        r1 = min(self.rows - 1, int((y + radius) / self.cell_size))

        min_d = float('inf')
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                for (px, py) in self.grid[r][c]:
                    d = np.sqrt((x - px)**2 + (y - py)**2)
                    if d < min_d:
                        min_d = d
        return min_d
```

### 4b. Streamline Generation with Hilbert-Curve Seeding Order

The critical question is: how do you generate a **single** continuous path that fills an arbitrary region with variable density? The approach combines three ideas:

1. **Hilbert-curve seed ordering** ensures the path visits all regions in a spatially coherent sequence (locality preservation minimizes long jumps).
2. **Local flow-field following** makes each segment follow the ETF for natural aesthetics.
3. **Tortuosity modulation** encodes local ink density along the path.

```
ALGORITHM: Single Continuous Density-Encoding Path

INPUT:
  - density[y][x]: target ink density for this channel (0-1)
  - tx[y][x], ty[y][x]: shared tangent flow field
  - pen_width: physical pen width in pixels
  - image_width, image_height

OUTPUT:
  - path: ordered list of (x, y) points forming a single continuous line

PARAMETERS:
  - base_spacing: nominal distance between parallel path segments
  - min_spacing: minimum gap (>= pen_width to prevent overlap)
  - max_amplitude: maximum wiggle amplitude (<= base_spacing/2)
  - step_size: integration step along flow field

PROCEDURE:

1. COMPUTE ADAPTIVE GRID
   Divide image into cells of size base_spacing × base_spacing.
   For each cell, compute average density.
   Generate a Hilbert curve through the cell grid at sufficient
   recursion depth to visit every cell.
   This produces a cell_visit_order: list of (cell_x, cell_y).

2. GENERATE BACKBONE PATH
   For each cell in cell_visit_order:
     Place a waypoint at the cell center.
   Smooth the waypoint sequence with Chaikin's algorithm (3 iterations)
   to produce a continuous backbone curve.

3. REFINE BACKBONE ALONG FLOW FIELD
   For each segment of the backbone:
     Resample at step_size intervals.
     At each sample, nudge position toward the local ETF direction:
       p_new = p + alpha * (tangent_etf - tangent_backbone) * step_size
     where alpha is a blending weight (0.3-0.6).

4. APPLY TORTUOSITY MODULATION
   Walk along the refined backbone. At each point p:
     Sample local density d = density[p.y][p.x]
     Compute local amplitude: A = max_amplitude * d^gamma_a
     Compute local frequency: f = base_freq + d * (max_freq - base_freq)
     Compute normal direction n = perpendicular to local tangent
     Offset: p_final = p + A * sin(accumulated_phase) * n
     accumulated_phase += 2π * f * step_size

5. ENFORCE NON-INTERSECTION
   Before committing each point p_final:
     Query occupancy grid: if min_distance(p_final) < min_spacing:
       Option A (deflect): push point outward along normal until clear
       Option B (skip): mark as travel move (pen up), advance until clear
     Else: commit point, insert into occupancy grid.

6. SMOOTH AND OUTPUT
   Apply Chaikin smoothing (2 iterations) to final path.
   Export as SVG polyline.
```

### 4c. Detailed Path Generation Implementation

```python
def generate_channel_path(density, tx, ty, pen_width, params):
    """
    Generate a single continuous non-self-intersecting path for one color channel.

    Returns: Nx2 numpy array of path points.
    """
    h, w = density.shape
    base_spacing = params['base_spacing']
    min_spacing = max(pen_width * 1.2, params['min_spacing'])
    max_amp = params['max_amplitude']
    step = params['step_size']

    # --- Step 1: Hilbert-ordered cell visitation ---
    cell_size = base_spacing
    grid_w = int(np.ceil(w / cell_size))
    grid_h = int(np.ceil(h / cell_size))
    n = max(grid_w, grid_h)
    hilbert_n = 1
    while hilbert_n < n:
        hilbert_n *= 2

    # Generate Hilbert curve visit order, filter to valid cells
    waypoints = []
    for d in range(hilbert_n * hilbert_n):
        gx, gy = d2xy(hilbert_n, d)
        if gx < grid_w and gy < grid_h:
            cx = (gx + 0.5) * cell_size
            cy = (gy + 0.5) * cell_size
            if cx < w and cy < h:
                # Skip cells with near-zero density (highlights)
                local_density = sample_density(density, cx, cy, cell_size)
                if local_density > 0.02:
                    waypoints.append([cx, cy])

    if len(waypoints) < 2:
        return np.array([])

    waypoints = np.array(waypoints)

    # --- Step 2: Smooth backbone ---
    backbone = chaikin_smooth(waypoints, iterations=3)

    # --- Step 3: Resample and align to flow field ---
    backbone = resample_polyline(backbone, step)
    backbone = nudge_toward_flow(backbone, tx, ty, alpha=0.4, step=step)

    # --- Step 4: Apply tortuosity modulation ---
    occupancy = OccupancyGrid(w, h, cell_size=min_spacing)
    final_path = []
    phase = 0.0

    for i in range(len(backbone)):
        px, py = backbone[i]
        if not (0 <= px < w and 0 <= py < h):
            continue

        d = density[int(np.clip(py, 0, h-1)), int(np.clip(px, 0, w-1))]

        # Tortuosity mapping
        amp = max_amp * (d ** 1.2)
        freq = params['base_freq'] + d * (params['max_freq'] - params['base_freq'])

        # Local normal (perpendicular to backbone tangent)
        if i < len(backbone) - 1:
            tangent = backbone[i+1] - backbone[i]
        else:
            tangent = backbone[i] - backbone[i-1]
        norm_t = np.sqrt(tangent[0]**2 + tangent[1]**2) + 1e-10
        normal = np.array([-tangent[1], tangent[0]]) / norm_t

        # Modulated position
        phase += 2 * np.pi * freq * step
        offset = amp * np.sin(phase)
        fx = px + offset * normal[0]
        fy = py + offset * normal[1]

        # --- Step 5: Non-intersection check ---
        dist = occupancy.min_distance(fx, fy, min_spacing * 2)
        if dist >= min_spacing:
            final_path.append([fx, fy])
            occupancy.insert(fx, fy)
        else:
            # Deflection: try reduced amplitude
            for scale in [0.5, 0.25, 0.0]:
                fx2 = px + offset * scale * normal[0]
                fy2 = py + offset * scale * normal[1]
                if occupancy.min_distance(fx2, fy2, min_spacing * 2) >= min_spacing:
                    final_path.append([fx2, fy2])
                    occupancy.insert(fx2, fy2)
                    break

    return np.array(final_path) if final_path else np.array([])
```

---

## 5. Multi-Channel Coordination

### 5a. Angular Separation Between Channels

To prevent visual muddiness where channels overlap, assign each channel a **base angle offset** that rotates its flow-field interpretation. This is analogous to halftone screen angles in CMYK printing (traditionally C=15°, M=75°, Y=0°, K=45°):

```python
CHANNEL_ANGLES = {
    'C': 15,   # degrees
    'M': 75,
    'Y': 0,
    'K': 45
}

def rotate_flow_field(tx, ty, angle_deg):
    """Rotate the tangent field by angle_deg."""
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    tx_r = tx * cos_t - ty * sin_t
    ty_r = tx * sin_t + ty * cos_t
    return tx_r, ty_r
```

Each channel gets its own rotated copy of the shared ETF. This ensures:
- Lines from different pens cross at angles rather than running parallel (preventing moiré).
- Each channel's path has a distinct visual "grain" that separates perceptually.

### 5b. Inter-Channel Collision Avoidance (Optional)

For the cleanest results, maintain a **global occupancy grid** shared across all channels. Process channels in order of decreasing visual importance (K first, then C, M, Y). Each subsequent channel's path avoids not only its own previous segments but also segments from earlier channels:

```python
def generate_all_channels(image_rgb, pen_colors, params):
    """Generate paths for all color channels with inter-channel awareness."""
    channels = rgb_to_cmyk(image_rgb)
    channel_names = ['K', 'C', 'M', 'Y']  # Process order: most important first
    channel_indices = [3, 0, 1, 2]  # K, C, M, Y indices in CMYK tuple

    h, w = image_rgb.shape[:2]
    tx, ty, mag = compute_etf(image_rgb)

    global_occupancy = OccupancyGrid(w, h, cell_size=params['min_spacing'])
    paths = {}

    for name, idx in zip(channel_names, channel_indices):
        density = prepare_density_field(channels[idx])
        angle = CHANNEL_ANGLES[name]
        tx_r, ty_r = rotate_flow_field(tx, ty, angle)

        path = generate_channel_path_with_global_occupancy(
            density, tx_r, ty_r,
            params['pen_width'], params,
            global_occupancy  # shared across channels
        )

        # Register this channel's points in global grid
        for pt in path:
            global_occupancy.insert(pt[0], pt[1])

        paths[name] = path

    return paths
```

**Trade-off note:** Inter-channel avoidance produces cleaner line work but reduces tonal fidelity in dense shadow regions where all four channels need high coverage. For most plotter art, allowing inter-channel crossings (but preventing intra-channel self-intersection) produces better tonal reproduction while still looking clean, since different-colored lines crossing is visually distinct from same-color tangles.

### 5c. Rendering Order

Plot channels in this order for best results:
1. **Yellow** first (lightest, least visible errors)
2. **Cyan** second
3. **Magenta** third
4. **Black** last (sharpest detail, covers alignment errors)

This mirrors offset printing convention where the darkest, highest-contrast layer goes last.

---

## 6. The Hilbert Curve Core: d2xy Implementation

The Hilbert curve provides the spatial visiting order that keeps the path locally coherent:

```python
def d2xy(n, d):
    """Convert Hilbert curve distance d to (x, y) coordinates in an n×n grid."""
    x = y = 0
    s = 1
    while s < n:
        rx = 1 if (d & 2) else 0
        ry = 1 if ((d & 1) ^ rx) else 0
        # Rotate quadrant
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d //= 4
        s *= 2
    return x, y
```

---

## 7. Smoothing and Post-Processing

### 7a. Chaikin Corner Cutting

Applied at multiple stages to eliminate the sharp 90° turns inherent in Hilbert curves:

```python
def chaikin_smooth(points, iterations=4):
    """Chaikin's corner-cutting: converges to quadratic B-spline."""
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
```

### 7b. Polyline Resampling

Ensures uniform point spacing for consistent tortuosity modulation:

```python
def resample_polyline(points, step_size):
    """Resample polyline to uniform arc-length spacing."""
    if len(points) < 2:
        return points
    diffs = np.diff(points, axis=0)
    segment_lengths = np.sqrt((diffs**2).sum(axis=1))
    cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]

    num_samples = max(2, int(total_length / step_size))
    target_dists = np.linspace(0, total_length, num_samples)

    resampled = np.zeros((num_samples, 2))
    for i, td in enumerate(target_dists):
        idx = np.searchsorted(cumulative, td, side='right') - 1
        idx = np.clip(idx, 0, len(points) - 2)
        t = (td - cumulative[idx]) / (segment_lengths[idx] + 1e-10)
        resampled[i] = points[idx] * (1 - t) + points[idx + 1] * t

    return resampled
```

---

## 8. SVG Output for Plotters

```python
import svgwrite

def export_multi_channel_svg(paths, pen_colors_hex, width, height,
                              filename, pen_width=0.5):
    """
    Export all channel paths to a single SVG with colored layers.

    paths: dict mapping channel_name -> Nx2 numpy array
    pen_colors_hex: dict mapping channel_name -> '#RRGGBB'
    """
    dwg = svgwrite.Drawing(filename, size=(f'{width}mm', f'{height}mm'),
                           viewBox=f'0 0 {width} {height}')

    plot_order = ['Y', 'C', 'M', 'K']  # lightest to darkest

    for channel_name in plot_order:
        if channel_name not in paths or len(paths[channel_name]) == 0:
            continue

        color = pen_colors_hex.get(channel_name, '#000000')
        group = dwg.add(dwg.g(
            id=f'layer_{channel_name}',
            stroke=color,
            fill='none',
            stroke_width=pen_width,
            stroke_linecap='round',
            stroke_linejoin='round'
        ))

        pts = paths[channel_name]
        point_list = [(float(p[0]), float(p[1])) for p in pts]
        group.add(dwg.polyline(point_list))

    dwg.save()
    print(f"Saved {filename}: {width}x{height}mm, {len(paths)} channels")
```

---

## 9. Parameter Tuning Guide

| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| `base_spacing` | 2.0 – 5.0 mm | Controls overall line density. Smaller = more detail, longer plot time |
| `min_spacing` | pen_width × 1.2 | Minimum gap to prevent ink bleeding between adjacent segments |
| `max_amplitude` | base_spacing × 0.45 | Maximum wiggle. Must be < base_spacing/2 to prevent crossings |
| `base_freq` | 0.02 cycles/px | Wiggle frequency in light areas (nearly straight) |
| `max_freq` | 0.15 cycles/px | Wiggle frequency in dark areas (dense scribble) |
| `gamma_a` | 1.0 – 1.5 | Amplitude gamma. Higher values push wiggle onset into darker tones |
| `gamma_s` | 1.2 – 1.8 | Spacing gamma. Controls highlight-to-midtone spacing gradient |
| `step_size` | 0.5 – 1.5 px | Path integration step. Smaller = smoother curves, more points |
| `etf_iterations` | 2 – 3 | ETF refinement passes. More = smoother flow, diminishing returns past 3 |
| `chaikin_iterations` | 3 – 5 | Smoothing passes. 4 is the sweet spot for most plotter resolutions |

---

## 10. Complete Pipeline Summary

```
INPUT: color image (RGB), pen color set, plotter parameters

┌─────────────────────────────────────────┐
│  1. COLOR DECOMPOSITION                 │
│     RGB → CMYK (or custom palette NNLS) │
│     → 4 (or N) density channels         │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│  2. PREPROCESSING (per channel)         │
│     Bilateral filter → gamma → smooth   │
│     → clean density fields              │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│  3. SHARED ORIENTATION FIELD            │
│     Structure tensor → ETF refinement   │
│     → (tx, ty) tangent field            │
│     Rotate per channel (screen angles)  │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│  4. PATH GENERATION (per channel)       │
│  a. Hilbert-order cell visitation       │
│  b. Backbone smoothing (Chaikin)        │
│  c. Flow-field alignment (ETF nudge)    │
│  d. Tortuosity modulation (sine wiggle) │
│  e. Non-intersection enforcement (grid) │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│  5. POST-PROCESSING                     │
│     Final Chaikin smooth                │
│     Polyline simplification (RDP)       │
│     SVG export with layer ordering      │
└───────────────┬─────────────────────────┘
                │
                ▼
OUTPUT: Multi-layer SVG with one polyline per pen color
```

---

## 11. Algorithmic Complexity

| Stage | Complexity | Notes |
|-------|-----------|-------|
| Color decomposition | O(W·H) per pixel, or O(1) with LUT | NNLS is expensive; use precomputed LUT |
| Density preprocessing | O(W·H) | Bilateral filter dominates at O(W·H·r²) |
| ETF computation | O(W·H·k²·I) | k=kernel radius, I=iterations |
| Hilbert curve generation | O(4ⁿ) | n = ⌈log₂(max(grid_w, grid_h))⌉ |
| Path generation per channel | O(P·G) | P=path points, G=occupancy grid queries |
| Chaikin smoothing | O(P·I) | P=points, I=iterations; doubles point count each pass |
| SVG export | O(P_total) | Linear in total point count |

For a typical 1000×1000 image with 4 channels and base_spacing=3px, expect roughly 500K path points total and 30–120 seconds of computation in Python (vectorized numpy). A C++ implementation would be 50–100× faster.

---

## 12. Extensions and Variations

**Spiral variant:** Instead of Hilbert-curve cell ordering, use an Archimedean spiral backbone per channel. This produces a more organic aesthetic but with slightly less uniform coverage near corners.

**Concentric variant:** For each channel, generate concentric offset paths from the image silhouette, with tortuosity modulation. Works exceptionally well for portraits and simple subjects.

**Adaptive pen width:** If the plotter supports variable pressure (e.g., brush pens), modulate pen pressure as a third tonal dimension alongside spacing and tortuosity, expanding dynamic range to approximately 200:1 × 4:1 × 3:1 ≈ 2400:1.

**White ink on dark paper:** Invert all density fields. The algorithm works identically — lines now represent lightness rather than darkness.

**Metallic/specialty pens:** Add channels for gold, silver, or fluorescent inks. Assign these to specific image features (e.g., specular highlights → silver channel) via luminance thresholding rather than color decomposition.
