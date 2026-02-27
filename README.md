# Image → SVG Plotter

A desktop application that converts color images into multi-channel SVG files optimised for pen plotters. Each pen color is rendered as a single continuous, non-self-intersecting path whose local density and tortuosity reproduce the tonal values of the original image.

---

## How It Works

The algorithm decomposes an input image into CMYK ink channels and generates one flow-guided, density-modulated streamline per channel:

1. **Color decomposition** — RGB image is converted to CMYK density fields (or an arbitrary N-pen palette via non-negative least squares).
2. **Density field preparation** — bilateral filtering, gamma correction, and Gaussian smoothing clean each channel.
3. **Edge Tangent Flow (ETF)** — a shared orientation field is computed from the image so lines from all pens flow in harmonious directions. Each channel receives a rotated copy (C=15°, M=75°, Y=0°, K=45°) to prevent moiré patterns.
4. **Path generation** — a Hilbert-curve visitation order seeds a backbone path through the image. The backbone is smoothed (Chaikin), aligned to the ETF, and then modulated with a sine wiggle whose amplitude and frequency encode local ink density.
5. **Non-intersection enforcement** — a spatial occupancy grid prevents each channel's path from crossing itself.
6. **SVG export** — paths are exported as coloured polylines in a single multi-layer SVG, ordered Y → C → M → K for optimal plotter results.

See [`color_plotter_algorithm.md`](color_plotter_algorithm.md) for the full algorithmic specification including all pseudocode and implementation details.

---

## Features

- Dark-themed Tkinter GUI — no browser required
- Live input image preview and rendered output preview
- Per-channel path rendering (Y, C, M, K) in the preview canvas
- Adjustable parameters: base spacing, pen width, wiggle amplitude/frequency, ETF iterations, smoothing passes
- Processing runs in a background thread with a live progress bar
- One-click SVG export
- Portable Windows `.exe` via PyInstaller — no Python installation needed on the target machine

---

## Requirements

- Python 3.10+
- Dependencies (installed automatically by `build.bat` or manually):

```
numpy
opencv-python
scipy
svgwrite
Pillow
```

---

## Running from Source

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the application
python main.py
```

---

## Building the Windows Portable App

Run the provided batch script on a Windows machine with Python 3.10+ on `PATH`:

```bat
build.bat
```

This script:
1. Installs all Python dependencies via `pip`
2. Installs PyInstaller
3. Builds the application using `plotter.spec`

The distributable output will be in `dist\PlotterApp\`. Run `dist\PlotterApp\PlotterApp.exe` — no Python installation required on the target machine.

---

## Using the Application

1. **Open Image** — click *Open Image…* and choose a PNG, JPEG, BMP, TIFF, or WebP file.
2. **Adjust settings** — use the right-hand panel to tune the output (see [Parameters](#parameters) below).
3. **Convert** — click *▶ Convert*. Progress is shown in the status bar. A live preview of the generated paths appears in the *Output* tab when complete.
4. **Save SVG** — click *Save SVG…* to export the multi-layer SVG file.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Max dimension (px) | 400 | Image is downscaled so its longest side is this many pixels before processing |
| Base spacing (px) | 3.0 | Nominal distance between parallel path segments. Smaller = more detail, longer processing |
| Pen width (px) | 1.5 | Physical pen width; determines minimum gap between segments |
| Max amplitude (px) | 1.2 | Maximum sine wiggle amplitude in dark areas. Must be < base spacing / 2 |
| Step size (px) | 1.0 | Path integration step. Smaller = smoother curves, more points |
| ETF iterations | 3 | Edge Tangent Flow refinement passes. More = smoother flow |
| Smoothing passes | 3 | Chaikin corner-cutting iterations applied to paths |
| Base frequency | 0.02 | Wiggle frequency in light areas (nearly straight lines) |
| Max frequency | 0.15 | Wiggle frequency in dark areas (dense scribble) |

---

## Project Structure

```
Plotter/
├── main.py                      # Entry point; launches the Tkinter GUI
├── requirements.txt             # Python dependencies
├── build.bat                    # Windows build script (PyInstaller)
├── plotter.spec                 # PyInstaller spec file
├── color_plotter_algorithm.md   # Full algorithm specification
├── algorithm/                   # Core algorithm package
│   ├── __init__.py
│   ├── color_decompose.py       # RGB → CMYK conversion
│   ├── density_field.py         # Density field preprocessing
│   ├── etf.py                   # Edge Tangent Flow computation
│   ├── hilbert.py               # Hilbert curve (d2xy)
│   ├── occupancy.py             # Spatial occupancy grid
│   ├── path_generator.py        # Main path generation pipeline
│   ├── smoothing.py             # Chaikin smoothing & polyline resampling
│   └── svg_export.py            # Multi-channel SVG export
└── ui/
    └── app.py                   # Tkinter UI (PlotterApp)
```
