# Image → SVG Plotter

A desktop application that converts color images into multi-channel SVG files optimised for pen plotters. Each pen color is rendered as a single continuous path whose point density reproduces the tonal values of the original image.

---

## How It Works

The algorithm decomposes an input image into CMYK ink channels and generates one stochastically-routed path per channel:

1. **Color decomposition** — RGB image is converted to CMYK density fields.
2. **Density field preparation** — bilateral filtering, gamma correction, and Gaussian smoothing clean each channel.
3. **Stochastic point distribution** — points are sampled from the density field using stratified random sampling. The image is divided into cells of size `base_spacing × base_spacing`; each cell is accepted with probability equal to its mean density and a point is placed at a random sub-pixel position within it. Dark areas receive many closely-spaced points; highlights receive few or none.
4. **Nearest-neighbor TSP routing** — the sampled points are connected into a single continuous path using a nearest-neighbor TSP heuristic. A directional penalty discourages the solver from continuing straight ahead, naturally breaking up long linear runs and introducing angular variety without sacrificing continuity.
5. **Smoothing** — Chaikin corner-cutting rounds the tour into a plotter-friendly curve.
6. **SVG export** — paths are exported as coloured polylines in a single multi-layer SVG, ordered Y → C → M → K for optimal plotter results.

---

## Features

- Dark-themed Tkinter GUI — no browser required
- Live input image preview and rendered output preview
- Per-channel path rendering (Y, C, M, K) in the preview canvas
- Adjustable parameters: base spacing, pen width, direction weight, smoothing passes
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
| Base spacing (px) | 3.0 | Cell size for stratified sampling. Smaller = more points, more detail, longer routing |
| Pen width (px) | 1.5 | Physical pen width; used for SVG stroke width |
| Direction weight | 1.0 | TSP directional penalty. `0` = pure nearest-neighbour (may create linear runs). `~1` = moderate angular variety. `2` = strong turn preference |
| Smoothing passes | 3 | Chaikin corner-cutting iterations applied to the final path |

---

## Project Structure

```
Plotter/
├── main.py                      # Entry point; launches the Tkinter GUI
├── requirements.txt             # Python dependencies
├── build.bat                    # Windows build script (PyInstaller)
├── plotter.spec                 # PyInstaller spec file
├── color_plotter_algorithm.md   # Background algorithmic notes
├── algorithm/                   # Core algorithm package
│   ├── __init__.py
│   ├── color_decompose.py       # RGB → CMYK conversion
│   ├── density_field.py         # Density field preprocessing
│   ├── path_generator.py        # Stochastic sampling + NN-TSP routing
│   ├── smoothing.py             # Chaikin smoothing & polyline resampling
│   └── svg_export.py            # Multi-channel SVG export
└── ui/
    └── app.py                   # Tkinter UI (PlotterApp)
```
