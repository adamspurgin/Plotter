# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Image → SVG Plotter
# Run with: pyinstaller plotter.spec

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[str(Path('.'))]  ,
    binaries=[],
    datas=[],
    hiddenimports=[
        'numpy',
        'numpy.core._multiarray_umath',
        'cv2',
        'scipy',
        'scipy.optimize',
        'scipy.optimize._nnls',
        'svgwrite',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'algorithm',
        'algorithm.hilbert',
        'algorithm.occupancy',
        'algorithm.smoothing',
        'algorithm.color_decompose',
        'algorithm.density_field',
        'algorithm.etf',
        'algorithm.path_generator',
        'algorithm.svg_export',
        'ui',
        'ui.app',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'pandas',
        'IPython',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PlotterApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # No console window — pure GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,              # Replace with 'icon.ico' if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PlotterApp',      # Output folder: dist/PlotterApp/
)
