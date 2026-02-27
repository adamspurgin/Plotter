"""
Tkinter UI for the Image → SVG Plotter application.
"""
import os
import sys
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

# Allow running directly from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm import generate_all_channels, export_multi_channel_svg
from algorithm.path_generator import CHANNEL_COLORS, CHANNEL_ORDER

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette for the UI
# ──────────────────────────────────────────────────────────────────────────────
BG         = '#1e1e2e'
PANEL_BG   = '#2a2a3e'
ACCENT     = '#7c6af7'
ACCENT_FG  = '#ffffff'
TEXT       = '#cdd6f4'
MUTED      = '#6e6c7e'
ENTRY_BG   = '#313244'
CANVAS_BG  = '#11111b'

PREVIEW_W  = 480
PREVIEW_H  = 400


class PlotterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Image → SVG Plotter")
        self.root.configure(bg=BG)
        self.root.minsize(960, 640)

        # State
        self.image_path: str | None = None
        self.image_pil: Image.Image | None = None
        self.image_rgb: np.ndarray | None = None
        self.result_paths: dict | None = None
        self.result_size: tuple[int, int] | None = None  # (w, h)
        self.is_processing = False
        self._progress_q: queue.Queue = queue.Queue()
        self._photo_input: ImageTk.PhotoImage | None = None
        self._photo_thumb: ImageTk.PhotoImage | None = None

        self._build_styles()
        self._build_ui()
        self._poll_progress()

    # ── Styles ────────────────────────────────────────────────────────────────

    def _build_styles(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('TFrame',       background=BG)
        style.configure('Panel.TFrame', background=PANEL_BG)
        style.configure('TLabel',       background=BG,       foreground=TEXT,
                        font=('Segoe UI', 10))
        style.configure('Panel.TLabel', background=PANEL_BG, foreground=TEXT,
                        font=('Segoe UI', 10))
        style.configure('Head.TLabel',  background=PANEL_BG, foreground=ACCENT,
                        font=('Segoe UI', 11, 'bold'))
        style.configure('Status.TLabel', background=BG,      foreground=MUTED,
                        font=('Segoe UI', 9))
        style.configure('TButton',      font=('Segoe UI', 10),
                        padding=(10, 5))
        style.configure('Accent.TButton', background=ACCENT, foreground=ACCENT_FG,
                        font=('Segoe UI', 10, 'bold'), padding=(14, 6))
        style.map('Accent.TButton',
                  background=[('active', '#9580ff'), ('disabled', MUTED)],
                  foreground=[('disabled', '#888888')])
        style.configure('TScale',       background=PANEL_BG, troughcolor=ENTRY_BG,
                        sliderthickness=14)
        style.configure('TProgressbar', troughcolor=ENTRY_BG,
                        background=ACCENT, thickness=8)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────────────────────
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(10, 4))

        ttk.Label(top, text="Image → SVG Plotter",
                  font=('Segoe UI', 14, 'bold'),
                  foreground=ACCENT, background=BG).pack(side=tk.LEFT)

        self._btn_save = ttk.Button(top, text="Save SVG…",
                                    command=self._save_svg, state=tk.DISABLED)
        self._btn_save.pack(side=tk.RIGHT, padx=(4, 0))

        self._btn_convert = ttk.Button(top, text="▶  Convert",
                                       command=self._start_convert,
                                       style='Accent.TButton', state=tk.DISABLED)
        self._btn_convert.pack(side=tk.RIGHT, padx=(4, 0))

        self._btn_open = ttk.Button(top, text="Open Image…",
                                    command=self._open_image)
        self._btn_open.pack(side=tk.RIGHT, padx=(4, 0))

        # ── Main area (preview | settings) ───────────────────────────────────
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, minsize=280, weight=0)
        main.rowconfigure(0, weight=1)

        # ── Preview pane ──────────────────────────────────────────────────────
        preview_frame = ttk.Frame(main, style='Panel.TFrame')
        preview_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 6))
        preview_frame.rowconfigure(1, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        tab_row = ttk.Frame(preview_frame, style='Panel.TFrame')
        tab_row.grid(row=0, column=0, sticky='ew', padx=4, pady=(6, 2))

        self._tab_var = tk.StringVar(value='input')
        for val, label in (('input', 'Input'), ('output', 'Output')):
            rb = tk.Radiobutton(
                tab_row, text=label, variable=self._tab_var, value=val,
                command=self._refresh_preview,
                bg=PANEL_BG, fg=TEXT, selectcolor=ACCENT,
                activebackground=PANEL_BG, activeforeground=TEXT,
                relief='flat', font=('Segoe UI', 10, 'bold'),
                indicatoron=False, padx=10, pady=3,
                bd=0, highlightthickness=0,
            )
            rb.pack(side=tk.LEFT, padx=2)

        self._canvas = tk.Canvas(preview_frame, bg=CANVAS_BG,
                                 highlightthickness=0,
                                 width=PREVIEW_W, height=PREVIEW_H)
        self._canvas.grid(row=1, column=0, sticky='nsew', padx=6, pady=6)
        self._canvas.bind('<Configure>', lambda e: self._refresh_preview())

        self._hint = self._canvas.create_text(
            PREVIEW_W // 2, PREVIEW_H // 2,
            text="Open an image to begin",
            fill=MUTED, font=('Segoe UI', 12),
        )

        # ── Settings pane ─────────────────────────────────────────────────────
        settings = ttk.Frame(main, style='Panel.TFrame')
        settings.grid(row=0, column=1, sticky='nsew')

        self._build_settings(settings)

        # ── Bottom status bar ─────────────────────────────────────────────────
        bottom = ttk.Frame(self.root)
        bottom.pack(fill=tk.X, padx=12, pady=(4, 8))

        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_bar = ttk.Progressbar(
            bottom, variable=self._progress_var,
            maximum=100.0, length=200,
        )
        self._progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self._status_var = tk.StringVar(value="Ready — open an image to begin.")
        ttk.Label(bottom, textvariable=self._status_var,
                  style='Status.TLabel').pack(side=tk.LEFT)

    def _build_settings(self, parent: ttk.Frame):
        parent.columnconfigure(0, weight=1)

        def section(text):
            ttk.Label(parent, text=text, style='Head.TLabel').pack(
                fill=tk.X, padx=10, pady=(14, 2))
            sep = tk.Frame(parent, bg=ACCENT, height=1)
            sep.pack(fill=tk.X, padx=10, pady=(0, 6))

        def slider_row(label, var, from_, to, resolution, fmt='{:.1f}'):
            row = ttk.Frame(parent, style='Panel.TFrame')
            row.pack(fill=tk.X, padx=10, pady=2)
            row.columnconfigure(1, weight=1)

            lbl = ttk.Label(row, text=label, style='Panel.TLabel', width=18)
            lbl.grid(row=0, column=0, sticky='w')

            val_lbl = ttk.Label(row, width=6, style='Panel.TLabel',
                                 anchor='e')
            val_lbl.grid(row=0, column=2, sticky='e', padx=(4, 0))

            def on_change(*_):
                val_lbl.config(text=fmt.format(var.get()))

            sc = ttk.Scale(row, from_=from_, to=to, variable=var,
                           orient=tk.HORIZONTAL, command=on_change)
            sc.grid(row=0, column=1, sticky='ew', padx=(4, 0))
            on_change()
            return sc

        # ── Image ──────────────────────────────────────────────────────────
        section("Image")
        self._max_size_var = tk.IntVar(value=400)
        slider_row("Max dimension (px)", self._max_size_var,
                   200, 1000, 50, fmt='{:.0f}')

        # ── Path style ─────────────────────────────────────────────────────
        section("Path Style")
        self._spacing_var = tk.DoubleVar(value=3.0)
        slider_row("Base spacing (px)", self._spacing_var, 2.0, 8.0, 0.5)

        self._pen_width_var = tk.DoubleVar(value=1.5)
        slider_row("Pen width (px)", self._pen_width_var, 0.5, 5.0, 0.5)

        self._amplitude_var = tk.DoubleVar(value=1.2)
        slider_row("Max amplitude (px)", self._amplitude_var, 0.3, 4.0, 0.1)

        self._step_var = tk.DoubleVar(value=1.0)
        slider_row("Step size (px)", self._step_var, 0.5, 2.0, 0.25)

        # ── Advanced ───────────────────────────────────────────────────────
        section("Advanced")
        self._etf_iter_var = tk.IntVar(value=3)
        slider_row("ETF iterations", self._etf_iter_var, 1, 3, 1, fmt='{:.0f}')

        self._chaikin_var = tk.IntVar(value=3)
        slider_row("Smoothing passes", self._chaikin_var, 2, 5, 1, fmt='{:.0f}')

        self._base_freq_var = tk.DoubleVar(value=0.02)
        slider_row("Base frequency", self._base_freq_var,
                   0.005, 0.05, 0.005, fmt='{:.3f}')

        self._max_freq_var = tk.DoubleVar(value=0.15)
        slider_row("Max frequency", self._max_freq_var,
                   0.05, 0.30, 0.01, fmt='{:.3f}')

        # Spacer
        ttk.Frame(parent, style='Panel.TFrame').pack(fill=tk.BOTH, expand=True)

        # Time estimate
        self._time_hint = ttk.Label(
            parent,
            text="",
            style='Panel.TLabel',
            foreground=MUTED,
            wraplength=240,
            justify='center',
        )
        self._time_hint.pack(pady=(0, 8))

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            img = Image.open(path).convert('RGB')
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open image:\n{exc}")
            return

        self.image_path = path
        self.image_pil = img
        self.result_paths = None
        self.result_size = None
        self._btn_save.config(state=tk.DISABLED)
        self._btn_convert.config(state=tk.NORMAL)
        self._status_var.set(f"Loaded: {os.path.basename(path)}  "
                             f"({img.width} × {img.height} px)")
        self._tab_var.set('input')
        self._refresh_preview()
        self._update_time_hint()

    def _save_svg(self):
        if not self.result_paths or not self.result_size:
            return
        path = filedialog.asksaveasfilename(
            title="Save SVG",
            defaultextension=".svg",
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
        )
        if not path:
            return
        w, h = self.result_size
        try:
            total = export_multi_channel_svg(
                self.result_paths, w, h, path,
                pen_width=self._pen_width_var.get(),
            )
            self._status_var.set(f"Saved {os.path.basename(path)}  "
                                 f"({total:,} total path points)")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))

    # ── Conversion ────────────────────────────────────────────────────────────

    def _start_convert(self):
        if self.is_processing or self.image_pil is None:
            return

        max_size = self._max_size_var.get()
        img = self.image_pil.copy()
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        self.image_rgb = np.asarray(img, dtype=np.uint8)
        w, h = img.size
        self.result_size = (w, h)

        spacing = self._spacing_var.get()
        params = {
            'base_spacing':     spacing,
            'min_spacing':      self._pen_width_var.get() * 1.2,
            'max_amplitude':    self._amplitude_var.get(),
            'step_size':        self._step_var.get(),
            'base_freq':        self._base_freq_var.get(),
            'max_freq':         self._max_freq_var.get(),
            'pen_width':        self._pen_width_var.get(),
            'etf_iterations':   self._etf_iter_var.get(),
            'chaikin_iterations': self._chaikin_var.get(),
        }

        self.is_processing = True
        self.result_paths = None
        self._btn_convert.config(state=tk.DISABLED)
        self._btn_save.config(state=tk.DISABLED)
        self._btn_open.config(state=tk.DISABLED)
        self._progress_var.set(0.0)
        self._status_var.set("Starting…")

        def worker():
            def cb(frac, msg=""):
                self._progress_q.put(('progress', frac, msg))

            try:
                paths = generate_all_channels(
                    self.image_rgb, params, progress_callback=cb)
                self._progress_q.put(('done', paths))
            except Exception as exc:
                self._progress_q.put(('error', str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_progress(self):
        try:
            while True:
                msg = self._progress_q.get_nowait()
                kind = msg[0]
                if kind == 'progress':
                    _, frac, status = msg
                    self._progress_var.set(frac * 100.0)
                    self._status_var.set(status)
                elif kind == 'done':
                    _, paths = msg
                    self.result_paths = paths
                    self.is_processing = False
                    self._progress_var.set(100.0)
                    total = sum(len(p) for p in paths.values() if len(p) > 0)
                    self._status_var.set(
                        f"Done — {total:,} path points across "
                        f"{sum(1 for p in paths.values() if len(p) > 0)} channels. "
                        f"Click 'Save SVG…' to export.")
                    self._btn_convert.config(state=tk.NORMAL)
                    self._btn_save.config(state=tk.NORMAL)
                    self._btn_open.config(state=tk.NORMAL)
                    self._tab_var.set('output')
                    self._refresh_preview()
                elif kind == 'error':
                    _, err = msg
                    self.is_processing = False
                    self._status_var.set(f"Error: {err}")
                    self._btn_convert.config(state=tk.NORMAL)
                    self._btn_open.config(state=tk.NORMAL)
                    messagebox.showerror("Conversion Error", err)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_progress)

    # ── Preview ───────────────────────────────────────────────────────────────

    def _refresh_preview(self):
        tab = self._tab_var.get()
        self._canvas.delete('all')
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = PREVIEW_W, PREVIEW_H

        if tab == 'input':
            self._draw_input_preview(cw, ch)
        else:
            self._draw_output_preview(cw, ch)

    def _draw_input_preview(self, cw, ch):
        if self.image_pil is None:
            self._canvas.create_text(
                cw // 2, ch // 2,
                text="Open an image to begin",
                fill=MUTED, font=('Segoe UI', 12),
            )
            return

        thumb = self.image_pil.copy()
        thumb.thumbnail((cw - 12, ch - 12), Image.LANCZOS)
        self._photo_input = ImageTk.PhotoImage(thumb)
        self._canvas.create_image(
            cw // 2, ch // 2,
            anchor=tk.CENTER,
            image=self._photo_input,
        )

    def _draw_output_preview(self, cw, ch):
        if not self.result_paths or not self.result_size:
            self._canvas.create_text(
                cw // 2, ch // 2,
                text="No output yet — click Convert",
                fill=MUTED, font=('Segoe UI', 12),
            )
            return

        img_w, img_h = self.result_size
        scale = min((cw - 12) / img_w, (ch - 12) / img_h)
        ox = (cw - img_w * scale) / 2
        oy = (ch - img_h * scale) / 2

        # Draw border
        self._canvas.create_rectangle(
            ox, oy, ox + img_w * scale, oy + img_h * scale,
            outline=MUTED, width=1,
        )

        # Draw each channel path (sub-sampled for speed)
        subsample = max(1, int(1 / scale * 0.5) + 1)

        for name in ('Y', 'C', 'M', 'K'):
            pts = self.result_paths.get(name)
            if pts is None or len(pts) < 2:
                continue
            color = CHANNEL_COLORS[name]
            # Sub-sample points
            pts_sub = pts[::subsample]
            coords = []
            for p in pts_sub:
                coords.append(ox + p[0] * scale)
                coords.append(oy + p[1] * scale)
            if len(coords) >= 4:
                self._canvas.create_line(
                    coords, fill=color, width=1,
                    smooth=False, capstyle=tk.ROUND,
                )

        total = sum(len(p) for p in self.result_paths.values() if len(p) > 0)
        self._canvas.create_text(
            4, ch - 4, anchor='sw',
            text=f"{total:,} pts",
            fill=MUTED, font=('Segoe UI', 8),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_time_hint(self):
        if self.image_pil is None:
            return
        max_size = self._max_size_var.get()
        w = min(self.image_pil.width, max_size)
        h = min(self.image_pil.height, max_size)
        # Rough estimate: ~0.05 s/Kpx for ETF + ~0.1 s/Kpx for path gen
        px_k = (w * h) / 1000.0
        secs = int(px_k * 0.5)
        if secs < 5:
            hint = "< 5 s"
        elif secs < 60:
            hint = f"~{secs} s"
        else:
            hint = f"~{secs // 60} min"
        spacing = self._spacing_var.get()
        self._time_hint.config(
            text=f"Processing {w}×{h} px at {spacing:.1f} px spacing\n"
                 f"Estimated time: {hint}")
