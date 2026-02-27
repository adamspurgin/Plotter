"""
Entry point for the Image → SVG Plotter portable application.
"""
import sys
import os

# When running as a PyInstaller bundle, sys._MEIPASS contains the temp dir
# where all packaged files are extracted. We add it to the path so that
# our 'algorithm' and 'ui' packages are importable.
if getattr(sys, 'frozen', False):
    _base = sys._MEIPASS
else:
    _base = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, _base)

import tkinter as tk
from ui.app import PlotterApp


def main():
    root = tk.Tk()
    root.withdraw()          # hide briefly while building UI
    app = PlotterApp(root)   # noqa: F841
    root.deiconify()
    root.mainloop()


if __name__ == '__main__':
    main()
