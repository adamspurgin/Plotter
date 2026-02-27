@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  Build script for Image → SVG Plotter (Windows portable app)
REM  Requirements: Python 3.10+ installed and on PATH
REM ─────────────────────────────────────────────────────────────────────────────

echo [1/3] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

echo.
echo [2/3] Installing PyInstaller...
pip install pyinstaller
if errorlevel 1 (
    echo ERROR: Could not install PyInstaller.
    pause
    exit /b 1
)

echo.
echo [3/3] Building portable application...
pyinstaller plotter.spec --clean --noconfirm
if errorlevel 1 (
    echo ERROR: PyInstaller build failed.
    pause
    exit /b 1
)

echo.
echo ─────────────────────────────────────────────────────────────────
echo  BUILD COMPLETE
echo  Distributable folder: dist\PlotterApp\
echo  Run: dist\PlotterApp\PlotterApp.exe
echo ─────────────────────────────────────────────────────────────────
pause
