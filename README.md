# image-stitcher

image stitching for microscopy with GPU acceleration support.

## Setup from Scratch

### Ubuntu 22.04
```bash
./setup_ubuntu_22_04.sh
```
Installs conda (if needed), creates `image-stitcher` environment, and installs dependencies.

### Ubuntu 22.04 with CUDA (Optional)
```bash
./setup_cuda_22.04.sh
```
Adds CuPy for optimal NVIDIA GPU performance (compatible with Driver 535.x).

### Windows 11

**Note:** If you haven't enabled script execution in PowerShell, you may need to run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then run the setup:
```powershell
.\setup_windows_11.ps1
```
Installs conda (if needed), creates `image-stitcher` environment, and installs dependencies.

### Windows 11 with CUDA (Optional)
```powershell
.\setup_cuda_windows_11.ps1
```
Adds CuPy for optimal NVIDIA GPU performance.

### Desktop Shortcut (Optional)
```bash
python create_desktop_shortcut.py
```
Creates a desktop launcher that auto-activates the conda environment.

## Launch

### GUI
```bash
./run_gui
```
Or manually:
```bash
conda activate image-stitcher
python -m image_stitcher.stitcher_gui
```

#### GPU Acceleration in GUI

The GUI includes a **Compute Backend** dropdown that allows you to select the tensor backend:

- **Auto (Recommended)** - Automatically selects the best available backend
- **NumPy (CPU)** - CPU-only processing using NumPy
- **PyTorch (GPU/CPU)** - Uses PyTorch with GPU acceleration if available
- **CuPy (NVIDIA GPU)** - Maximum performance with NVIDIA GPU acceleration

The dropdown shows the actual backend status (GPU/CPU).. Backend selection affects the registration process performance significantly.

## Devtools

This repository is set up with [ruff](https://docs.astral.sh/ruff/) for linting
and formatting, and [mypy](https://mypy.readthedocs.io) for type checking. The
shell scripts in the dev directory can be used to invoke these tools and should
be run from the repository root.

You can install them with `pip install mypy ruff`
