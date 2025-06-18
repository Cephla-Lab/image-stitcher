# GPU Setup for CUDA Acceleration

This project uses CuPy for GPU-accelerated image processing. Follow these steps to set up CUDA and CuPy correctly.

## Prerequisites

1. **NVIDIA GPU**: Ensure you have a CUDA-compatible NVIDIA GPU
2. **NVIDIA Driver**: Install a recent NVIDIA driver for your GPU

Check your setup:
```bash
nvidia-smi
```

## Version Compatibility

**Critical Rule: CuPy package must match your installed CUDA Toolkit version**

| CUDA Toolkit Version | CuPy Package Required | Verification Command |
|---------------------|----------------------|---------------------|
| 11.0 - 11.8 | `cupy-cuda11x` | `nvcc --version` |
| 12.0 - 12.6 | `cupy-cuda12x` | `nvcc --version` |

**NVIDIA Driver Requirements:**
- For CUDA 11.x: Driver ≥ 450.80
- For CUDA 12.x: Driver ≥ 525.60

**Note**: Your NVIDIA driver version determines the maximum CUDA runtime version supported, but the actual match that matters is between your installed CUDA Toolkit and CuPy package.

## Installation Steps

### 1. Check Your Current Setup
```bash
# Check NVIDIA driver and maximum CUDA support
nvidia-smi

# Check installed CUDA Toolkit (if any)
nvcc --version
```

### 2. Install CUDA Toolkit 12.4 (Recommended)
```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update

# Install CUDA toolkit
sudo apt install cuda-toolkit-12-4
```

### 3. Set Environment Variables
Add these to your `~/.bashrc` or `~/.zshrc`:
```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

Reload your shell:
```bash
source ~/.bashrc
```

### 4. Install CuPy
```bash
# For CUDA 12.x (matches CUDA Toolkit 12.4)
pip install cupy-cuda12x

# For CUDA 11.x (if you have CUDA Toolkit 11.x installed)
pip install cupy-cuda11x
```

### 5. Verify Installation
```bash
# Check CUDA compiler version
nvcc --version

# Test CuPy installation
python -c "import cupy; print('CuPy version:', cupy.__version__)"
python -c "import cupy; print('CUDA runtime:', cupy.cuda.runtime.runtimeGetVersion())"

# Test FFT functionality (critical for this project)
python -c "import cupy; import cupy.fft; x = cupy.random.random((64, 64)); y = cupy.fft.fft2(x); print('GPU FFT test passed')"
```

## Troubleshooting

**Error: `cannot open source file "cuda_fp16.h"`**
- **Cause**: CuPy package doesn't match your CUDA Toolkit version
- **Solution**: Check `nvcc --version` and install the matching CuPy package

**Error: `CUFFT_INTERNAL_ERROR`**
- **Cause**: Version mismatch between CUDA Toolkit and CuPy
- **Solution**: Ensure CUDA Toolkit and CuPy versions are compatible (see table above)

**Error: `NVRTC_ERROR_COMPILATION`**
- **Cause**: Missing CUDA development headers
- **Solution**: Install complete CUDA Toolkit (not just runtime)

**Out of Memory Errors**
- Monitor GPU memory: `nvidia-smi`
- The code includes automatic memory cleanup between regions

## Example Working Configuration

```bash
# Successful setup example:
$ nvidia-smi
# Driver Version: 570.133.07    CUDA Version: 12.8

$ nvcc --version  
# Cuda compilation tools, release 12.4, V12.4.131

$ python -c "import cupy; print(cupy.__version__)"
# 13.4.1

$ pip list | grep cupy
# cupy-cuda12x    13.4.1
```

## Alternative: CPU-Only Mode

If GPU setup is problematic, you can modify the code to run on CPU only by replacing CuPy operations with NumPy equivalents, though this will be significantly slower for large image datasets.
