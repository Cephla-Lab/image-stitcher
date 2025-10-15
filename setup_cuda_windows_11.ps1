$ErrorActionPreference = "Stop"

# Update not needed on Windows - use Windows Update or GeForce Experience
# Install NVIDIA driver from https://www.nvidia.com/Download/index.aspx

Set-Location -Path (Join-Path $env:USERPROFILE "Downloads")

$cuda_installer = "cuda_12.1.0_windows_network.exe"
(New-Object System.Net.WebClient).DownloadFile("https://developer.download.nvidia.com/compute/cuda/12.1.0/network_installers/cuda_12.1.0_windows_network.exe", $cuda_installer)

Start-Process -FilePath ".\$cuda_installer" -ArgumentList "-s" -Wait -NoNewWindow

pip install cuda-python
pip install cupy-cuda12x
pip install torch torchvision torchaudio
