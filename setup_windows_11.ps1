$ErrorActionPreference = "Stop"

$script_dir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition

Set-Location -Path $script_dir

$conda_exists = $null -ne (Get-Command conda -ErrorAction SilentlyContinue)

if (-not $conda_exists) {
    Write-Host "Installing conda..."
    $miniconda_install_dir = Join-Path $env:USERPROFILE "miniconda3"
    $miniconda_installer = Join-Path $miniconda_install_dir "miniconda.exe"
    
    New-Item -ItemType Directory -Force -Path $miniconda_install_dir | Out-Null
    
    (New-Object System.Net.WebClient).DownloadFile(
        "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe",
        $miniconda_installer
    )
    
    Start-Process -FilePath $miniconda_installer -ArgumentList "/InstallationType=JustMe", "/RegisterPython=0", "/AddToPath=0", "/S", "/D=$miniconda_install_dir" -Wait -NoNewWindow
    
    & "$miniconda_install_dir\Scripts\conda.exe" init powershell
    
    $miniconda_base_dir = $miniconda_install_dir
    Write-Host "Using base environment location: '$miniconda_base_dir'"
} else {
    $miniconda_base_dir = & conda info --base
    Write-Host "Conda is already installed, using base environment location: '$miniconda_base_dir'"
}

Set-Location -Path $script_dir

$conda_exe = Join-Path $miniconda_base_dir "Scripts\conda.exe"

& $conda_exe env create --file environment.yml

$env_pip = Join-Path $miniconda_base_dir "envs\image-stitcher\Scripts\pip.exe"

& $env_pip install basicpy

Write-Host ""
Write-Host "Setup successful, run the following in your shell to activate the conda environment:"
Write-Host "  conda activate image-stitcher"
Write-Host " Then see the README.md for example usage."
