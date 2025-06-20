@echo off
setlocal enabledelayedexpansion

:: Get the directory where this script is located
set "script_dir=%~dp0"
cd /d "%script_dir%"

echo Setting up Image Stitcher for Windows...
echo.

:: Check if conda is installed
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Miniconda...
    
    :: Set installation directory
    set "miniconda_install_dir=%USERPROFILE%\miniconda3"
    set "miniconda_installer=%TEMP%\Miniconda3-latest-Windows-x86_64.exe"
    
    :: Download Miniconda installer
    echo Downloading Miniconda installer...
    powershell -Command "Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe' -OutFile '%miniconda_installer%'"
    
    :: Install Miniconda silently
    echo Installing Miniconda to %miniconda_install_dir%...
    start /wait "" "%miniconda_installer%" /InstallationType=JustMe /RegisterPython=0 /S /D=%miniconda_install_dir%
    
    :: Initialize conda for cmd.exe
    call "%miniconda_install_dir%\Scripts\conda.exe" init cmd.exe
    
    :: Clean up installer
    del "%miniconda_installer%"
    
    set "miniconda_base_dir=%miniconda_install_dir%"
    echo Using base environment location: !miniconda_base_dir!
) else (
    echo Conda is already installed
    
    :: Get conda base directory
    for /f "tokens=*" %%i in ('conda info --base') do set "miniconda_base_dir=%%i"
    echo Using base environment location: !miniconda_base_dir!
)

:: Activate conda base environment
echo Activating conda base environment...
call "%miniconda_base_dir%\Scripts\activate.bat"

:: Create the image-stitcher environment from environment.yml
echo Creating image-stitcher conda environment...
cd /d "%script_dir%"
call conda env create --file environment.yml

:: Activate the environment and install basicpy
echo Activating image-stitcher environment...
call conda activate image-stitcher

echo Installing basicpy with pip...
pip install basicpy

echo.
echo Fixing aicsimageio dependency conflicts...
echo Installing compatible versions of dependencies...
pip install --force-reinstall "tifffile>=2021.8.30,<2023.3.15" "xmlschema<2.0.0" "lxml<5.0.0" "xarray>=0.16.1,<2023.02.0"
echo Reinstalling aicsimageio with compatible dependencies...
pip install --force-reinstall aicsimageio==4.10.0

echo.
echo ========================================
echo Setup successful!
echo.
echo To activate the conda environment, run:
echo   conda activate image-stitcher
echo.
echo Then see the README.md for example usage.
echo ========================================
echo.

pause
