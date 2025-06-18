@echo off
setlocal enabledelayedexpansion

:: Get the directory where this script is located
set "script_dir=%~dp0"
cd /d "%script_dir%"

echo Setting up conda...

:: Get conda base directory
for /f "tokens=*" %%i in ('conda info --base 2^>nul') do set "conda_base=%%i"

if not defined conda_base (
    echo Error: Conda not found. Please run setup_windows.bat first.
    pause
    exit /b 1
)

:: Activate conda
call "%conda_base%\Scripts\activate.bat"

echo Activating the image-stitcher conda environment...
call conda activate image-stitcher

if %errorlevel% neq 0 (
    echo Error: Failed to activate image-stitcher environment.
    echo Please run setup_windows.bat first.
    pause
    exit /b 1
)

echo Running the image stitcher gui...
python -m image_stitcher.stitcher_gui

pause
