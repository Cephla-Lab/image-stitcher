#!/bin/bash

set -e

readonly script_dir="$(dirname "$(realpath -- "${BASH_SOURCE[0]}")")"

sudo apt-get update && sudo apt-get install -y \
    curl wget libxml2 libxslt1.1 fontconfig libgl1 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
    libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 libxkbcommon0 \
    libxkbcommon-x11-0
cd -- "${script_dir}"

if ! command -v conda >/dev/null 2>&1
then
    echo "Installing conda..."
    readonly miniconda_install_dir="$HOME/miniconda3"
    readonly minoconda_installer="${miniconda_install_dir}/miniconda.sh"
    mkdir -p "${miniconda_install_dir}"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "${minoconda_installer}"
    bash "${minoconda_installer}" -b -u -p "${miniconda_install_dir}"

    source "${miniconda_install_dir}/bin/activate"

    conda init --all
else
    echo "conda is already installed"
fi

cd -- "${script_dir}"

conda env create --file environment.yml
conda activate image-stitcher

pip install basicpy

echo "Setup successful, run the following in your shell to activate the conda environment:"
echo "  source \"${miniconda_install_dir}/bin/activate\" && conda activate image-stitcher"
echo " Then see the README.md for example usage."
