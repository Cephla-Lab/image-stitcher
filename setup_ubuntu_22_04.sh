#!/bin/bash

set -e

sudo apt-get update && sudo apt-get install -y \
    curl libxml2 libxslt1.1 fontconfig libgl1 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
    libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 libxkbcommon0 \
    libxkbcommon-x11-0
cd -- "$(dirname "$(realpath -- "${BASH_SOURCE[0]}")")"
./dev/ensure_uv.sh
