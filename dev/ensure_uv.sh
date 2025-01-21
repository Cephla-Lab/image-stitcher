#!/bin/bash

set -e

if ! command -v uv 2>&1 >/dev/null
then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv is already installed"
fi
