#!/bin/bash

# Autofix lint that can be autofixed. Run this from the repository root.

set -ex

if ! which ruff > /dev/null 2>&1; then
    echo "autolint fixing requires the ruff tool, please install it."
    exit 1
fi
ruff check --fix .
