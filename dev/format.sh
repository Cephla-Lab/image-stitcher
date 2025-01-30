#!/bin/bash

# Format the codebase using ruff. Run this from the repository root.

set -ex

if ! which ruff > /dev/null 2>&1; then
    echo "formatting requires the ruff tool, please install it."
    exit 1
fi

ruff format .
