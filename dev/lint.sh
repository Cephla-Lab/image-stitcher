#!/bin/bash

# Run the ruff linter and a format check. Run this script from the repository root.

set -ex

if ! which ruff > /dev/null 2>&1; then
    echo "linting requires the ruff tool, please install it."
    exit 1
fi

ruff format --check
ruff check
