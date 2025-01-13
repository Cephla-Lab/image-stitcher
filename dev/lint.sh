#!/bin/bash

# Run the ruff linter and a format check. Run this script from the repository root.

set -ex

uv run ruff format --check 
uv run ruff check
