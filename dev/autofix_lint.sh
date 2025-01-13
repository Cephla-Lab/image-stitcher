#!/bin/bash

# Autofix lint that can be autofixed. Run this from the repository root.

set -ex

uv run ruff check --fix .
