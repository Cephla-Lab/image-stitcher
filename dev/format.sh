#!/bin/bash

# Format the codebase using ruff. Run this from the repository root.

set -ex

uv run ruff format .
