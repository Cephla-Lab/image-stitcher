#!/bin/bash

# Do type checking via mypy. Run this script from the repository root.

set -ex

uv run mypy image_stitcher
