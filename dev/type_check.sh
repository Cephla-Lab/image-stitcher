#!/bin/bash

# Do type checking via mypy. Run this script from the repository root.

set -ex

if ! which mypy > /dev/null 2>&1; then
    echo "type checking requires the mypy tool, please install it."
    exit 1
fi

mypy image_stitcher
