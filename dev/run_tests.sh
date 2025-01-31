#!/bin/bash

set -ex

python -m unittest \
  image_stitcher.parameters_test \
  image_stitcher.stitcher_test
