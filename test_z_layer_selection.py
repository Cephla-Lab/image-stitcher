#!/usr/bin/env python3
"""Test script for z-layer selection functionality.

This script demonstrates how to use the new z-layer selection feature
to stitch only specific z-layers from a stack.
"""

import logging
import sys
from pathlib import Path

from image_stitcher.parameters import StitchingParameters
from image_stitcher.stitcher import Stitcher


def test_z_layer_selection(input_folder: str):
    """Test z-layer selection with different strategies."""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Test 1: Stitch with middle layer selection (now the default)
    logging.info("=" * 60)
    logging.info("Test 1: Stitching with middle z-layer selection (default)")
    logging.info("=" * 60)

    params_middle = StitchingParameters(
        input_folder=input_folder,
        # z_layer_selection defaults to "middle" now
        verbose=True,
    )

    stitcher_middle = Stitcher(params_middle)

    # Print some info about what will be stitched
    logging.info(
        f"Total z-layers in acquisition: {stitcher_middle.computed_parameters.num_z}"
    )
    logging.info(f"Z-layer selection strategy: {params_middle.z_layer_selection}")
    middle_idx = stitcher_middle.computed_parameters.num_z // 2
    logging.info(f"Middle layer index: {middle_idx}")

    # You can run the stitching with:
    # stitcher_middle.run()

    # Test 2: Stitch all layers
    logging.info("\n" + "=" * 60)
    logging.info("Test 2: Stitching with all z-layers")
    logging.info("=" * 60)

    params_all = StitchingParameters(
        input_folder=input_folder,
        z_layer_selection="all",  # Explicitly request all z-layers
        verbose=True,
    )

    stitcher_all = Stitcher(params_all)

    logging.info(
        f"Total z-layers in acquisition: {stitcher_all.computed_parameters.num_z}"
    )
    logging.info(f"Z-layer selection strategy: {params_all.z_layer_selection}")

    # You can run the stitching with:
    # stitcher_all.run()

    # Test 3: Stitch a specific layer
    logging.info("\n" + "=" * 60)
    logging.info("Test 3: Stitching a specific z-layer")
    logging.info("=" * 60)

    # Let's stitch layer 0 (the first layer)
    params_specific = StitchingParameters(
        input_folder=input_folder,
        z_layer_selection="0",  # Select layer 0
        verbose=True,
    )

    stitcher_specific = Stitcher(params_specific)

    logging.info(
        f"Total z-layers in acquisition: {stitcher_specific.computed_parameters.num_z}"
    )
    logging.info(f"Z-layer selection strategy: {params_specific.z_layer_selection}")
    logging.info(f"Selected layer index: 0")

    # You can run the stitching with:
    # stitcher_specific.run()

    # Test 4: Try to select an invalid layer (for demonstration)
    logging.info("\n" + "=" * 60)
    logging.info("Test 4: Demonstrating error handling for invalid layer")
    logging.info("=" * 60)

    try:
        # Try to select a layer that's likely out of range
        params_invalid = StitchingParameters(
            input_folder=input_folder,
            z_layer_selection="999",  # Probably out of range
            verbose=True,
        )

        stitcher_invalid = Stitcher(params_invalid)
        # This would fail during stitching if the layer is out of range
        logging.info(
            "Note: Layer 999 might be valid if your dataset has 1000+ z-layers"
        )
    except Exception as e:
        logging.error(f"Error creating stitcher: {e}")

    logging.info(
        "\nTo actually run the stitching, uncomment the stitcher.run() lines above."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_z_layer_selection.py <input_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    if not Path(input_folder).exists():
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)

    test_z_layer_selection(input_folder)
