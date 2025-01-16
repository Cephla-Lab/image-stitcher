import json
import os
import time

import dask.array as da
import numpy as np
import zarr


def create_ome_metadata(shape, dtype, pixel_size=None):
    """Create OME metadata dictionary."""
    metadata = {
        "version": "0.4",
        "datasets": [
            {
                "path": str(i),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [
                            1,
                            1,
                            1,
                            2**i,
                            2**i,
                        ],  # Scale factor for each resolution level
                    }
                ],
            }
            for i in range(4)
        ],  # 4 resolution levels
        "omero": {
            "version": "0.4",
            "channels": [
                {
                    "label": "Block Values",
                    "color": "FFFFFF",
                }
            ],
            "rdefs": {
                "defaultT": 0,
                "defaultZ": 0,
                "model": "greyscale",
            },
        },
    }
    return metadata


def create_scale_pyramid(base_array, num_levels=5):
    """Create a list of arrays at different scales."""
    pyramid = [base_array]

    for i in range(1, num_levels):
        # Calculate new shape
        new_shape = list(base_array.shape)
        new_shape[3] = new_shape[3] // (2**i)
        new_shape[4] = new_shape[4] // (2**i)

        # Create downscaled array
        downscaled = da.coarsen(
            np.mean, pyramid[0], {3: 2**i, 4: 2**i}, trim_excess=True
        ).astype(base_array.dtype)

        pyramid.append(downscaled)

    return pyramid


def write_ome_zarr(output_path, base_data, xy_chunk_size):
    # Set up base parameters
    base_chunks = (1, 1, 1, xy_chunk_size, xy_chunk_size)

    # Create pyramid
    print("Creating resolution pyramid...")
    pyramid = create_scale_pyramid(base_data)

    # Configure compression
    compressor = None

    # Create zarr hierarchy and ensure directory exists
    zarr_path = output_path
    os.makedirs(zarr_path, exist_ok=True)
    store = zarr.DirectoryStore(zarr_path)

    # Save OME metadata
    metadata = create_ome_metadata(base_data.shape, base_data.dtype)
    with open(os.path.join(zarr_path, ".zattrs"), "w") as f:
        json.dump(metadata, f)

    # Create and save zgroup metadata
    zgroup_metadata = {"zarr_format": 2}
    with open(os.path.join(zarr_path, ".zgroup"), "w") as f:
        json.dump(zgroup_metadata, f)

    # Time the saving operation
    print("Starting to save array pyramid...")

    # Save each resolution level
    for level, data in enumerate(pyramid):
        start_time = time.time()
        print(f"Saving resolution level {level}...")
        # Create directory for this level
        level_path = os.path.join(zarr_path, str(level))
        os.makedirs(level_path, exist_ok=True)

        # Adjust chunk size for each level
        level_chunks = list(base_chunks)
        level_chunks[3] = min(base_chunks[3], data.shape[3])
        level_chunks[4] = min(base_chunks[4], data.shape[4])

        # Create zarr array for this level
        z = zarr.create(
            shape=data.shape,
            chunks=tuple(level_chunks),
            dtype=np.uint16,
            store=store,
            path=str(level),
            compressor=compressor,
        )

        # Save the data
        da.store(data, z)

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(
            f"Time taken: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)"
        )

    # Print information about each resolution level
    print("\nResolution Level Information:")
    for level, data in enumerate(pyramid):
        print(f"\nLevel {level}:")
        print(f"Shape: {data.shape}")
        print(f"Size (uncompressed): {data.nbytes / (1024**3):.2f} GB")
