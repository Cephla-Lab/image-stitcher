import logging
import random
from typing import Callable

import numpy as np
from basicpy import BaSiC
from skimage.io import imread

from .parameters import StitchingComputedParameters

MAX_FLATFIELD_IMAGES_PER_T = 32
MAX_FLATFIELD_IMAGES = 48


def compute_flatfield_correction(
    computed_params: StitchingComputedParameters, progress_callback: Callable[[], None]
) -> dict[int, np.ndarray]:
    """Compute a flatfield correction across many images for each channel.

    This does not modify computed_params.flatfields; store the output there if
    desired external to this function.
    """

    flatfields = {}

    def process_images(images: np.ndarray, channel_name: str):
        if images.size == 0:
            logging.warning(f"No images found for channel {channel_name}")
            return

        if images.ndim != 3 and images.ndim != 4:
            raise ValueError(
                f"Images must be 3 or 4-dimensional array, with dimension of (T, Y, X) or (T, Z, Y, X). Got shape {images.shape}"
            )

        basic = BaSiC(get_darkfield=False, smoothness_flatfield=1)
        basic.fit(images)
        channel_index = computed_params.monochrome_channels.index(channel_name)
        flatfields[channel_index] = basic.flatfield
        if progress_callback:
            progress_callback()

    for channel in computed_params.channel_names:
        logging.info(f"Calculating {channel} flatfield...")

        all_tile_infos = []
        for t in computed_params.timepoints:
            # We might have a ton of images for this time point.  We only want to select up to MAX_FLATFIELD_IMAGES
            # to use for flatfield correction.  To do this, load all the metadata and pick random tile infos
            # to use for this timepoint.
            all_tile_infos.extend([
                tile
                for key, tile in computed_params.acquisition_metadata.items()
                if tile.channel == channel and key.t == int(t)
            ])

        random.shuffle(all_tile_infos)
        selected_tile_infos = all_tile_infos[: min(MAX_FLATFIELD_IMAGES, len(all_tile_infos))]

        if len(selected_tile_infos) < len(all_tile_infos):
            logging.warning(f"Limiting flatfield correction to {len(selected_tile_infos)} images instead of using all {len(all_tile_infos)}")

        images = [
            imread(tile.filepath)
            for tile in selected_tile_infos
        ]

        if not images:
            logging.warning(f"No images found for channel {channel} for any timepoint")
            continue

        images = np.array(images)

        if images.ndim in (3, 4):
            # Images are in the shape (N, Y, X) or (N, Z, Y, X)
            process_images(images, channel)
        else:
            raise ValueError(
                f"Unexpected number of dimensions in images array: {images.ndim}"
            )

    return flatfields
