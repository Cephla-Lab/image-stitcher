# image_stitcher/flatfield_utils.py
import json
import logging
from pathlib import Path
from typing import Optional
import numpy as np
from .parameters import StitchingComputedParameters


def load_flatfield_correction(
    manifest_filepath: Path, computed_params: StitchingComputedParameters
) -> dict[int, np.ndarray]:
    """
    Load flatfield correction data from a specified manifest file.

    The manifest file (JSON) should contain a "files" dictionary,
    mapping channel keys to .npy filenames. These .npy files are expected
    to be in the same directory as the manifest file.

    Args:
        manifest_filepath: Path to the flatfield manifest JSON file.
        computed_params: Computed stitching parameters.

    Returns:
        A dictionary mapping channel-index to the flatfield numpy array.
        Returns an empty dictionary if the manifest is not found, is invalid,
        or if referenced .npy files are missing.
    """
    if not manifest_filepath.is_file():
        logging.warning(f"Flatfield manifest file not found or is not a file: {manifest_filepath!r}")
        return {}

    try:
        data = json.loads(manifest_filepath.read_text())
    except json.JSONDecodeError as e:
        logging.warning(f"Error decoding JSON from manifest file {manifest_filepath!r}: {e}")
        return {}

    flatfields: dict[int, np.ndarray] = {}

    if "files" not in data or not isinstance(data["files"], dict):
        logging.warning(f"Manifest file {manifest_filepath!r} is missing 'files' dictionary or it is malformed.")
        return {}

    manifest_dir = manifest_filepath.parent
    for ch_key, fname in data["files"].items():
        # Ensure fname is a string, as it comes from JSON
        if not isinstance(fname, str):
            logging.warning(f"Invalid filename type for channel key '{ch_key}' in manifest {manifest_filepath!r}. Expected string, got {type(fname)}. Skipping.")
            continue

        p = manifest_dir / fname
        if not p.exists(): # Check .is_file() if you want to be more strict
            logging.warning(f"Flatfield file {p!r} (referenced by manifest {manifest_filepath!r} for channel '{ch_key}') missing; skipping channel.")
            continue

        try:
            arr = np.load(p)
        except Exception as e: # Catch potential errors during np.load
            logging.warning(f"Error loading flatfield array from {p!r} for channel '{ch_key}': {e}. Skipping.")
            continue

        # Try to interpret manifest key as an integer (wavelength or index)
        try:
            idx = int(ch_key)
            matches = [
                i
                for i, name in enumerate(computed_params.monochrome_channels)
                if str(idx) in name
            ]
            if matches:
                idx = matches[0]
            elif idx < 0 or idx >= len(computed_params.monochrome_channels): # Check bounds if it's an index
                logging.warning(
                    f"Channel index '{ch_key}' out of bounds for monochrome_channels (len {len(computed_params.monochrome_channels)}); skipping"
                )
                continue
        except ValueError:
            # Otherwise treat it as a channel-name and look up its index
            try:
                idx = computed_params.monochrome_channels.index(ch_key)
            except ValueError:
                logging.warning(f"Unknown channel name '{ch_key}' in manifest {manifest_filepath!r}; skipping")
                continue

        flatfields[idx] = arr

    if flatfields:
        logging.info(f"Loaded flatfields from {manifest_filepath!r} for channel-indices: {list(flatfields.keys())}")
    else:
        logging.warning(f"No flatfields were successfully loaded from manifest {manifest_filepath!r}.")
    return flatfields