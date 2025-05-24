# image_stitcher/flatfield_utils.py
import json
import logging
from pathlib import Path
from typing import Optional
import numpy as np
from .parameters import StitchingComputedParameters


def _find_manifest(directory: Path) -> Optional[Path]:
    """
    Search for the first .json or .npy file in the directory and return its Path.
    Returns None if no such file is found.
    """
    for pat in ("*.json", "*.npy"):
        hits = list(directory.glob(pat))
        if hits:
            return hits[0]
    return None


def load_flatfield_correction(
    directory: Path, computed_params: StitchingComputedParameters
) -> dict[int, np.ndarray]:
    """
    Look for flatfield_manifest.{json,npy} in directory,
    load each channel's .npy, and return channel-idx → flatfield array.
    """
    manifest_path = _find_manifest(directory)
    if manifest_path is None:
        logging.warning(f"No flatfield manifest found in {directory!r}")
        return {}

    data = json.loads(manifest_path.read_text())
    flatfields: dict[int, np.ndarray] = {}

    for ch_key, fname in data["files"].items():
        p = directory / fname
        if not p.exists():
            logging.warning(f"Flatfield file {p!r} missing; skipping channel {ch_key}")
            continue

        arr = np.load(p)

        # Try to interpret manifest key as an integer (wavelength or index)
        try:
            idx = int(ch_key)
            # If the int is a wavelength, map to channel index by substring match
            matches = [
                i
                for i, name in enumerate(computed_params.monochrome_channels)
                if str(idx) in name
            ]
            if matches:
                idx = matches[0]
            # If not found, assume it's already an index
            elif idx < len(computed_params.monochrome_channels):
                pass
            else:
                logging.warning(
                    f"No channel found for manifest key '{ch_key}'; skipping"
                )
                continue
        except ValueError:
            # Otherwise treat it as a channel-name and look up its index
            try:
                idx = computed_params.monochrome_channels.index(ch_key)
            except ValueError:
                logging.warning(f"Unknown channel '{ch_key}' in manifest; skipping")
                continue

        flatfields[idx] = arr

    logging.info(f"Loaded flatfields for channel-indices: {list(flatfields.keys())}")
    return flatfields
