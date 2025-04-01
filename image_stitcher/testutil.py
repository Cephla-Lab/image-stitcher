import contextlib
import json
import os
import pathlib
import tempfile
from datetime import datetime, timezone
from typing import Generator, Optional

import numpy as np
import pandas as pd
import skimage

from .parameters import (
    DATETIME_FORMAT,
    ImagePlaneDims,
    StitchingComputedParameters,
    StitchingParameters,
)

PARAMETERS_FIXTURE_FILE = (
    pathlib.Path(__file__).parent.parent
    / "test_fixtures"
    / "parameters_test"
    / "parameters.json"
)


@contextlib.contextmanager
def temporary_image_directory_params(
    n_rows: int,
    n_cols: int,
    im_size: ImagePlaneDims,
    channel_names: list[str],
    name: str = "image_inputs",
    step_mm: tuple[float, float] = (3.2, 3.2),
    sensor_pixel_size_um: float = 7.52,
    magnification: float = 20.0,
    disk_based_output_arr: bool = False,
    pyramid_levels: Optional[int] = None,
    flatfield_correction: bool = False
) -> Generator[StitchingComputedParameters, None, None]:
    """Set up the files that the computed parameters requires for setup.

    TODO(colin): create tests with multiple timepoints and/or regions and/or z-slices.

    This includes:
        - images
        - the coordinates CSV file
        - the acquisition params file
    """
    with tempfile.TemporaryDirectory() as d:
        base_dir = pathlib.Path(d) / name
        os.makedirs(base_dir / "0", exist_ok=True)
        coords_file = base_dir / "0" / "coordinates.csv"
        acq_params_file = base_dir / "acquisition parameters.json"

        def image_filename(fov: int, channel: str) -> str:
            mod_channel = channel.replace(" ", "_")
            return f"R0_{fov}_0_{mod_channel}.tiff"

        def make_fake_image(fov: int) -> np.ndarray:
            return np.ones((im_size.height_px, im_size.width_px), dtype=np.uint16) * (
                fov % 65536
            )

        step_x_mm, step_y_mm = step_mm
        coordinates = []
        fov_counter = 0
        for r in range(n_rows):
            for c in range(n_cols):
                x_pos = c * step_x_mm
                y_pos = r * step_y_mm
                coordinates.append(
                    {
                        "region": "R0",
                        "fov": fov_counter,
                        "z_level": 0,
                        "x (mm)": x_pos,
                        "y (mm)": y_pos,
                        "z (um)": 0,
                        "time": datetime.now(timezone.utc).strftime(DATETIME_FORMAT),
                    }
                )
                for ch in channel_names:
                    im_file = base_dir / "0" / image_filename(fov_counter, ch)
                    skimage.io.imsave(im_file, make_fake_image(fov_counter), check_contrast=False)
                fov_counter += 1

        coords = pd.DataFrame(coordinates)
        coords.to_csv(coords_file, index=False)

        acq_params = {
            "dx(mm)": step_x_mm,
            "Nx": n_cols,
            "dy(mm)": step_y_mm,
            "Ny": n_rows,
            "dz(um)": 1.5,
            "Nz": 1,
            "dt(s)": 0.0,
            "Nt": 1,
            "with AF": False,
            "with reflection AF": False,
            "objective": {
                "magnification": magnification,
                "NA": 0.8,
                "tube_lens_f_mm": 180.0,
                "name": "20x",
            },
            "sensor_pixel_size_um": sensor_pixel_size_um,
            "tube_lens_mm": 180,
        }

        with open(acq_params_file, "w") as f:
            json.dump(acq_params, f)

        base_params = StitchingParameters.from_json_file(str(PARAMETERS_FIXTURE_FILE))
        if pyramid_levels:
            base_params.num_pyramid_levels = pyramid_levels
        if disk_based_output_arr:
            base_params.force_stitch_to_disk = True
        if flatfield_correction:
            base_params.apply_flatfield = flatfield_correction
        base_params.input_folder = str(base_dir)
        computed = StitchingComputedParameters(base_params)
        yield computed
