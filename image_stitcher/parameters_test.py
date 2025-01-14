import contextlib
import json
import math
import os
import pathlib
import tempfile
import unittest
from datetime import datetime, timezone
from typing import Generator

import numpy as np
import pandas as pd
import skimage.io

from .parameters import (
    DATETIME_FORMAT,
    ImagePlaneDims,
    OutputFormat,
    ScanPattern,
    StitchingComputedParameters,
    StitchingParameters,
)


class ParametersTest(unittest.TestCase):
    fixture_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "test_fixtures",
        "parameters_test",
        "parameters.json",
    )

    def test_parsing(self) -> None:
        params = StitchingParameters.from_json_file(self.fixture_file)
        self.assertEqual(params.scan_pattern, ScanPattern.unidirectional)
        self.assertEqual(params.output_format, OutputFormat.ome_zarr)
        self.assertEqual(params.input_folder, "/")

    def test_roundtrip(self) -> None:
        with tempfile.NamedTemporaryFile("w+", delete=True) as f:
            params = StitchingParameters.from_json_file(self.fixture_file)
            params.to_json_file(f.name)
            f.flush()
            f.seek(0)
            contents = f.read()

        with open(self.fixture_file) as f:
            fixture_contents = f.read()

        self.assertEqual(contents, fixture_contents)


@contextlib.contextmanager
def temporary_image_directory_params(
    n_rows: int,
    n_cols: int,
    im_size: ImagePlaneDims,
    channel_names: list[str],
) -> Generator[StitchingComputedParameters, None, None]:
    """Set up the files that the computed parameters requires for setup.

    TODO(colin): create tests with multiple timepoints and/or regions and/or z-slices.

    This includes:
        - images
        - the coordinates CSV file
        - the acquisition params file
    """
    with tempfile.TemporaryDirectory(delete=True) as d:
        base_dir = pathlib.Path(d)
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

        step_x_mm = 3.2
        step_y_mm = 3.2
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
                    skimage.io.imsave(im_file, make_fake_image(fov_counter))
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
                "magnification": 20.0,
                "NA": 0.8,
                "tube_lens_f_mm": 180.0,
                "name": "20x",
            },
            "sensor_pixel_size_um": 7.52,
            "tube_lens_mm": 180,
        }

        with open(acq_params_file, "w") as f:
            json.dump(acq_params, f)

        base_params = StitchingParameters.from_json_file(ParametersTest.fixture_file)
        base_params.input_folder = str(base_dir)
        base_params.use_registration = False
        computed = StitchingComputedParameters(base_params)
        yield computed


class ComputedParametersTest(unittest.TestCase):
    def test_pixel_size(self) -> None:
        with temporary_image_directory_params(
            2,
            2,
            ImagePlaneDims(1024, 1024),
            ["Fluorescence 405 nm Ex", "Fluorescence 488 nm Ex"],
        ) as computed:
            self.assertEqual(computed.pixel_size_um, 7.52 / 20)

    def test_channel_names_and_colors(self) -> None:
        channels = ["Fluorescence 405 nm Ex", "Fluorescence 488 nm Ex"]
        with temporary_image_directory_params(
            2,
            2,
            ImagePlaneDims(1024, 1024),
            channels,
        ) as computed:
            self.assertEqual(computed.channel_names, channels)
            self.assertEqual(computed.monochrome_channels, channels)
            self.assertEqual(computed.monochrome_colors, [0x0000FF, 0x00FF00])

    def test_chunks_and_image_dims(self) -> None:
        dims = ImagePlaneDims(width_px=1024, height_px=768)
        with temporary_image_directory_params(
            2,
            2,
            dims,
            ["Fluorescence 405 nm Ex", "Fluorescence 488 nm Ex"],
        ) as computed:
            self.assertEqual(computed.input_height, dims.height_px)
            self.assertEqual(computed.input_width, dims.width_px)
            _, _, _, height_chunk, width_chunk = computed.chunks
            self.assertGreaterEqual(height_chunk, dims.height_px)
            self.assertGreaterEqual(width_chunk, dims.width_px)

    def test_output_dims(self) -> None:
        dims = ImagePlaneDims(width_px=1024, height_px=768)
        n_rows = 2
        n_cols = 2
        with temporary_image_directory_params(
            n_rows,
            n_cols,
            dims,
            ["Fluorescence 405 nm Ex", "Fluorescence 488 nm Ex"],
        ) as computed:
            output_dims = computed.calculate_output_dimensions(timepoint=0, region="R0")
            px_size_um = 7.52 / 20
            self.assertEqual(
                output_dims.height_px,
                int(
                    math.ceil(
                        ((n_rows - 1) * 3.2 * 1000 + dims.height_px * px_size_um)
                        / px_size_um
                    )
                ),
            )
            self.assertEqual(
                output_dims.width_px,
                int(
                    math.ceil(
                        ((n_cols - 1) * 3.2 * 1000 + dims.width_px * px_size_um)
                        / px_size_um
                    )
                ),
            )
