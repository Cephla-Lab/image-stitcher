import math
import tempfile
import unittest
import pathlib

from image_stitcher.testutil import (
    PARAMETERS_FIXTURE_FILE,
    temporary_image_directory_params,
)

from .parameters import (
    ImagePlaneDims,
    OutputFormat,
    ScanPattern,
    StitchingParameters,
)


class ParametersTest(unittest.TestCase):
    def test_parsing(self) -> None:
        params = StitchingParameters.from_json_file(str(PARAMETERS_FIXTURE_FILE))
        self.assertEqual(params.scan_pattern, ScanPattern.unidirectional)
        self.assertEqual(params.output_format, OutputFormat.ome_zarr)
        self.assertEqual(params.input_folder, "/")

    def test_roundtrip(self) -> None:
        with tempfile.NamedTemporaryFile("w+", delete=True) as f:
            params = StitchingParameters.from_json_file(str(PARAMETERS_FIXTURE_FILE))
            params.to_json_file(f.name)
            f.flush()
            f.seek(0)
            contents = f.read()

        with open(PARAMETERS_FIXTURE_FILE) as f:
            fixture_contents = f.read()

        self.assertEqual(contents, fixture_contents)

    def test_flatfield_manifest_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir_path = pathlib.Path(temp_dir_name)
            # Create a dummy flatfield manifest file
            dummy_flatfield_manifest = temp_dir_path / "my_flatfields.json"
            with open(dummy_flatfield_manifest, "w") as ff_man_f:
                ff_man_f.write('{"version": 1}') # Minimal valid JSON content

            # Create a temporary parameters file that uses this flatfield manifest
            params_dict = {
                "input_folder": temp_dir_name, # Needs to be a valid path
                "output_format": ".ome.zarr",
                "scan_pattern": "Unidirectional",
                "apply_flatfield": True,
                "flatfield_manifest": str(dummy_flatfield_manifest.resolve()),
                "verbose": False,
                "force_stitch_to_disk": False,
                "num_pyramid_levels": None,
                "output_compression": "default",
            }
            temp_params_file = temp_dir_path / "temp_params.json"
            with open(temp_params_file, "w") as tp_f:
                import json # Added import for json.dump
                json.dump(params_dict, tp_f)

            # Load parameters and check flatfield_manifest
            params = StitchingParameters.from_json_file(str(temp_params_file))
            self.assertTrue(params.apply_flatfield)
            self.assertIsNotNone(params.flatfield_manifest)
            self.assertIsInstance(params.flatfield_manifest, pathlib.Path)
            # Pydantic should resolve the string to a Path object.
            # We also used resolve() when writing, so it should be absolute.
            self.assertEqual(params.flatfield_manifest, dummy_flatfield_manifest.resolve())
            self.assertTrue(params.flatfield_manifest.is_absolute())


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
