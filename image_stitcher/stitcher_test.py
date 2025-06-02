import unittest
import logging

from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

from .parameters import (
    ImagePlaneDims,
    OutputFormat,
    ScanPattern,
    StitchingParameters,
    ZLayerSelection,
)
from .stitcher import ProgressCallbacks, Stitcher
from .testutil import temporary_image_directory_params


class StitcherTest(unittest.TestCase):
    def test_basic_stage_stitching(self) -> None:
        with temporary_image_directory_params(
            n_rows=3,
            n_cols=3,
            # Exactly non-overlapping images aligned in a grid.
            im_size=ImagePlaneDims(1000, 1000),
            channel_names=["DAPI", "FITC", "TRITC"],
            step_mm=(1.0, 1.0),
            sensor_pixel_size_um=20.0,
            magnification=20.0,
            flatfield_correction=False
        ) as params:
            output_filename = None

            def finished_saving(output_path: str, _dtype: object) -> None:
                nonlocal output_filename
                output_filename = output_path

            callbacks = ProgressCallbacks.no_op()
            callbacks.finished_saving = finished_saving
            stitcher = Stitcher(params.parent, callbacks)
            stitcher.run()
            self.assertIsNotNone(output_filename)

            im = next(Reader(parse_url(output_filename))()).data[0]
            self.assertEqual(im.shape, (1, 3, 1, 3000, 3000))
            self.assertEqual(im[0, 0, 0, 0, 0].compute(), 0)
            self.assertEqual(im[0, 0, 0, 1000, 0].compute(), 3)
            self.assertEqual(im[0, 0, 0, 1500, 0].compute(), 3)
            self.assertEqual(im[0, 0, 0, 2000, 0].compute(), 6)
            self.assertEqual(im[0, 0, 0, 0, 1000].compute(), 1)
            self.assertEqual(im[0, 0, 0, 0, 1500].compute(), 1)
            self.assertEqual(im[0, 0, 0, 0, 2000].compute(), 2)
            self.assertEqual(im[0, 0, 0, 2999, 2999].compute(), 8)

    def test_basic_stage_stitching_with_flatfield(self) -> None:
        with temporary_image_directory_params(
            n_rows=3,
            n_cols=3,
            # Exactly non-overlapping images aligned in a grid.
            im_size=ImagePlaneDims(1000, 1000),
            channel_names=["DAPI", "FITC", "TRITC"],
            step_mm=(1.0, 1.0),
            sensor_pixel_size_um=20.0,
            magnification=20.0,
            flatfield_correction=True
        ) as params:
            output_filename = None

            def finished_saving(output_path: str, _dtype: object) -> None:
                nonlocal output_filename
                output_filename = output_path

            callbacks = ProgressCallbacks.no_op()
            callbacks.finished_saving = finished_saving
            stitcher = Stitcher(params.parent, callbacks)
            stitcher.run()
            self.assertIsNotNone(output_filename)

            im = next(Reader(parse_url(output_filename))()).data[0]
            self.assertEqual(im.shape, (1, 3, 1, 3000, 3000))
            self.assertEqual(im[0, 0, 0, 0, 0].compute(), 0)
            self.assertEqual(im[0, 0, 0, 1000, 0].compute(), 3)
            self.assertEqual(im[0, 0, 0, 1500, 0].compute(), 3)
            self.assertEqual(im[0, 0, 0, 2000, 0].compute(), 6)
            self.assertEqual(im[0, 0, 0, 0, 1000].compute(), 1)
            self.assertEqual(im[0, 0, 0, 0, 1500].compute(), 1)
            self.assertEqual(im[0, 0, 0, 0, 2000].compute(), 2)
            self.assertEqual(im[0, 0, 0, 2999, 2999].compute(), 8)

    def test_basic_stage_stitching_zarr_backed(self) -> None:
        with temporary_image_directory_params(
            n_rows=3,
            n_cols=3,
            # Exactly non-overlapping images aligned in a grid.
            im_size=ImagePlaneDims(1000, 1000),
            channel_names=["DAPI", "FITC", "TRITC"],
            step_mm=(1.0, 1.0),
            sensor_pixel_size_um=20.0,
            magnification=20.0,
            disk_based_output_arr=True,
        ) as params:
            output_filename = None

            def finished_saving(output_path: str, _dtype: object) -> None:
                nonlocal output_filename
                output_filename = output_path

            callbacks = ProgressCallbacks.no_op()
            callbacks.finished_saving = finished_saving
            stitcher = Stitcher(params.parent, callbacks)
            stitcher.run()
            self.assertIsNotNone(output_filename)

            im = next(Reader(parse_url(output_filename))()).data[0]
            self.assertEqual(im.shape, (1, 3, 1, 3000, 3000))
            self.assertEqual(im[0, 0, 0, 0, 0].compute(), 0)
            self.assertEqual(im[0, 0, 0, 1000, 0].compute(), 3)
            self.assertEqual(im[0, 0, 0, 1500, 0].compute(), 3)
            self.assertEqual(im[0, 0, 0, 2000, 0].compute(), 6)
            self.assertEqual(im[0, 0, 0, 0, 1000].compute(), 1)
            self.assertEqual(im[0, 0, 0, 0, 1500].compute(), 1)
            self.assertEqual(im[0, 0, 0, 0, 2000].compute(), 2)
            self.assertEqual(im[0, 0, 0, 2999, 2999].compute(), 8)

    def test_stitch_with_pyramid_and_zarr_out(self) -> None:
        with temporary_image_directory_params(
                n_rows=5,
                n_cols=5,
                # Exactly non-overlapping images aligned in a grid.
                im_size=ImagePlaneDims(1000, 1000),
                channel_names=["DAPI", "FITC", "TRITC"],
                step_mm=(1.0, 1.0),
                sensor_pixel_size_um=20.0,
                magnification=20.0,
                pyramid_levels=6
        ) as params:
            output_filename = None

            def finished_saving(output_path: str, _dtype: object) -> None:
                nonlocal output_filename
                output_filename = output_path

            callbacks = ProgressCallbacks.no_op()
            callbacks.finished_saving = finished_saving

            stitcher = Stitcher(params.parent, callbacks)
            stitcher.run()
            self.assertIsNotNone(output_filename)

            im = next(Reader(parse_url(output_filename))()).data[0]
            self.assertEqual(im.shape, (1, 3, 1, 5000, 5000))
            # The generated images have values corresponding to the field of view of each capture,
            # so we can check for valid ordering (up to the fov level) by checking that below.
            self.assertEqual(im[0, 0, 0, 0, 0].compute(), 0)
            self.assertEqual(im[0, 0, 0, 1000, 0].compute(), 5)
            self.assertEqual(im[0, 0, 0, 1500, 0].compute(), 5)
            self.assertEqual(im[0, 0, 0, 2000, 0].compute(), 10)
            self.assertEqual(im[0, 0, 0, 0, 1000].compute(), 1)
            self.assertEqual(im[0, 0, 0, 0, 1500].compute(), 1)
            self.assertEqual(im[0, 0, 0, 0, 2000].compute(), 2)
            self.assertEqual(im[0, 0, 0, 2999, 2999].compute(), 12)

    def test_z_layer_selection_all(self) -> None:
        """Test stitching with ZLayerSelection.ALL."""
        with temporary_image_directory_params(
            n_rows=2,
            n_cols=2,
            im_size=ImagePlaneDims(1000, 1000),
            channel_names=["DAPI"],
            step_mm=(1.0, 1.0),
            sensor_pixel_size_um=20.0,
            magnification=20.0,
            num_z=3  # Create 3 z-layers
        ) as params:
            output_filename = None

            def finished_saving(output_path: str, _dtype: object) -> None:
                nonlocal output_filename
                output_filename = output_path

            callbacks = ProgressCallbacks.no_op()
            callbacks.finished_saving = finished_saving
            params.parent.z_layer_selection = ZLayerSelection.ALL
            stitcher = Stitcher(params.parent, callbacks)
            stitcher.run()
            self.assertIsNotNone(output_filename)

            im = next(Reader(parse_url(output_filename))()).data[0]
            # Should have all 3 z-layers
            self.assertEqual(im.shape, (1, 1, 3, 2000, 2000))
            # Check that each z-layer has the expected values
            for z in range(3):
                # Each z-layer should have unique values based on fov and z-level
                self.assertEqual(im[0, 0, z, 0, 0].compute(), z)  # First FOV (0)
                self.assertEqual(im[0, 0, z, 0, 1000].compute(), 3 + z)  # Second FOV (1)
                self.assertEqual(im[0, 0, z, 1000, 0].compute(), 6 + z)  # Third FOV (2)
                self.assertEqual(im[0, 0, z, 1000, 1000].compute(), 9 + z)  # Fourth FOV (3)

    def test_z_layer_selection_middle(self) -> None:
        """Test stitching with ZLayerSelection.MIDDLE."""
        with temporary_image_directory_params(
            n_rows=2,
            n_cols=2,
            im_size=ImagePlaneDims(1000, 1000),
            channel_names=["DAPI"],
            step_mm=(1.0, 1.0),
            sensor_pixel_size_um=20.0,
            magnification=20.0,
            num_z=3  # Create 3 z-layers
        ) as params:
            output_filename = None

            def finished_saving(output_path: str, _dtype: object) -> None:
                nonlocal output_filename
                output_filename = output_path

            callbacks = ProgressCallbacks.no_op()
            callbacks.finished_saving = finished_saving
            params.parent.z_layer_selection = ZLayerSelection.MIDDLE
            stitcher = Stitcher(params.parent, callbacks)
            stitcher.run()
            self.assertIsNotNone(output_filename)

            im = next(Reader(parse_url(output_filename))()).data[0]
            # Should have only 1 z-layer (the middle one)
            self.assertEqual(im.shape, (1, 1, 1, 2000, 2000))
            # Check that the middle z-layer (index 1) has the expected values
            self.assertEqual(im[0, 0, 0, 0, 0].compute(), 1)  # First FOV (0), z=1
            self.assertEqual(im[0, 0, 0, 0, 1000].compute(), 4)  # Second FOV (1), z=1
            self.assertEqual(im[0, 0, 0, 1000, 0].compute(), 7)  # Third FOV (2), z=1
            self.assertEqual(im[0, 0, 0, 1000, 1000].compute(), 10)  # Fourth FOV (3), z=1

    def test_z_layer_selection_specific(self) -> None:
        """Test stitching with a specific z-layer index."""
        with temporary_image_directory_params(
            n_rows=2,
            n_cols=2,
            im_size=ImagePlaneDims(1000, 1000),
            channel_names=["DAPI"],
            step_mm=(1.0, 1.0),
            sensor_pixel_size_um=20.0,
            magnification=20.0,
            num_z=3  # Create 3 z-layers
        ) as params:
            output_filename = None

            def finished_saving(output_path: str, _dtype: object) -> None:
                nonlocal output_filename
                output_filename = output_path

            callbacks = ProgressCallbacks.no_op()
            callbacks.finished_saving = finished_saving
            params.parent.z_layer_selection = 2  # Select the last z-layer
            stitcher = Stitcher(params.parent, callbacks)
            stitcher.run()
            self.assertIsNotNone(output_filename)

            im = next(Reader(parse_url(output_filename))()).data[0]
            # Should have only 1 z-layer (the selected one)
            self.assertEqual(im.shape, (1, 1, 1, 2000, 2000))
            # Check that the selected z-layer (index 2) has the expected values
            self.assertEqual(im[0, 0, 0, 0, 0].compute(), 2)  # First FOV (0), z=2
            self.assertEqual(im[0, 0, 0, 0, 1000].compute(), 5)  # Second FOV (1), z=2
            self.assertEqual(im[0, 0, 0, 1000, 0].compute(), 8)  # Third FOV (2), z=2
            self.assertEqual(im[0, 0, 0, 1000, 1000].compute(), 11)  # Fourth FOV (3), z=2
