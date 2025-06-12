import unittest
import tempfile
import pathlib
import json
import numpy as np
import logging

import numpy as np
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
            flatfield_correction=False,
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
            flatfield_correction=True,
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
            pyramid_levels=6,
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

    def test_compute_mip(self) -> None:
        # Test with 2D tiles (grayscale)
        tiles_2d = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 1], [7, 3]]),
            np.array([[2, 6], [1, 0]]),
        ]
        expected_mip_2d = np.array([[5, 6], [7, 4]])
        mip_2d = Stitcher.compute_mip(tiles_2d)
        np.testing.assert_array_equal(mip_2d, expected_mip_2d)

        # Test with 3D tiles (RGB)
        tiles_3d = [
            np.array([[[1, 0, 0], [2, 0, 0]], [[3, 0, 0], [4, 0, 0]]]), # R
            np.array([[[0, 5, 0], [0, 1, 0]], [[0, 7, 0], [0, 3, 0]]]), # G
            np.array([[[0, 0, 2], [0, 0, 6]], [[0, 0, 1], [0, 0, 0]]]), # B
        ]
        # Expected MIP should take the max across the first dimension (z-axis)
        # For the first pixel (0,0): max(R[0,0,0], G[0,0,1], B[0,0,2]) -> [1,5,2] is not how MIP works for multi-channel.
        # It computes MIP per channel if they are stacked as separate items in the list,
        # or if they are truly 3D (depth, height, width), it projects along depth.
        # The current compute_mip stacks along a new axis 0. So for 3D tiles like (H, W, C)
        # it will become (N, H, W, C) and max will be along axis 0.
        
        # Re-evaluating based on current compute_mip logic:
        # tiles_3d are list of (H, W, C) arrays.
        # np.stack(tiles_3d, axis=0) makes it (N, H, W, C)
        # np.max(stacked, axis=0) makes it (H, W, C)
        
        # Example based on this logic:
        tiles_3d_example = [
            np.array([[[1,10,100], [2,20,200]], [[3,30,300], [4,40,400]]]), # Tile 1 (H=2, W=2, C=3)
            np.array([[[5,15,105], [6,25,205]], [[7,35,305], [8,45,405]]]), # Tile 2
        ]
        expected_mip_3d = np.array([[[5,15,105], [6,25,205]], [[7,35,305], [8,45,405]]])

        mip_3d = Stitcher.compute_mip(tiles_3d_example)
        np.testing.assert_array_equal(mip_3d, expected_mip_3d)

        # Test with np.zeros (2D)
        tiles_zeros_2d = [
            np.zeros((2, 2), dtype=np.int8),
            np.zeros((2, 2), dtype=np.int8),
            np.zeros((2, 2), dtype=np.int8),
        ]
        expected_mip_zeros_2d = np.zeros((2, 2), dtype=np.int8)
        mip_zeros_2d = Stitcher.compute_mip(tiles_zeros_2d)
        np.testing.assert_array_equal(mip_zeros_2d, expected_mip_zeros_2d)

        # Test with np.zeros (3D)
        tiles_zeros_3d = [
            np.zeros((2, 2, 3), dtype=np.int8),
            np.zeros((2, 2, 3), dtype=np.int8),
        ]
        expected_mip_zeros_3d = np.zeros((2, 2, 3), dtype=np.int8)
        mip_zeros_3d = Stitcher.compute_mip(tiles_zeros_3d)
        np.testing.assert_array_equal(mip_zeros_3d, expected_mip_zeros_3d)

        # Test with np.ones (2D)
        tiles_ones_2d = [
            np.ones((3, 2), dtype=np.uint16),
            np.ones((3, 2), dtype=np.uint16),
        ]
        expected_mip_ones_2d = np.ones((3, 2), dtype=np.uint16)
        mip_ones_2d = Stitcher.compute_mip(tiles_ones_2d)
        np.testing.assert_array_equal(mip_ones_2d, expected_mip_ones_2d)

        # Test with np.ones (3D)
        tiles_ones_3d = [
            np.ones((2, 3, 4), dtype=np.float32),
            np.ones((2, 3, 4), dtype=np.float32),
            np.ones((2, 3, 4), dtype=np.float32),
        ]
        expected_mip_ones_3d = np.ones((2, 3, 4), dtype=np.float32)
        mip_ones_3d = Stitcher.compute_mip(tiles_ones_3d)
        np.testing.assert_array_equal(mip_ones_3d, expected_mip_ones_3d)

        # Test with a mix of 2D and 3D tiles - this should fail as per implementation (expects uniform tile shapes)
        # The code currently checks tiles[0].shape and assumes all tiles are the same.
        # Let's test the error case for an empty list
        with self.assertRaises(ValueError) as context_empty:
            Stitcher.compute_mip([])
        self.assertTrue("Cannot compute MIP from empty tile list" in str(context_empty.exception))

    def test_stitching_with_flatfield_manifest(self) -> None:
        im_size = ImagePlaneDims(1000, 1000)
        # Using 3 channels to match pixel assertions from test_basic_stage_stitching
        channel_names = ["DAPI", "FITC", "TRITC"]

        with tempfile.TemporaryDirectory() as temp_manifest_dir_str:
            temp_manifest_dir = pathlib.Path(temp_manifest_dir_str)

            # Create dummy .npy flatfield files (arrays of ones for no-op)
            flatfield_files_data = {}
            for ch_name in channel_names:
                npy_filename = f"{ch_name}_flatfield.npy"
                # Flatfield array should be float for division, ones for no-op
                flatfield_array = np.ones((im_size.height_px, im_size.width_px), dtype=np.float32)
                np.save(temp_manifest_dir / npy_filename, flatfield_array)
                flatfield_files_data[ch_name] = npy_filename

            # Create dummy flatfield_manifest.json
            manifest_content = {"files": flatfield_files_data}
            manifest_path = temp_manifest_dir / "flatfield_manifest.json"
            with open(manifest_path, "w") as f_manifest:
                json.dump(manifest_content, f_manifest)

            with temporary_image_directory_params(
                n_rows=3,
                n_cols=3,
                im_size=im_size,
                channel_names=channel_names,
                step_mm=(1.0, 1.0),
                sensor_pixel_size_um=20.0,
                magnification=20.0,
                # flatfield_correction is False by default in temporary_image_directory_params
                # We will set apply_flatfield and flatfield_manifest on the params directly
            ) as computed_params:
                # Configure StitchingParameters (parent of computed_params) to use the manifest
                computed_params.parent.apply_flatfield = True
                computed_params.parent.flatfield_manifest = manifest_path

                output_filename = None

                def finished_saving(output_path_str: str, _dtype: object) -> None:
                    nonlocal output_filename
                    output_filename = output_path_str

                callbacks = ProgressCallbacks.no_op()
                callbacks.finished_saving = finished_saving
                # Stitcher expects StitchingParameters (computed_params.parent)
                stitcher = Stitcher(computed_params.parent, callbacks)
                stitcher.run()
                self.assertIsNotNone(output_filename)

                im = next(Reader(parse_url(output_filename))()).data[0]
                # Assertions match test_basic_stage_stitching due to no-op flatfield
                self.assertEqual(im.shape, (1, len(channel_names), 1, 3000, 3000))
                # Pixel values are (fov % 65536)
                self.assertEqual(im[0, 0, 0, 0, 0].compute(), 0)    # FOV 0
                self.assertEqual(im[0, 0, 0, 1000, 0].compute(), 3) # FOV 3
                self.assertEqual(im[0, 0, 0, 1500, 0].compute(), 3) # FOV 3
                self.assertEqual(im[0, 0, 0, 2000, 0].compute(), 6) # FOV 6
                self.assertEqual(im[0, 0, 0, 0, 1000].compute(), 1) # FOV 1
                self.assertEqual(im[0, 0, 0, 0, 1500].compute(), 1) # FOV 1
                self.assertEqual(im[0, 0, 0, 0, 2000].compute(), 2) # FOV 2
                self.assertEqual(im[0, 0, 0, 2999, 2999].compute(), 8) # FOV 8

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
