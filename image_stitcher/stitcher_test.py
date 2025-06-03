import unittest

import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

from .parameters import ImagePlaneDims
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

        # Test with tiles of inconsistent shapes (e.g. one 2D, one 3D)
        # The current implementation derives behavior from the first tile.
        # If first is 2D, it expects all to be 2D. If first is 3D, expects all to be 3D.
        # A direct test for mixed types isn't straightforward without knowing if it *should* error or try to adapt.
        # The current code would likely error during np.stack if shapes are not broadcastable.
        # Let's test the ValueError for unexpected tile shape (e.g. 1D or 4D tile)
        with self.assertRaises(ValueError) as context_shape:
            Stitcher.compute_mip([np.array([1,2,3,4])]) # 1D array
        self.assertTrue("Unexpected tile shape: (4,)" in str(context_shape.exception))
        
        with self.assertRaises(ValueError) as context_shape_4d:
            Stitcher.compute_mip([np.array([[[[1]]]])]) # 4D array
        self.assertTrue("Unexpected tile shape: (1, 1, 1, 1)" in str(context_shape_4d.exception))
