import unittest

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
