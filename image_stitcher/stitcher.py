import functools
import logging
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Self, TypeAlias

import dask.array as da
import numpy as np
import ome_zarr
import psutil
import skimage
import zarr
import zarr.storage
from aicsimageio import types as aics_types
from aicsimageio.writers import OmeTiffWriter

from image_stitcher.benchmarking_util import debug_timing

from .parameters import (
    OutputFormat,
    StitchingComputedParameters,
    StitchingParameters,
)


@dataclass
class ProgressCallbacks:
    update_progress: Callable[[int, int], None]
    getting_flatfields: Callable[[], None]
    starting_stitching: Callable[[], None]
    starting_saving: Callable[[bool], None]
    finished_saving: Callable[[str, object], None]

    @classmethod
    def no_op(cls) -> Self:
        return cls(
            update_progress=lambda _a, _b: None,
            getting_flatfields=lambda: None,
            starting_stitching=lambda: None,
            starting_saving=lambda _: None,
            finished_saving=lambda _s, _obj: None,
        )


@dataclass
class Paths:
    """Output path construction logic for this module's various modes."""

    output_folder: pathlib.Path
    output_format: OutputFormat

    def per_timepoint_region_output(self, timepoint: int, region: str) -> pathlib.Path:
        return (
            self.per_timepoint_dir(timepoint)
            / f"{region}_stitched{self.output_format.value}"
        )

    def per_timepoint_dir(self, timepoint: int) -> pathlib.Path:
        return self.output_folder / f"{timepoint}_stitched"


# We want to choose the type of the stitched array based on the amount of memory
# available.  If we can fit the whole computation in memory, use numpy for the
# stitching since that is faster, otherwise use dask.
AnyArray: TypeAlias = np.ndarray | da.Array | zarr.Array


class Stitcher:
    def __init__(
        self,
        params: StitchingParameters,
        callbacks: ProgressCallbacks = ProgressCallbacks.no_op(),
    ):
        self.params = params
        self.callbacks = callbacks
        self.computed_parameters = StitchingComputedParameters(self.params)
        self._paths: Paths | None = None

    @property
    def paths(self) -> Paths:
        if self._paths is None:
            self._paths = Paths(self.params.stitched_folder, self.params.output_format)
        return self._paths

    def create_output_array(self, timepoint: int, region: str) -> AnyArray:
        width, height = self.computed_parameters.calculate_output_dimensions(
            timepoint, region
        )
        # create zeros with the right shape/dtype per timepoint per region
        output_shape = (
            1,
            self.computed_parameters.num_c,
            self.computed_parameters.num_z,
            height,
            width,
        )
        logging.debug(
            f"region {region} timepoint {timepoint} output array dimensions: {output_shape}"
        )
        output_estimated_memory_required = (
            functools.reduce(lambda a, b: a * b, output_shape)
            * self.computed_parameters.dtype.itemsize
        )
        # psutil docs say the available item of this function's output tries to
        # be a cross-platform measure of how much memory could be used before
        # swap is required
        estimated_available_memory = psutil.virtual_memory().available

        if (
            output_estimated_memory_required < 0.45 * estimated_available_memory
            and not self.params.force_stitch_to_disk
        ):
            logging.debug("Using an in-memory numpy array for stitching.")
            # If the whole stitched image fits into memory, just store it as a
            # single numpy array.
            return np.zeros(output_shape, dtype=self.computed_parameters.dtype)
        else:
            logging.debug("Using an on-disk zarr array for stitching.")
            # TODO(colin): this might be too small in many cases; we should try
            # setting a chunk size based on available memory? Need a benchmark
            # for this case to test that out.
            chunks = self.computed_parameters.chunks
            output_path = self.paths.per_timepoint_region_output(timepoint, region)
            store = zarr.storage.DirectoryStore(output_path)
            root = zarr.group(store)
            return root.zeros(
                name="0",
                shape=output_shape,
                dtype=self.computed_parameters.dtype,
                chunks=chunks,
                dimension_separator="/",
            )

    def place_tile(
        self,
        stitched_region: AnyArray,
        tile: AnyArray,
        x_pixel: int,
        y_pixel: int,
        z_level: int,
        channel: str,
    ) -> None:
        if len(tile.shape) == 2:
            # Handle 2D grayscale image
            channel_idx = self.computed_parameters.monochrome_channels.index(channel)
            self.place_single_channel_tile(
                stitched_region, tile, x_pixel, y_pixel, z_level, channel_idx
            )

        elif len(tile.shape) == 3:
            if tile.shape[2] == 3:
                # Handle RGB image
                channel = channel.split("_")[0]
                for i, color in enumerate(["R", "G", "B"]):
                    channel_idx = self.computed_parameters.monochrome_channels.index(
                        f"{channel}_{color}"
                    )
                    self.place_single_channel_tile(
                        stitched_region,
                        tile[:, :, i],
                        x_pixel,
                        y_pixel,
                        z_level,
                        channel_idx,
                    )
            elif tile.shape[0] == 1:
                channel_idx = self.computed_parameters.monochrome_channels.index(
                    channel
                )
                self.place_single_channel_tile(
                    stitched_region,
                    tile[0],
                    x_pixel,
                    y_pixel,
                    z_level,
                    channel_idx,
                )
        else:
            raise ValueError(f"Unexpected tile shape: {tile.shape}")

    def place_single_channel_tile(
        self,
        stitched_region: AnyArray,
        tile: AnyArray,
        x_pixel: int,
        y_pixel: int,
        z_level: int,
        channel_idx: int,
    ) -> None:
        if len(stitched_region.shape) != 5:
            raise ValueError(
                f"Unexpected stitched_region shape: {stitched_region.shape}. Expected 5D array (t, c, z, y, x)."
            )

        # Calculate end points based on stitched_region shape
        y_end = min(y_pixel + tile.shape[0], stitched_region.shape[3])
        x_end = min(x_pixel + tile.shape[1], stitched_region.shape[4])

        # Extract the tile slice we'll use
        tile_slice = tile[: y_end - y_pixel, : x_end - x_pixel]

        try:
            # Place the tile slice - use t=0 since we're working with 1-timepoint arrays
            stitched_region[0, channel_idx, z_level, y_pixel:y_end, x_pixel:x_end] = (
                tile_slice
            )
        except Exception as e:
            logging.error(f"Failed to place tile. Details: {str(e)}")
            logging.debug(
                f"t:0, channel_idx:{channel_idx}, z_level:{z_level}, y:{y_pixel}-{y_end}, x:{x_pixel}-{x_end}"
            )
            logging.debug(f"tile slice shape: {tile_slice.shape}")
            logging.debug(f"stitched_region shape: {stitched_region.shape}")
            logging.debug(
                f"output location shape: {stitched_region[0, channel_idx, z_level, y_pixel:y_end, x_pixel:x_end].shape}"
            )
            raise

    def stitch_region(self, timepoint: int, region: str) -> AnyArray:
        """Stitch and save single region for a specific timepoint."""
        start_time = time.time()
        # Initialize output array
        region_metadata = self.computed_parameters.get_region_metadata(
            int(timepoint), region
        )
        stitched_region = self.create_output_array(timepoint, region)
        x_min = min(self.computed_parameters.x_positions)
        y_min = min(self.computed_parameters.y_positions)
        total_tiles = len(region_metadata)
        processed_tiles = 0
        logging.info(
            f"Beginning stitching of {total_tiles} tiles for region {region} timepoint {timepoint}"
        )

        # Process each tile with progress tracking
        for key, tile_info in region_metadata.items():
            t, _, _, z_level, channel = key
            tile = skimage.io.imread(tile_info.filepath)

            x_pixel = int(
                (tile_info.x - x_min) * 1000 / self.computed_parameters.pixel_size_um
            )
            y_pixel = int(
                (tile_info.y - y_min) * 1000 / self.computed_parameters.pixel_size_um
            )

            self.place_tile(stitched_region, tile, x_pixel, y_pixel, z_level, channel)

            self.callbacks.update_progress(processed_tiles, total_tiles)
            processed_tiles += 1

        logging.info(
            f"Time to stitch region {region} timepoint {t}: {time.time() - start_time}"
        )
        return stitched_region

    def save_region_aics(
        self, timepoint: int, region: str, stitched_region: da.Array
    ) -> pathlib.Path:
        """Save stitched region data as OME-ZARR or OME-TIFF using aicsimageio."""
        start_time = time.time()
        # Ensure output directory exists
        output_path = self.paths.per_timepoint_region_output(timepoint, region)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        # Create physical pixel sizes object
        physical_pixel_sizes = aics_types.PhysicalPixelSizes(
            Z=self.computed_parameters.acquisition_params.get("dz(um)", 1.0),
            Y=self.computed_parameters.pixel_size_um,
            X=self.computed_parameters.pixel_size_um,
        )

        # Convert colors to RGB lists for OME format
        rgb_colors = [
            [c >> 16, (c >> 8) & 0xFF, c & 0xFF]
            for c in self.computed_parameters.monochrome_colors
        ]

        assert self.params.output_format == OutputFormat.ome_tiff
        logging.info(f"Writing OME-TIFF to: {output_path}")

        # Build OME metadata for TIFF
        ome_meta = OmeTiffWriter.build_ome(
            data_shapes=[stitched_region.shape],
            data_types=[stitched_region.dtype],
            dimension_order=["TCZYX"],
            channel_names=[self.computed_parameters.monochrome_channels],
            image_name=[f"{region}_t{timepoint}"],
            physical_pixel_sizes=[physical_pixel_sizes],
            channel_colors=[rgb_colors],
        )

        # Write the image with metadata
        OmeTiffWriter.save(
            data=stitched_region,
            uri=output_path,
            dim_order="TCZYX",
            ome_xml=ome_meta,
            channel_names=self.computed_parameters.monochrome_channels,
            physical_pixel_sizes=physical_pixel_sizes,
            channel_colors=rgb_colors,
        )

        logging.info(f"Successfully saved to: {output_path}")
        logging.info(
            f"Time to save region {region} timepoint {timepoint}: {time.time() - start_time}"
        )
        return output_path

    def save_region_ome_zarr(
        self, timepoint: int, region: str, stitched_region: da.Array
    ) -> pathlib.Path:
        """Save stitched region data as OME-ZARR using direct pyramid writing.

        Optimized for large microscopy datasets with proper physical coordinates.

        Args:
            timepoint: The timepoint of the data
            region: The region identifier
            stitched_region: The 5D image data array (TCZYX)

        Returns:
            path to the saved OME-ZARR file
        """
        start_time = time.time()
        output_path = self.paths.per_timepoint_region_output(timepoint, region)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        logging.info(f"Writing OME-ZARR to: {output_path}")

        # Create zarr store and root group
        store = ome_zarr.io.parse_url(output_path, mode="w").store
        root = zarr.group(store=store)

        if not isinstance(stitched_region, da.Array):
            pyramid = self.generate_pyramid(
                da.from_array(stitched_region),
                self.computed_parameters.num_pyramid_levels,
            )
        else:
            pyramid = self.generate_pyramid(
                stitched_region, self.computed_parameters.num_pyramid_levels
            )

        # Define correct physical coordinates with proper micrometer scaling
        transforms: list[list[dict[str, Any]]] = []
        for level in range(self.computed_parameters.num_pyramid_levels):
            scale = 2**level
            transforms.append(
                [
                    {
                        "type": "scale",
                        "scale": [
                            1,  # time
                            1,  # channels
                            float(
                                self.computed_parameters.acquisition_params.get(
                                    "dz(um)", 1.0
                                )
                            ),  # z in microns
                            float(
                                self.computed_parameters.pixel_size_um * scale
                            ),  # y with pyramid scaling
                            float(
                                self.computed_parameters.pixel_size_um * scale
                            ),  # x with pyramid scaling
                        ],
                    }
                ]
            )

        # Configure storage options with optimal chunking
        storage_opts = {
            "chunks": self.computed_parameters.chunks,
            "compressor": zarr.storage.default_compressor,
        }

        if isinstance(stitched_region, zarr.Array):
            # We already stored the higest resolution version it its target
            # location. In that case, for the first level we just need to write
            # the metadata.
            start_index = 1
            datasets: list[dict[str, Any]] = [{"path": "0"}]
        else:
            start_index = 0
            datasets = []

        with debug_timing("write image data pyramid"):
            for pyramid_idx in range(start_index, len(pyramid)):
                da.to_zarr(
                    arr=pyramid[pyramid_idx],
                    url=root.store,
                    component=str(pathlib.Path(root.path) / str(pyramid_idx)),
                    storage_options=storage_opts,
                    compressor=storage_opts.get(
                        "compressor", zarr.storage.default_compressor
                    ),
                    dimension_separator="/",
                    compute=True,
                )
                datasets.append({"path": str(pyramid_idx)})

        fmt = ome_zarr.format.CurrentFormat()
        fmt.validate_coordinate_transformations(
            len(pyramid[0].shape), len(pyramid), transforms
        )

        for dataset, transform in zip(datasets, transforms):
            dataset["coordinateTransformations"] = transform

        with debug_timing(".write_multiscale_metadata()"):
            ome_zarr.writer.write_multiscales_metadata(
                group=root,
                datasets=datasets,
                fmt=fmt,
                axes=[  # Required for OME-ZARR >= 0.3
                    {"name": "t", "type": "time", "unit": "second"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                name=f"{region}_t{timepoint}",
            )

        # Add complete OMERO metadata for visualization
        # Note(colin): this is an unusual API, but reading the Zarr library
        # code, I believe that modifying root.attrs actually syncs them to disk!
        root.attrs["omero"] = {
            "id": 1,
            "name": f"{region}_t{timepoint}",
            "version": "0.4",
            "channels": [
                {
                    "label": name,
                    "color": f"{color:06X}",
                    "window": {
                        "start": 0,
                        "end": np.iinfo(self.computed_parameters.dtype).max,
                        "min": 0,
                        "max": np.iinfo(self.computed_parameters.dtype).max,
                    },
                    "active": True,
                    "coefficient": 1,
                    "family": "linear",
                }
                for name, color in zip(
                    self.computed_parameters.monochrome_channels,
                    self.computed_parameters.monochrome_colors,
                )
            ],
        }
        logging.info(f"Successfully saved OME-ZARR to: {output_path}")
        logging.info(
            f"Time to save region {region} timepoint {timepoint}: {time.time() - start_time}"
        )
        return output_path

    def generate_pyramid(self, image: da.Array, num_levels: int) -> list[da.Array]:
        pyramid = [image]
        for level in range(1, num_levels):
            scale_factor = 2**level
            factors = {0: 1, 1: 1, 2: 1, 3: scale_factor, 4: scale_factor}
            # TODO/NOTE(colin): there are many possible ways to downscale an
            # image that are more or less appopriate for different usecases.
            # Expose the method as a parameter? (For example, np.mean is
            # inapproriate for binarized or labelled masks stored as normal images.)
            downsampled = da.coarsen(np.mean, image, factors, trim_excess=True)
            pyramid.append(downsampled)
        return pyramid

    def run(self) -> None:
        """Main execution method handling timepoints and regions."""
        stime = time.time()
        # Initial setup
        self.paths.output_folder.mkdir(exist_ok=True, parents=True)

        # Process each timepoint and region
        for timepoint in self.computed_parameters.timepoints:
            timepoint = int(timepoint)

            ttime = time.time()
            logging.info(f"Processing timepoint {timepoint}")

            # Create timepoint output directory
            t_output_dir = self.paths.per_timepoint_dir(timepoint)
            t_output_dir.mkdir(exist_ok=True, parents=True)

            for region in self.computed_parameters.regions:
                rtime = time.time()
                logging.info(f"Processing region {region}...")

                # Stitch region
                self.callbacks.starting_stitching()

                stitched_region = self.stitch_region(timepoint, region)
                with debug_timing("rechunking"):
                    if isinstance(stitched_region, np.ndarray):
                        dask_stitched_region = da.from_array(
                            stitched_region,
                            chunks=self.computed_parameters.chunks,
                            name=f"stitched:t={timepoint},r={region}",
                        )
                    else:
                        dask_stitched_region = stitched_region

                # Save the region
                self.callbacks.starting_saving(False)
                if self.params.output_format == OutputFormat.ome_zarr:
                    output_path = self.save_region_ome_zarr(
                        timepoint, region, dask_stitched_region
                    )
                else:
                    assert self.params.output_format == OutputFormat.ome_tiff
                    output_path = self.save_region_aics(
                        timepoint, region, dask_stitched_region
                    )

                logging.info(
                    f"Completed region {region} (saved to {output_path}): {time.time() - rtime}"
                )

            logging.info(f"Completed timepoint {timepoint}: {time.time() - ttime}")

        # Post-processing based on merge settings
        post_time = time.time()
        self.callbacks.starting_saving(True)

        # Emit finished signal with the last saved path
        final_path = self.paths.per_timepoint_region_output(
            self.computed_parameters.timepoints[-1],
            self.computed_parameters.regions[-1],
        )
        self.callbacks.finished_saving(str(final_path), self.computed_parameters.dtype)

        logging.info(f"Post-processing time: {time.time() - post_time}")
        logging.info(f"Total processing time: {time.time() - stime}")
