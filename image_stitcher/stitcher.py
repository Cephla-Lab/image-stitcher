import logging
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Callable, Self, cast

import dask.array as da
import numpy as np
import ome_zarr
import zarr
from aicsimageio import types as aics_types
from aicsimageio.writers import OmeTiffWriter, OmeZarrWriter
from dask_image.imread import imread as dask_imread

from .parameters import (
    OutputFormat,
    SPatternScanParams,
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

    @property
    def region_time_series_dir(self) -> pathlib.Path:
        return self.output_folder / "region_time_series"

    def merged_timepoints_output(self, region: str) -> pathlib.Path:
        return (
            self.region_time_series_dir
            / f"{region}_time_series{self.output_format.value}"
        )

    @property
    def hcs_timepoints_dir(self) -> pathlib.Path:
        return self.output_folder / "hcs_timepoints"

    def merged_hcs_output(self, timepoint: int) -> pathlib.Path:
        return self.hcs_timepoints_dir / f"{timepoint}_hcs{self.output_format.value}"

    @property
    def complete_hcs_output(self) -> pathlib.Path:
        return self.hcs_timepoints_dir / f"complete_hcs{self.output_format.value}"


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

    def create_output_array(self, timepoint: int, region: str) -> da.Array:
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
        return cast(
            da.Array,
            da.zeros(
                output_shape,
                dtype=self.computed_parameters.dtype,
                chunks=self.computed_parameters.chunks,
            ),
        )

    def get_tile(
        self, t: int, region: str, x: float, y: float, channel: str, z_level: int
    ) -> da.Array | None:
        """Get a specific tile using standardized data access."""
        region_metadata = self.computed_parameters.get_region_metadata(t, region)

        for value in region_metadata.values():
            if (
                value.x == x
                and value.y == y
                and value.channel == channel
                and value.z_level == z_level
            ):
                try:
                    return cast(da.Array, dask_imread(value.filepath)[0])
                except FileNotFoundError:
                    logging.warning(f"Warning: Tile file not found: {value.filepath}")
                    return None

        logging.warning(
            f"Warning: No matching tile found for region {region}, x={x}, y={y}, channel={channel}, z={z_level}"
        )
        return None

    def place_tile(
        self,
        stitched_region: da.Array,
        tile: da.Array,
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
        stitched_region: da.Array,
        tile: da.Array,
        x_pixel: int,
        y_pixel: int,
        z_level: int,
        channel_idx: int,
    ) -> None:
        if len(stitched_region.shape) != 5:
            raise ValueError(
                f"Unexpected stitched_region shape: {stitched_region.shape}. Expected 5D array (t, c, z, y, x)."
            )

        if self.params.apply_flatfield:
            raise NotImplementedError("Flatfield correction not yet implemented.")

        if self.params.use_registration:
            raise NotImplementedError("Registration not yet implemented.")

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

    def stitch_region(self, timepoint: int, region: str) -> da.Array:
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
            tile = dask_imread(tile_info.filepath)[0]

            if self.params.use_registration:
                col_index = self.computed_parameters.x_positions.index(tile_info.x)
                row_index = self.computed_parameters.y_positions.index(tile_info.y)

                if isinstance(
                    self.computed_parameters.scan_params, SPatternScanParams
                ) and self.computed_parameters.scan_params.h_shift_rev_rows.is_reversed(
                    row_index
                ):
                    h_shift = self.computed_parameters.scan_params.h_shift_rev
                else:
                    h_shift = self.computed_parameters.scan_params.h_shift

                x_pixel = int(
                    col_index * (self.computed_parameters.input_width + h_shift[1])
                )
                y_pixel = int(
                    row_index
                    * (
                        self.computed_parameters.input_height
                        + self.computed_parameters.scan_params.v_shift[0]
                    )
                )

                # TODO(colin): this looks like we're attempting to deal with
                # non-grid-aligned patterns perhaps, but I think our assumptions
                # on how we calculate the number of rows and columns are
                # incompatible with that unless we're moving on a perfect
                # diagonal?
                if h_shift[0] < 0:
                    y_pixel += int(
                        (len(self.computed_parameters.x_positions) - 1 - col_index)
                        * abs(h_shift[0])
                    )
                else:
                    y_pixel += int(col_index * h_shift[0])

                if self.computed_parameters.scan_params.v_shift[1] < 0:
                    x_pixel += int(
                        (len(self.computed_parameters.y_positions) - 1 - row_index)
                        * abs(self.computed_parameters.scan_params.v_shift[1])
                    )
                else:
                    x_pixel += int(
                        row_index * self.computed_parameters.scan_params.v_shift[1]
                    )
            else:
                x_pixel = int(
                    (tile_info.x - x_min)
                    * 1000
                    / self.computed_parameters.pixel_size_um
                )
                y_pixel = int(
                    (tile_info.y - y_min)
                    * 1000
                    / self.computed_parameters.pixel_size_um
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

        if self.params.output_format == OutputFormat.ome_zarr:
            logging.info(f"Writing OME-ZARR to: {output_path}")
            writer = OmeZarrWriter(output_path)

            # Build OME metadata for Zarr
            # ome_meta = writer.build_ome(
            #     size_z=self.num_z,
            #     image_name=f"{region}_t{timepoint}",
            #     channel_names=self.monochrome_channels,
            #     channel_colors=self.monochrome_colors,
            #     channel_minmax=channel_minmax
            # )

            # Write the image with metadata
            writer.write_image(
                image_data=stitched_region,
                image_name=f"{region}_t{timepoint}",
                physical_pixel_sizes=physical_pixel_sizes,
                channel_names=self.computed_parameters.monochrome_channels,
                channel_colors=self.computed_parameters.monochrome_colors,  # rgb_colors,
                chunk_dims=self.computed_parameters.chunks,
                scale_num_levels=self.computed_parameters.num_pyramid_levels,
                scale_factor=2.0,
                dimension_order="TCZYX",
            )

        else:
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

        # Calculate pyramid using scaler - maintains efficiency for both dask and numpy arrays
        scaler = ome_zarr.scale.Scaler(
            max_layer=self.computed_parameters.num_pyramid_levels - 1
        )
        pyramid = scaler.nearest(stitched_region)

        # Define correct physical coordinates with proper micrometer scaling
        transforms = []
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

        # Write pyramid data with full metadata
        ome_zarr.writer.write_multiscale(
            pyramid=pyramid,
            group=root,
            axes=[  # Required for OME-ZARR >= 0.3
                {"name": "t", "type": "time", "unit": "second"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            coordinate_transformations=transforms,
            storage_options=storage_opts,
            name=f"{region}_t{timepoint}",
            fmt=ome_zarr.format.CurrentFormat(),
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

    def merge_timepoints_per_region(self) -> None:
        # For each region, load and merge its timepoints
        for region in self.computed_parameters.regions:
            output_path = self.paths.merged_timepoints_output(region)
            store = ome_zarr.io.parse_url(output_path, mode="w").store
            root = zarr.group(store=store)

            # Load and merge data
            merged_data = self.load_and_merge_timepoints(region)

            # Create region group and write metadata
            region_group = root.create_group(region)

            # Prepare dataset and transformation metadata
            datasets = [
                {
                    "path": str(i),
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": [
                                1,
                                1,
                                self.computed_parameters.acquisition_params.get(
                                    "dz(um)", 1
                                ),
                                self.computed_parameters.pixel_size_um * (2**i),
                                self.computed_parameters.pixel_size_um * (2**i),
                            ],
                        }
                    ],
                }
                for i in range(self.computed_parameters.num_pyramid_levels)
            ]

            axes = [
                {"name": "t", "type": "time", "unit": "second"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ]

            # Write multiscales metadata
            ome_zarr.writer.write_multiscales_metadata(
                region_group, datasets, axes=axes, name=region
            )

            # Generate and write pyramid
            pyramid = self.generate_pyramid(
                merged_data, self.computed_parameters.num_pyramid_levels
            )
            storage_options = {"chunks": self.computed_parameters.chunks}
            logging.info(f"Writing time series for region {region}")
            ome_zarr.writer.write_multiscale(
                pyramid=pyramid,
                group=region_group,
                axes=axes,
                coordinate_transformations=[
                    d["coordinateTransformations"] for d in datasets
                ],
                storage_options=storage_options,
                name=region,
            )

            # Add OMERO metadata
            region_group.attrs["omero"] = {
                "name": f"Region_{region}",
                "version": "0.4",
                "channels": [
                    {
                        "label": name,
                        "color": f"{color:06X}",
                        "window": {
                            "start": 0,
                            "end": np.iinfo(self.computed_parameters.dtype).max,
                        },
                    }
                    for name, color in zip(
                        self.computed_parameters.monochrome_channels,
                        self.computed_parameters.monochrome_colors,
                    )
                ],
            }

        self.callbacks.finished_saving(str(output_path), self.computed_parameters.dtype)

    def load_and_merge_timepoints(self, region: str) -> da.Array:
        """Load and merge all timepoints for a specific region."""
        t_data = []
        t_shapes = []

        for t in self.computed_parameters.timepoints:
            zarr_path = self.paths.per_timepoint_region_output(t, region)
            logging.info(f"Loading t:{t} region:{region}, path:{zarr_path}")

            try:
                z = zarr.open(zarr_path, mode="r")
                t_array = cast(
                    da.Array,
                    da.from_array(z["0"], chunks=self.computed_parameters.chunks),
                )
                t_data.append(t_array)
                t_shapes.append(t_array.shape)
            except Exception as e:
                logging.error(f"Error loading timepoint {t}, region {region}: {e}")
                continue

        if not t_data:
            raise ValueError(f"No data loaded from any timepoints for region {region}")

        # Handle single vs multiple timepoints
        if len(t_data) == 1:
            return t_data[0]

        # Pad arrays to largest size and concatenate
        max_shape = tuple(max(s) for s in zip(*t_shapes))
        padded_data = [self.pad_to_largest(t, max_shape) for t in t_data]
        merged_data = cast(da.Array, da.concatenate(padded_data, axis=0))
        logging.debug(
            f"Merged timepoints shape for region {region}: {merged_data.shape}"
        )
        return merged_data

    def pad_to_largest(
        self, array: da.Array, target_shape: tuple[int, ...]
    ) -> da.Array:
        """Pad array to match target shape."""
        assert len(array.shape) == len(target_shape)
        if array.shape == target_shape:
            return array
        pad_widths = [(0, max(0, ts - s)) for s, ts in zip(array.shape, target_shape)]
        return cast(
            da.Array, da.pad(array, pad_widths, mode="constant", constant_values=0)
        )

    def create_hcs_ome_zarr_per_timepoint(self) -> None:
        """Create separate HCS OME-ZARR files for each timepoint."""
        for t in self.computed_parameters.timepoints:
            output_path = self.paths.merged_hcs_output(t)

            store = ome_zarr.io.parse_url(output_path, mode="w").store
            root = zarr.group(store=store)

            # Write plate metadata
            rows = sorted(set(region[0] for region in self.computed_parameters.regions))
            columns = sorted(
                set(region[1:] for region in self.computed_parameters.regions)
            )
            well_paths = [
                f"{well_id[0]}/{well_id[1:]}"
                for well_id in sorted(self.computed_parameters.regions)
            ]

            acquisitions = [
                {"id": 0, "maximumfieldcount": 1, "name": f"Timepoint {t} Acquisition"}
            ]

            ome_zarr.writer.write_plate_metadata(
                root,
                rows=rows,
                columns=[str(col) for col in columns],
                wells=well_paths,
                acquisitions=acquisitions,
                name=f"HCS Dataset - Timepoint {t}",
                field_count=1,
            )

            # Process each region (well) for this timepoint
            for region in self.computed_parameters.regions:
                # Load existing timepoint-region data
                region_path = self.paths.per_timepoint_region_output(t, region)

                if not os.path.exists(region_path):
                    logging.warning(
                        f"Warning: Missing data for timepoint {t}, region {region}"
                    )
                    continue

                # Load data from existing zarr
                z = zarr.open(region_path, mode="r")
                data = da.from_array(z["0"])

                # Create well hierarchy
                row, col = region[0], region[1:]
                row_group = root.require_group(row)
                well_group = row_group.require_group(col)

                # Write well metadata
                ome_zarr.writer.write_well_metadata(
                    well_group, images=[{"path": "0", "acquisition": 0}]
                )

                # Write image data
                image_group = well_group.require_group("0")

                # Prepare dataset and transformation metadata
                datasets = [
                    {
                        "path": str(i),
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": [
                                    1,
                                    1,
                                    self.computed_parameters.acquisition_params.get(
                                        "dz(um)", 1
                                    ),
                                    self.computed_parameters.pixel_size_um * (2**i),
                                    self.computed_parameters.pixel_size_um * (2**i),
                                ],
                            }
                        ],
                    }
                    for i in range(self.computed_parameters.num_pyramid_levels)
                ]

                axes = [
                    {"name": "t", "type": "time", "unit": "second"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ]

                # Write multiscales metadata
                ome_zarr.writer.write_multiscales_metadata(
                    image_group, datasets, axes=axes, name=f"Well_{region}_t{t}"
                )

                # Generate and write pyramid
                pyramid = self.generate_pyramid(
                    data, self.computed_parameters.num_pyramid_levels
                )
                storage_options = {"chunks": self.computed_parameters.chunks}

                ome_zarr.writer.write_multiscale(
                    pyramid=pyramid,
                    group=image_group,
                    axes=axes,
                    coordinate_transformations=[
                        d["coordinateTransformations"] for d in datasets
                    ],
                    storage_options=storage_options,
                    name=f"Well_{region}_t{t}",
                )

                # Add OMERO metadata
                image_group.attrs["omero"] = {
                    "name": f"Well_{region}_t{t}",
                    "version": "0.4",
                    "channels": [
                        {
                            "label": name,
                            "color": f"{color:06X}",
                            "window": {
                                "start": 0,
                                "end": np.iinfo(self.computed_parameters.dtype).max,
                            },
                        }
                        for name, color in zip(
                            self.computed_parameters.monochrome_channels,
                            self.computed_parameters.monochrome_colors,
                        )
                    ],
                }

            if t == self.computed_parameters.timepoints[-1]:
                self.callbacks.finished_saving(
                    str(output_path), self.computed_parameters.dtype
                )

    def create_complete_hcs_ome_zarr(self) -> None:
        """Create complete HCS OME-ZARR with merged timepoints."""
        output_path = self.paths.complete_hcs_output

        store = ome_zarr.io.parse_url(output_path, mode="w").store
        root = zarr.group(store=store)

        # Write plate metadata with correct parameters
        rows = sorted(set(region[0] for region in self.computed_parameters.regions))
        columns = sorted(set(region[1:] for region in self.computed_parameters.regions))
        well_paths = [
            f"{well_id[0]}/{well_id[1:]}"
            for well_id in sorted(self.computed_parameters.regions)
        ]

        acquisitions = [
            {"id": 0, "maximumfieldcount": 1, "name": "Stitched Acquisition"}
        ]

        ome_zarr.writer.write_plate_metadata(
            root,
            rows=rows,
            columns=[str(col) for col in columns],
            wells=well_paths,
            acquisitions=acquisitions,
            name="Complete HCS Dataset",
            field_count=1,
        )

        # Process each region (well)
        for region in self.computed_parameters.regions:
            # Load and merge timepoints for this region
            merged_data = self.load_and_merge_timepoints(region)

            # Create well hierarchy
            row, col = region[0], region[1:]
            row_group = root.require_group(row)
            well_group = row_group.require_group(col)

            # Write well metadata
            ome_zarr.writer.write_well_metadata(
                well_group, images=[{"path": "0", "acquisition": 0}]
            )

            # Write image data
            image_group = well_group.require_group("0")

            # Write multiscales metadata first
            datasets = [
                {
                    "path": str(i),
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": [
                                1,
                                1,
                                self.computed_parameters.acquisition_params.get(
                                    "dz(um)", 1
                                ),
                                self.computed_parameters.pixel_size_um * (2**i),
                                self.computed_parameters.pixel_size_um * (2**i),
                            ],
                        }
                    ],
                }
                for i in range(self.computed_parameters.num_pyramid_levels)
            ]

            axes = [
                {"name": "t", "type": "time", "unit": "second"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ]

            ome_zarr.writer.write_multiscales_metadata(
                image_group, datasets, axes=axes, name=f"Well_{region}"
            )

            # Generate and write pyramid data
            pyramid = self.generate_pyramid(
                merged_data, self.computed_parameters.num_pyramid_levels
            )
            storage_options = {"chunks": self.computed_parameters.chunks}

            ome_zarr.writer.write_multiscale(
                pyramid=pyramid,
                group=image_group,
                axes=axes,
                coordinate_transformations=[
                    d["coordinateTransformations"] for d in datasets
                ],
                storage_options=storage_options,
                name=f"Well_{region}",
            )

            # Add OMERO metadata
            image_group.attrs["omero"] = {
                "name": f"Well_{region}",
                "version": "0.4",
                "channels": [
                    {
                        "label": name,
                        "color": f"{color:06X}",
                        "window": {
                            "start": 0,
                            "end": np.iinfo(self.computed_parameters.dtype).max,
                        },
                    }
                    for name, color in zip(
                        self.computed_parameters.monochrome_channels,
                        self.computed_parameters.monochrome_colors,
                    )
                ],
            }

        self.callbacks.finished_saving(str(output_path), self.computed_parameters.dtype)

    def run(self) -> None:
        """Main execution method handling timepoints and regions."""
        stime = time.time()
        # Initial setup
        self.paths.output_folder.mkdir(exist_ok=True, parents=True)

        if self.params.apply_flatfield:
            raise NotImplementedError("Flatfield correction is not yet implemented.")

        # Calculate registration shifts once if using registration
        if self.params.use_registration:
            raise NotImplementedError("Registration is not yet implemented.")

        # Process each timepoint and region
        for timepoint in self.computed_parameters.timepoints:
            timepoint = int(timepoint)

            ttime = time.time()
            logging.info(f"\nProcessing timepoint {timepoint}")

            # Create timepoint output directory
            t_output_dir = self.paths.per_timepoint_dir(timepoint)
            t_output_dir.mkdir(exist_ok=True, parents=True)

            for region in self.computed_parameters.regions:
                rtime = time.time()
                logging.info(f"Processing region {region}...")

                # Stitch region
                self.callbacks.starting_stitching()
                stitched_region = self.stitch_region(timepoint, region)

                # Save the region
                self.callbacks.starting_saving(False)
                if self.params.output_format == OutputFormat.ome_zarr:
                    output_path = self.save_region_ome_zarr(
                        timepoint, region, stitched_region
                    )
                else:
                    assert self.params.output_format == OutputFormat.ome_tiff
                    output_path = self.save_region_aics(
                        timepoint, region, stitched_region
                    )

                logging.info(
                    f"Completed region {region} (saved to {output_path}): {time.time() - rtime}"
                )

            logging.info(f"Completed timepoint {timepoint}: {time.time() - ttime}")

        # Post-processing based on merge settings
        post_time = time.time()
        self.callbacks.starting_saving(True)

        if self.params.merge_timepoints and self.params.merge_hcs_regions:
            logging.info("Creating complete HCS OME-ZARR with merged timepoints...")
            self.create_complete_hcs_ome_zarr()
        elif self.params.merge_timepoints:
            logging.info("Creating merged timepoints OME-ZARR...")
            self.merge_timepoints_per_region()
        elif self.params.merge_hcs_regions:
            logging.info("Creating HCS OME-ZARR per timepoint...")
            self.create_hcs_ome_zarr_per_timepoint()
        else:
            # Emit finished signal with the last saved path
            final_path = self.paths.per_timepoint_region_output(
                self.computed_parameters.timepoints[-1],
                self.computed_parameters.regions[-1],
            )
            self.callbacks.finished_saving(
                str(final_path), self.computed_parameters.dtype
            )

        logging.info(f"Post-processing time: {time.time() - post_time}")
        logging.info(f"Total processing time: {time.time() - stime}")
