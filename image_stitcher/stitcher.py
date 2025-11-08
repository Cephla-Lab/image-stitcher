import functools
import logging
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeAlias

import dask.array as da
import numpy as np
import ome_zarr
import ome_zarr.io
import ome_zarr.writer
import ome_zarr.format
import psutil
import skimage
import zarr
import zarr.storage
from aicsimageio import types as aics_types
from aicsimageio.writers import OmeTiffWriter
from tqdm import tqdm

from . import flatfield_correction
from .benchmarking_util import debug_timing
from .parameters import (
    AcquisitionMetadata,
    MetaKey,
    OutputFormat,
    StitchingComputedParameters,
    StitchingParameters,
    ZLayerSelection,
)
from .z_layer_selection import (
    ZLayerSelector,
    create_z_layer_selector,
    filter_metadata_by_z_layers,
)


@dataclass
class ProgressCallbacks:
    update_progress: Callable[[int, int], None]
    getting_flatfields: Callable[[], None]
    starting_stitching: Callable[[], None]
    starting_saving: Callable[[bool], None]
    finished_saving: Callable[[str, object], None]

    @classmethod
    def no_op(cls):
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
        self.tqdm_class = tqdm

        self.metadata: dict[MetaKey, AcquisitionMetadata] = {}
        self.z_selector: ZLayerSelector | None = None

    @property
    def paths(self) -> Paths:
        if self._paths is None:
            self._paths = Paths(self.params.stitched_folder, self.params.output_format)
        return self._paths

    @staticmethod
    def compute_mip(tiles: list[np.ndarray]) -> np.ndarray:
        """Compute Maximum Intensity Projection from a list of z-stack tiles.
        
        Args:
            tiles: List of 2D or 3D numpy arrays representing different z-levels
            
        Returns:
            2D numpy array representing the MIP
        """
        if not tiles:
            raise ValueError("Cannot compute MIP from empty tile list")
        
        # Stack all tiles along a new axis (z-axis)
        if len(tiles[0].shape) in (2, 3):
            # 2D (grayscale) or 3D (RGB/multi-channel) tiles
            stacked = np.stack(tiles, axis=0)
            return np.max(stacked, axis=0)
        else:
            raise ValueError(f"Unexpected tile shape: {tiles[0].shape}")

    def load_image(self, tile_info) -> np.ndarray:
        """Load an image from file, handling single files, multi-page TIFF, and OME-TIFF files."""
        if hasattr(tile_info, 'frame_idx') and tile_info.frame_idx is not None:
            # Check if it's an OME-TIFF (frame_idx is tuple) or multi-page TIFF (frame_idx is int)
            if isinstance(tile_info.frame_idx, tuple):
                # OME-TIFF: frame_idx is (channel_idx, z_idx)
                from .image_loaders import create_image_loader
                loader = create_image_loader(tile_info.filepath, format_hint='ome_tiff')
                channel_idx, z_idx = tile_info.frame_idx
                return loader.read_slice(channel=channel_idx, z=z_idx)
            elif tile_info.frame_idx > 0:
                # Multi-page TIFF file: frame_idx is page number
                import tifffile
                with tifffile.TiffFile(tile_info.filepath) as tif:
                    return tif.pages[tile_info.frame_idx].asarray()
            else:
                # frame_idx is 0, treat as single file
                return skimage.io.imread(tile_info.filepath)
        else:
            # Single file
            return skimage.io.imread(tile_info.filepath)

    def create_output_array(
        self, timepoint: int, region: str, num_z_layers: int
    ) -> AnyArray:
        width, height = self.computed_parameters.calculate_output_dimensions(
            timepoint, region
        )
        # Use provided num_z_layers
        z_dimension = 1 if self.params.apply_mip else num_z_layers

        # create zeros with the right shape/dtype per timepoint per region
        output_shape = (
            1,
            self.computed_parameters.num_c,
            z_dimension,
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
            
            # Detect zarr version and use appropriate store API
            zarr_major_version = int(zarr.__version__.split('.')[0])
            
            if zarr_major_version >= 3:
                # zarr v3: use LocalStore
                store = zarr.storage.LocalStore(str(output_path))
            else:
                # zarr v2: use DirectoryStore
                store = zarr.storage.DirectoryStore(str(output_path))
            
            root = zarr.group(store=store)
            return root.zeros(
                name="0",
                shape=output_shape,
                dtype=self.computed_parameters.dtype,
                chunks=chunks,
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

    def apply_flatfield_correction(
        self, tile: np.ndarray, channel_idx: int
    ) -> np.ndarray:
        """Apply a precomputed flatfield correction to an image tile."""
        if channel_idx in self.computed_parameters.flatfields:
            return (
                (tile / self.computed_parameters.flatfields[channel_idx])
                .clip(
                    min=np.iinfo(self.computed_parameters.dtype).min,
                    max=np.iinfo(self.computed_parameters.dtype).max,
                )
                .astype(self.computed_parameters.dtype)
            )
        else:
            logging.warning(f"No flatfield correction found for channel #{channel_idx}")
            return tile

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
        if self.params.apply_flatfield:
            logging.debug("Applying flatfield to channel_idx: %s", channel_idx)
            tile = self.apply_flatfield_correction(tile, channel_idx)

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
        full_metadata_region = self.computed_parameters.get_region_metadata(
            int(timepoint), region
        )

        # Apply z-layer selection
        assert self.z_selector is not None # Should be initialized in run()
        selected_z_layers = self.z_selector.select_z_layers(
            full_metadata_region, self.computed_parameters.num_z
        )
        # Create a mapping from original z-index to new z-index
        # Ensure selected_z_layers are sorted for consistent z_index_map
        z_index_map = {z: i for i, z in enumerate(sorted(list(selected_z_layers)))}
        
        self.metadata = filter_metadata_by_z_layers(full_metadata_region, selected_z_layers)
        
        logging.info(
            f"Using z-layer selection strategy '{self.z_selector.get_name()}': "
            f"selected layers {sorted(list(selected_z_layers))}"
        )

        # Create output array with appropriate z-dimension
        stitched_region = self.create_output_array(timepoint, region, len(selected_z_layers))
        x_min = min(self.computed_parameters.x_positions)
        y_min = min(self.computed_parameters.y_positions)
        total_tiles = len(self.metadata)
        processed_tiles = 0
        logging.info(
            f"Beginning stitching of {total_tiles} tiles for region {region} timepoint {timepoint}"
        )

        if self.params.apply_mip:
            logging.info(f"Applying Maximum Intensity Projection (MIP) to z-stacks")
            # Group tiles by (t, region, fov_idx, channel) for MIP processing
            tile_groups = {}
            for key, tile_info in self.metadata.items():
                t, region_name, fov_idx, z_level, channel = key
                group_key = (t, region_name, fov_idx, channel)
                if group_key not in tile_groups:
                    tile_groups[group_key] = []
                tile_groups[group_key].append((z_level, tile_info))

            # Process each group with MIP
            for group_key, z_tiles in tile_groups.items():
                t, region_name, fov_idx, channel = group_key
                
                # Load all tiles for the current FOV/channel group
                # The order of tiles does not matter for compute_mip
                tiles = []
                for z_level, tile_info in z_tiles:
                    tile = self.load_image(tile_info)
                    tiles.append(tile)
                
                # Compute MIP
                mip_tile = self.compute_mip(tiles)
                
                # Get position from the first tile in the group
                first_tile_info = z_tiles[0][1]
                x_pixel = int(
                    (first_tile_info.x - x_min) * 1000 / self.computed_parameters.pixel_size_um
                )
                y_pixel = int(
                    (first_tile_info.y - y_min) * 1000 / self.computed_parameters.pixel_size_um
                )

                # Place MIP tile at z_level=0 (since we only have 1 z-level in output)
                self.place_tile(stitched_region, mip_tile, x_pixel, y_pixel, 0, channel)

                self.callbacks.update_progress(processed_tiles, total_tiles)
                processed_tiles += len(z_tiles)
        else:
            # Original processing logic for non-MIP case
            for key, tile_info in self.metadata.items():
                t, _, _, z_level, channel = key
                tile = self.load_image(tile_info)

                x_pixel = int(
                    (tile_info.x - x_min) * 1000 / self.computed_parameters.pixel_size_um
                )
                y_pixel = int(
                    (tile_info.y - y_min) * 1000 / self.computed_parameters.pixel_size_um
                )

                # Map z_level to output array index
                output_z_level = z_index_map[z_level]

                self.place_tile(stitched_region, tile, x_pixel, y_pixel, output_z_level, channel)

                self.callbacks.update_progress(processed_tiles, total_tiles)
                processed_tiles += 1

        logging.info(
            f"Time to stitch region {region} timepoint {timepoint}: {time.time() - start_time}"
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
        # Detect zarr version and configure compression accordingly
        zarr_major_version = int(zarr.__version__.split('.')[0])
        
        if zarr_major_version >= 3:
            # zarr v3: use codecs
            if self.params.output_compression == "none":
                codecs_list = None
            else:  # "default"
                try:
                    codecs_list = [zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")]
                except AttributeError:
                    codecs_list = None
            compressor_v2 = None
        else:
            # zarr v2: use compressor
            codecs_list = None
            if self.params.output_compression == "none":
                compressor_v2 = None
            else:  # "default"
                try:
                    from numcodecs import Blosc
                    compressor_v2 = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
                except ImportError:
                    compressor_v2 = None
        
        storage_opts = {
            "chunks": self.computed_parameters.chunks,
        }
        if zarr_major_version < 3 and compressor_v2 is not None:
            storage_opts["compressor"] = compressor_v2

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
                # Write to zarr array - compatible with both v2 and v3
                array_name = str(pyramid_idx)
                
                if zarr_major_version >= 3:
                    # zarr v3: use create_array with compressors parameter
                    arr = root.create_array(
                        name=array_name,
                        shape=pyramid[pyramid_idx].shape,
                        dtype=pyramid[pyramid_idx].dtype,
                        chunks=storage_opts["chunks"],
                        compressors=codecs_list if codecs_list else None,
                    )
                else:
                    # zarr v2: use zeros or create_dataset with compressor parameter
                    arr = root.zeros(
                        name=array_name,
                        shape=pyramid[pyramid_idx].shape,
                        dtype=pyramid[pyramid_idx].dtype,
                        chunks=storage_opts["chunks"],
                        compressor=storage_opts.get("compressor"),
                    )
                
                # Store the dask array data into the zarr array
                da.store(pyramid[pyramid_idx], arr, compute=True)
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

            # TODO/NOTE(imo): Without re-chunking here, we'd get errors downstream because some operations
            # want "regular chunks" (aka: all chunks the same size).  We use the same chunk size for all levels
            # of the pyramid so that we're consistent across the board, but that isn't necessary.
            pyramid.append(downsampled.rechunk(chunks=self.computed_parameters.chunks))
        return pyramid

    def run(self) -> None:
        """Main execution method handling timepoints and regions."""
        stime = time.time()
        # Initial setup
        self.paths.output_folder.mkdir(exist_ok=True, parents=True)

        if self.params.apply_flatfield:
            # Check if flatfields were already computed (e.g., by registration)
            acquisition_folder = pathlib.Path(self.params.input_folder)
            auto_flatfield_manifest = acquisition_folder / "flatfields" / "flatfield_manifest.json"
            
            logging.info(f"Flatfield loading: params.flatfield_manifest={self.params.flatfield_manifest}, auto_path={auto_flatfield_manifest}, exists={auto_flatfield_manifest.exists()}")
            
            # Priority: explicit manifest > auto-discovered > compute new
            if self.params.flatfield_manifest:
                # User explicitly specified a manifest
                from .flatfield_utils import load_flatfield_correction
                logging.info(f"Loading flatfields from explicit manifest: {self.params.flatfield_manifest}")
                self.computed_parameters.flatfields = load_flatfield_correction(
                    self.params.flatfield_manifest,
                    self.computed_parameters,
                )
            elif auto_flatfield_manifest.exists():
                # Flatfields exist from previous computation (e.g., registration)
                from .flatfield_utils import load_flatfield_correction
                logging.info(f"Loading existing flatfields from: {auto_flatfield_manifest}")
                try:
                    loaded_flatfields = load_flatfield_correction(
                        auto_flatfield_manifest,
                        self.computed_parameters,
                    )
                    if loaded_flatfields:
                        self.computed_parameters.flatfields = loaded_flatfields
                        logging.info("Successfully loaded existing flatfields")
                    else:
                        logging.warning("Flatfield manifest exists but no valid flatfields loaded. Computing new ones.")
                        self.computed_parameters.flatfields = None
                except Exception as e:
                    logging.warning(f"Failed to load existing flatfields: {e}. Computing new ones.")
                    # Fall through to compute new flatfields
                    self.computed_parameters.flatfields = None
            
            # Compute flatfields if not loaded
            if not self.computed_parameters.flatfields:
                logging.info("Computing new flatfield corrections")
                self.computed_parameters.flatfields = (
                    flatfield_correction.compute_flatfield_correction(
                        self.computed_parameters,
                        self.callbacks.getting_flatfields,
                    )
                )
                
                # Save the computed flatfields to the acquisition folder
                if self.computed_parameters.flatfields:
                    from .flatfield_utils import save_flatfield_correction
                    flatfield_dir = acquisition_folder / "flatfields"
                    
                    try:
                        manifest_path = save_flatfield_correction(
                            self.computed_parameters.flatfields,
                            self.computed_parameters,
                            flatfield_dir,
                        )
                        logging.info(f"Saved computed flatfields to {flatfield_dir}")
                        logging.info(f"Flatfield manifest created at {manifest_path}")
                    except Exception as e:
                        logging.error(f"Failed to save computed flatfields: {e}")
                        # Continue processing even if saving fails

            # Validate loaded/computed flatfields
            if self.computed_parameters.flatfields:
                expected_shape = (
                    self.computed_parameters.input_height,
                    self.computed_parameters.input_width,
                )
                # Iterate over a copy of keys for safe removal during iteration
                for ch_idx in list(self.computed_parameters.flatfields.keys()):
                    ff_array = self.computed_parameters.flatfields[ch_idx]
                    if not isinstance(ff_array, np.ndarray):
                        logging.warning(
                            f"Flatfield for channel index {ch_idx} is not a numpy array (type: {type(ff_array)}). "
                            "This flatfield will not be used."
                        )
                        del self.computed_parameters.flatfields[ch_idx]
                        continue # Skip to next flatfield

                    if ff_array.ndim != 2 or ff_array.shape != expected_shape:
                        logging.warning(
                            f"Flatfield for channel index {ch_idx} has incorrect shape {ff_array.shape}. "
                            f"Expected 2D array of shape {expected_shape}. "
                            "This flatfield will not be used."
                        )
                        del self.computed_parameters.flatfields[ch_idx]
                    # Optional: Add further checks like dtype or value range if necessary

            # Log loaded/computed AND VALIDATED flatfield indices here
            if self.computed_parameters.flatfields:
                logging.debug(
                    "Validated and using flatfields for channel indices: %s",
                    list(self.computed_parameters.flatfields.keys()),
                )
            else:
                logging.debug(
                    "Flatfield application was enabled, but no valid flatfields were loaded, computed, or passed validation."
                )

        # Initialize z-layer selector
        self.z_selector = create_z_layer_selector(self.params.z_layer_selection)
        logging.info(f"Using z-layer selection strategy: {self.z_selector.get_name()}")

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
                    chunk_shapes = self.computed_parameters.chunks
                    logging.debug(
                        f"Re-chunking to make region has {chunk_shapes} chunks."
                    )
                    if isinstance(stitched_region, np.ndarray):
                        dask_stitched_region = da.from_array(
                            stitched_region,
                            chunks=chunk_shapes,
                            name=f"stitched:t={timepoint},r={region}",
                        )
                    elif isinstance(stitched_region, da.Array):
                        dask_stitched_region = stitched_region.rechunk(chunk_shapes)
                    else:
                        # NOTE(imo): Do we need a separate Zarr re-chunk case here?
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

    def stitch_all_regions_and_timepoints(self) -> None:
        """Main entrypoint to stitch all regions and timepoints based on parameters."""
        logging.info(
            f"Stitching all regions and timepoints, output: {self.params.output_format.value}"
        )

        self.z_selector = create_z_layer_selector(self.params.z_layer_selection)
        logging.info(f"Using z-layer selection strategy: {self.z_selector.get_name()}")

        # Loop over timepoints and regions
        for t_idx, t in enumerate(self.computed_parameters.timepoints):
            for r_idx, region in enumerate(self.computed_parameters.regions):
                self.stitch_region(t, region)
