import enum
import json
import logging
import math
import os
import pathlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, ClassVar, Literal, NamedTuple, Optional, Union
import numpy as np
import pandas as pd
import tifffile
from dask_image.imread import imread as dask_imread
from pydantic import AfterValidator, BaseModel, Field, computed_field, ConfigDict

from .z_layer_selection import ZLayerSelector

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S.%f"


class OutputFormat(enum.Enum):
    ome_zarr = ".ome.zarr"
    ome_tiff = ".ome.tiff"


class ScanPattern(enum.Enum):
    unidirectional = "Unidirectional"
    s_pattern = "S-Pattern"


class ZLayerSelection(enum.Enum):
    ALL = "all"
    MIDDLE = "middle"


def input_path_exists(path: str) -> str:
    """Pydantic validator to check the path exists."""
    if not os.path.exists(path):
        raise ValueError(f"Input folder does not exist: {path}")

    return path


class StitchingParameters(
    BaseModel,
    use_attribute_docstrings=True,
):
    """Parameters for microscopy image stitching operations."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        exclude={"z_layer_selector"},
        json_schema_extra={"exclude": {"z_layer_selector"}}
    )

    input_folder: Annotated[str, AfterValidator(input_path_exists)]
    """A folder on the local machine containing an image acqusition.

    This should be in the latest format output by the Cephla microscopes. 
    """

    # Output configuration
    output_format: OutputFormat = OutputFormat.ome_zarr
    """Output format for the stitched data."""

    # Scanning and stitching configuration
    scan_pattern: ScanPattern = ScanPattern.unidirectional
    """The scan pattern used for the acquisition.

    unidirectional means all rows go the same direction; S-pattern indicates every
    other row goes the other direction.
    """

    z_layer_selection: Union[ZLayerSelection, int] = ZLayerSelection.MIDDLE
    """Strategy for selecting z-layers to stitch.

    Accepts:
    - `ZLayerSelection.ALL` (input as string: "all"): Stitch all z-layers.
    - `ZLayerSelection.MIDDLE` (input as string: "middle"): Stitch only the middle z-layer from the stack (default).
    - An integer (e.g., `2`; or input as string like "2"): Stitch a specific z-layer by its 0-based index.
    
    When providing via JSON or similar string-based configuration, use "all", "middle", 
    or a string/number for the layer index (e.g., "0", "1", 2).
    Internally, these are converted to `ZLayerSelection` enum members or `int`.
    
    Future options could include "max_intensity", "user_selected", etc.
    """

    apply_flatfield: bool = False
    """Whether to apply a flatfield correction to the images prior to stitching."""

    apply_mip: bool = False
    """Whether to apply Maximum Intensity Projection (MIP) to z-stacks before stitching."""

    flatfield_manifest: Optional[pathlib.Path] = None
    """If set, a path (folder or .json) from which to load precomputed flatfields."""

    verbose: bool = False
    """Show debug-level logging."""

    force_stitch_to_disk: bool = False
    """If true, force using a disk-backed stitching implementation regardless of input size.

    (Otherwise, we use a disk-based implementation only if the full image
    doesn't fit into memory.)

    This can be useful for debugging and testing, or if you want to keep memory
    usage as low as possible, at the cost of some processing speed.
    """

    num_pyramid_levels: int | None = None
    """Total number of pyramid levels (including the full-resolution one) in the output.

    Ignored if not writing to ome-zarr as the output format. The default, `None`
    means we infer the number of output levels based on the size of the input images.
    """

    output_compression: Literal["default", "none"] = "default"
    """Override the compression for the output zarr array.

    (Ignored unless the output type is ome-zarr.) Currently the only options are
    "none" for uncompressed output and "default" to use the zarr library's
    default compression algorithm.
    """

    @computed_field
    @property
    def z_layer_selector(self) -> ZLayerSelector:
        """The ZLayerSelector instance that will be used for selecting z-layers."""
        from .z_layer_selection import create_z_layer_selector
        return create_z_layer_selector(self.z_layer_selection)

    @property
    def stitched_folder(self) -> pathlib.Path:
        """Path to folder containing stitched outputs."""
        return pathlib.Path(
            self.input_folder + "_stitched_" + datetime.now().strftime(DATETIME_FORMAT)
        )

    @classmethod
    def from_json_file(cls, json_path: str) -> "StitchingParameters":
        """Create parameters from a JSON file.

        Args:
            json_path: Path to JSON file containing parameters

        Returns:
            StitchingParameters: New instance with values from JSON
        """
        with open(json_path) as f:
            return cls.model_validate_json(f.read())

    def to_json_file(self, json_path: str) -> None:
        """Save parameters to a JSON file.

        Args:
            json_path: Path where JSON file should be saved
        """
        with open(json_path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    def model_dump(self, **kwargs) -> dict:
        """Override model_dump to exclude z_layer_selector."""
        kwargs["exclude"] = {"z_layer_selector"}
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs) -> str:
        """Override model_dump_json to exclude z_layer_selector."""
        kwargs["exclude"] = {"z_layer_selector"}
        return super().model_dump_json(**kwargs)


@dataclass
class UnidirectionalScanPatternParams:
    """Computed parameters for a unidirectional scan pattern."""

    h_shift: tuple[int, int] = (0, 0)
    v_shift: tuple[int, int] = (0, 0)


class ReverseRows(enum.Enum):
    """Whether an S-pattern reverses even or odd rows."""

    even = "even"
    odd = "odd"

    def is_reversed(self, row_idx: int) -> bool:
        if self == ReverseRows.even:
            return row_idx % 2 == 0
        elif self == ReverseRows.odd:
            return row_idx % 2 == 1
        else:
            raise RuntimeError(f"Unexpected ReverseRows value: {self}")


@dataclass
class SPatternScanParams:
    """Computed parameters for an S-pattern scan pattern."""

    h_shift: tuple[int, int] = (0, 0)
    v_shift: tuple[int, int] = (0, 0)
    h_shift_rev: tuple[int, int] = (0, 0)
    h_shift_rev_rows: ReverseRows = ReverseRows.even


ScanParams = UnidirectionalScanPatternParams | SPatternScanParams


def default_scan_params(pattern: ScanPattern) -> ScanParams:
    if pattern == ScanPattern.unidirectional:
        return UnidirectionalScanPatternParams()
    elif pattern == ScanPattern.s_pattern:
        return SPatternScanParams()
    else:
        raise RuntimeError(f"Unexpected ScanPattern value: {pattern}")


class MetaKey(NamedTuple):
    """A (timepoint, region, fov_idx, z_level, channel) key."""

    t: int
    region: str
    fov: int
    z_level: int
    channel: str


@dataclass
class AcquisitionMetadata:
    filepath: pathlib.Path
    """Path to the image file."""
    x: float
    """X position of the stage in mm."""
    y: float
    """Y position of the stage in mm."""
    z: float
    """Z position of the stage in µm."""
    channel: str
    """The name of the channel."""
    z_level: int
    """The z-index of the current plane when z-sectioning a single field of view."""
    region: str
    fov_idx: int
    """The index of the current field of view."""
    t: int
    """The current timepoint."""
    frame_idx: int = 0
    """The frame index within a multi-page TIFF file (0 for single-page files)."""

    @property
    def key(self) -> MetaKey:
        """A (timepoint, region, fov_idx, z_level, channel) key.

        This uniquely identifies this image within an acquisition.
        """
        return MetaKey(self.t, self.region, self.fov_idx, self.z_level, self.channel)


class ImagePlaneDims(NamedTuple):
    width_px: int
    height_px: int


@dataclass
class StitchingComputedParameters:
    """Parameters computed from other parameters, images, or files on disk.

    TODO(colin): refactor to remove all the defaults that end up never used.
    """

    CHUNK_SIZE_LIMIT_PX: ClassVar[int] = (
        12288  # 12288 = 8192 + 4096, just arbitrarily chosen multiples of 1024
    )
    """An upper bound on the size of a dask array chunk.
    
    We assume many chunks can fit into memory at once, and for a single square
    plane in float64, this works out to about 1GB/chunk.
    """
    parent: StitchingParameters
    scan_params: ScanParams = field(init=False)
    pixel_size_um: int = field(init=False)
    pixel_binning: int = field(init=False)
    num_z: int = field(init=False)
    num_c: int = field(init=False)
    num_t: int = field(init=False)
    input_height: int = field(init=False)
    input_width: int = field(init=False)
    num_pyramid_levels: int = field(init=False)
    acquisition_params: dict[str, Any] = field(init=False)
    timepoints: list[int] = field(init=False)
    monochrome_channels: list[str] = field(init=False)
    monochrome_colors: list[int] = field(init=False)
    flatfields: dict[int, np.ndarray] = field(init=False)
    acquisition_metadata: dict[MetaKey, AcquisitionMetadata] = field(init=False)
    dtype: np.dtype = field(init=False)
    chunks: tuple[int, int, int, int, int] = field(init=False)
    xy_positions: list[tuple[float, float]] = field(init=False)
    regions: list[str] = field(default_factory=list)
    channel_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.scan_params = default_scan_params(self.parent.scan_pattern)
        self.init_timepoints()
        self.init_acquisition_parameters()
        self.init_pixel_size()
        
        # Detect image format
        self.image_format = self.detect_image_format()
        
        # Choose parsing method based on detected format
        if self.image_format == 'ome_tiff':
            logging.info("Detected OME-TIFF format, using parse_ome_tiff")
            self.parse_ome_tiff()
        elif self.image_format == 'multipage_tiff':
            logging.info("Detected multi-page TIFF format, using parse_multipage_tiff")
            self.parse_multipage_tiff()
        else:
            logging.info("Detected individual file format, using parse_acquisition_metadata")
            self.parse_acquisition_metadata()

    def init_timepoints(self) -> None:
        self.timepoints = [
            int(d, 10)
            for d in os.listdir(self.parent.input_folder)
            if os.path.isdir(os.path.join(self.parent.input_folder, d)) and d.isdigit()
        ]
        self.timepoints.sort()

    def detect_image_format(self) -> str:
        """Detect the image format used in the acquisition.
        
        Returns:
            One of: 'ome_tiff', 'multipage_tiff', 'individual_files'
        """
        if not self.timepoints:
            return 'individual_files'
        
        # Check the first timepoint directory
        first_timepoint = self.timepoints[0]
        image_folder = pathlib.Path(self.parent.input_folder) / str(first_timepoint)
        
        if not image_folder.exists():
            return 'individual_files'
        
        # Use the image_loaders module for detection
        from .image_loaders import detect_image_format
        return detect_image_format(image_folder)
    
    def is_multipage_tiff_format(self) -> bool:
        """Check if the acquisition uses multi-page TIFF format (deprecated, use detect_image_format)."""
        return self.detect_image_format() == 'multipage_tiff'
    
    def is_ome_tiff_format(self) -> bool:
        """Check if the acquisition uses OME-TIFF format."""
        return self.detect_image_format() == 'ome_tiff'

    def init_acquisition_parameters(self) -> None:
        acquistion_params_path = os.path.join(
            self.parent.input_folder, "acquisition parameters.json"
        )
        with open(acquistion_params_path, "r") as file:
            self.acquisition_params = json.load(file)

    def init_pixel_size(self) -> None:
        """Initialize the pixel size parameters.

        This must be called after init_acquisition_params

        TODO(colin): refactor so that there are not implicit dependencies
        between these initialization functions.
        """
        obj_mag = self.acquisition_params["objective"]["magnification"]
        obj_tube_lens_mm = self.acquisition_params["objective"]["tube_lens_f_mm"]
        sensor_pixel_size_um = self.acquisition_params["sensor_pixel_size_um"]
        tube_lens_mm = self.acquisition_params["tube_lens_mm"]
        self.pixel_binning = self.acquisition_params.get("pixel_binning", 1)
        assert isinstance(self.pixel_binning, int)
        obj_focal_length_mm = obj_tube_lens_mm / obj_mag
        actual_mag = tube_lens_mm / obj_focal_length_mm
        self.pixel_size_um = sensor_pixel_size_um / actual_mag
        logging.info(f"pixel_size_um: {self.pixel_size_um}")

    def parse_ome_tiff(self) -> None:
        """Parse OME-TIFF files using the image_loaders module."""
        from .image_loaders import create_image_loader, parse_ome_tiff_filename
        
        input_path = pathlib.Path(self.parent.input_folder)
        self.acquisition_metadata = {}
        
        max_z = 0
        max_fov = 0
        
        # Iterate over each timepoint
        for timepoint in self.timepoints:
            image_folder = input_path / str(timepoint)
            coordinates_path = image_folder / "coordinates.csv"
            
            logging.info(f"Processing OME-TIFF timepoint {timepoint}, folder: {image_folder}")
            
            try:
                coordinates_df = pd.read_csv(coordinates_path)
            except FileNotFoundError:
                logging.warning(f"coordinates.csv not found for timepoint {timepoint}")
                continue
            
            # Find all OME-TIFF files
            ome_tiff_files = sorted([
                f.resolve() for f in image_folder.iterdir()
                if f.suffix.lower() in (".tif", ".tiff") and 
                (f.name.lower().endswith('.ome.tif') or f.name.lower().endswith('.ome.tiff'))
            ])
            
            logging.info(f"Found {len(ome_tiff_files)} OME-TIFF files in timepoint {timepoint}")
            
            # Process each OME-TIFF file
            for ome_file in ome_tiff_files:
                # Parse filename to get region and fov
                parsed = parse_ome_tiff_filename(ome_file.name)
                if not parsed:
                    logging.warning(f"Could not parse OME-TIFF filename: {ome_file.name}")
                    continue
                
                region = parsed['region']
                fov = parsed['fov']
                
                try:
                    # Create loader for this OME-TIFF file
                    loader = create_image_loader(ome_file, format_hint='ome_tiff')
                    meta = loader.metadata
                    
                    num_channels = meta['num_channels']
                    num_z_slices = meta['num_z']
                    channel_names = meta.get('channel_names') or [f"channel_{i}" for i in range(num_channels)]
                    
                    logging.debug(f"OME-TIFF {ome_file.name}: {num_channels} channels, {num_z_slices} z-slices")
                    
                    # Get coordinates for this region/fov
                    fov_coords = coordinates_df[
                        (coordinates_df['region'] == region) & 
                        (coordinates_df['fov'] == fov)
                    ]
                    
                    if fov_coords.empty:
                        logging.warning(f"No coordinates for {region}, FOV {fov}")
                        continue
                    
                    # Process each channel and z-slice combination
                    for channel_idx in range(num_channels):
                        channel_name = channel_names[channel_idx]
                        # Clean up channel name (remove "Channel_" prefix if present)
                        if channel_name.startswith('Channel_'):
                            channel_name = channel_name[8:]  # Remove "Channel_" prefix
                        channel_name = channel_name.replace("_", " ").replace("full ", "full_")
                        
                        for z_idx in range(num_z_slices):
                            # Find coordinate row for this z-level
                            # In OME-TIFF, z_idx corresponds to z_level in coordinates
                            coord_rows = fov_coords[fov_coords['z_level'] == z_idx]
                            
                            if coord_rows.empty:
                                # If no exact z_level match, use first row
                                if len(fov_coords) > 0:
                                    coord_row = fov_coords.iloc[0]
                                else:
                                    logging.warning(f"No coordinates for {region}, FOV {fov}, z={z_idx}")
                                    continue
                            else:
                                coord_row = coord_rows.iloc[0]
                            
                            # Create metadata object
                            # Store channel_idx and z_idx in frame_idx as tuple for later retrieval
                            meta_obj = AcquisitionMetadata(
                                filepath=ome_file,
                                x=coord_row["x (mm)"],
                                y=coord_row["y (mm)"],
                                z=coord_row["z (um)"],
                                channel=channel_name,
                                z_level=z_idx,
                                region=region,
                                fov_idx=fov,
                                t=timepoint,
                                frame_idx=(channel_idx, z_idx)  # Store as tuple for OME-TIFF
                            )
                            
                            self.acquisition_metadata[meta_obj.key] = meta_obj
                            
                            # Track regions and channels
                            self.regions.append(region)
                            self.channel_names.append(channel_name)
                            
                            # Update max values
                            max_z = max(max_z, z_idx)
                            max_fov = max(max_fov, fov)
                
                except Exception as e:
                    logging.error(f"Error processing OME-TIFF {ome_file.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Finalize metadata
        self.regions = sorted(set(self.regions))
        self.channel_names = sorted(set(self.channel_names))
        
        self.num_t = len(self.timepoints)
        self.num_z = max_z + 1
        self.num_fovs_per_region = max_fov + 1
        
        if not self.acquisition_metadata:
            logging.error("No OME-TIFF acquisition metadata found!")
            return
        
        # Set up image parameters from first image
        first_meta = list(self.acquisition_metadata.values())[0]
        
        try:
            loader = create_image_loader(first_meta.filepath, format_hint='ome_tiff')
            channel_idx, z_idx = first_meta.frame_idx
            first_image = loader.read_slice(channel=channel_idx, z=z_idx)
        except Exception as e:
            logging.error(f"Could not read first OME-TIFF image: {e}")
            return
        
        self.dtype = first_image.dtype
        if len(first_image.shape) == 2:
            self.input_height, self.input_width = first_image.shape
        else:
            raise ValueError(f"Unexpected OME-TIFF image shape: {first_image.shape}")
        
        # Set up chunks
        self.chunks = (
            1,
            1,
            1,
            min(self.input_height, self.CHUNK_SIZE_LIMIT_PX),
            min(self.input_width, self.CHUNK_SIZE_LIMIT_PX),
        )
        
        # Set up monochrome channels (OME-TIFF is already split by channel)
        self.monochrome_channels = self.channel_names.copy()
        self.num_c = len(self.monochrome_channels)
        self.monochrome_colors = [
            self.get_channel_color(name) for name in self.monochrome_channels
        ]
        
        # Log dataset info
        logging.info(f"OME-TIFF Dataset - Regions: {self.regions}, Channels: {self.channel_names}")
        logging.info(f"FOV dimensions: {self.input_height}x{self.input_width}")
        logging.info(f"{self.num_z} Z levels, {self.num_t} Time points, {self.num_c} Channels")

    def parse_acquisition_metadata(self) -> None:
        """Parse image filenames and matche them to coordinates for stitching.

        multiple channels, regions, timepoints, z levels

        Must be called after self.init_timepoints().

        TODO(colin): refactor so that there are not implicit dependencies
        between these initialization functions.
        """
        input_path = pathlib.Path(self.parent.input_folder)
        self.acquisition_metadata = {}
        max_z = 0
        max_fov = 0

        # Iterate over each timepoint
        for timepoint in self.timepoints:
            image_folder = input_path / str(timepoint)
            coordinates_path = image_folder / "coordinates.csv"

            logging.info(
                f"Processing timepoint {timepoint}, image folder: {image_folder}"
            )

            try:
                coordinates_df = pd.read_csv(coordinates_path)
            except FileNotFoundError:
                logging.warning(f"coordinates.csv not found for timepoint {timepoint}")
                continue

            # Process each image file
            image_files = sorted(
                [
                    f.resolve()
                    for f in image_folder.iterdir()
                    if f.suffix in (".bmp", ".tiff", ".tif", ".jpg", ".jpeg", ".png")
                    and "focus_camera" not in f.name
                ]
            )

            for file in image_files:
                parts = file.name.split("_", 3)
                region, fov, z_level, channel = (
                    parts[0],
                    int(parts[1]),
                    int(parts[2]),
                    os.path.splitext(parts[3])[0],
                )
                channel = channel.replace("_", " ").replace("full ", "full_")

                coord_rows = coordinates_df[
                    (coordinates_df["region"] == region)
                    & (coordinates_df["fov"] == fov)
                    & (coordinates_df["z_level"] == z_level)
                ]

                if coord_rows.empty:
                    logging.warning(f"No coordinates for {file}")
                    continue

                coord_row = coord_rows.iloc[0]

                meta = AcquisitionMetadata(
                    filepath=file,
                    x=coord_row["x (mm)"],
                    y=coord_row["y (mm)"],
                    z=coord_row["z (um)"],
                    channel=channel,
                    z_level=z_level,
                    region=region,
                    fov_idx=fov,
                    t=timepoint,
                )

                self.acquisition_metadata[meta.key] = meta

                # Add region and channel names to the sets
                self.regions.append(region)
                self.channel_names.append(channel)

                # Update max_z and max_fov values
                max_z = max(max_z, z_level)
                max_fov = max(max_fov, fov)

        # After processing all timepoints, finalize the list of regions and channels
        self.regions = sorted(set(self.regions))
        self.channel_names = sorted(set(self.channel_names))

        # Calculate number of timepoints (t), Z levels, and FOVs per region
        self.num_t = len(self.timepoints)
        self.num_z = max_z + 1
        self.num_fovs_per_region = max_fov + 1

        # When MIP is enabled, the number of input z-levels (self.num_z)
        # remains the count of z-levels in the source data.
        # The stitching process itself will handle producing a 1 z-level output
        # if MIP is active.

        # Set up image parameters based on the first image
        first_meta = list(self.acquisition_metadata.values())[0]
        first_image = dask_imread(first_meta.filepath)[0]

        self.dtype = first_image.dtype
        if len(first_image.shape) == 2:
            self.input_height, self.input_width = first_image.shape
        elif len(first_image.shape) == 3:
            self.input_height, self.input_width = first_image.shape[:2]
        else:
            raise ValueError(f"Unexpected image shape: {first_image.shape}")
        # TODO(colin): further tune the chunk size. Empirically, increasing from
        # 512x512 to the image size was a huge (25x or more in some cases)
        # speedup. Is even bigger better?
        self.chunks = (
            1,
            1,
            1,
            min(self.input_height, self.CHUNK_SIZE_LIMIT_PX),
            min(self.input_width, self.CHUNK_SIZE_LIMIT_PX),
        )

        # Set up final monochrome channels
        self.monochrome_channels = []
        for channel in self.channel_names:
            (t, region, fov, z_level, _) = first_meta.key
            channel_key = MetaKey(t, region, fov, z_level, channel)
            channel_image = dask_imread(
                self.acquisition_metadata[channel_key].filepath
            )[0]
            if len(channel_image.shape) == 3 and channel_image.shape[2] == 3:
                channel = channel.split("_")[0]
                self.monochrome_channels.extend(
                    [f"{channel}_R", f"{channel}_G", f"{channel}_B"]
                )
            else:
                self.monochrome_channels.append(channel)

        self.num_c = len(self.monochrome_channels)
        self.monochrome_colors = [
            self.get_channel_color(name) for name in self.monochrome_channels
        ]

        # Print out information about the dataset
        logging.info(f"Regions: {self.regions}, Channels: {self.channel_names}")
        logging.info(f"FOV dimensions: {self.input_height}x{self.input_width}")
        logging.info(f"{self.num_z} Z levels, {self.num_t} Time points")
        logging.info(f"{self.num_c} Channels: {self.monochrome_channels}")
        logging.info(f"{len(self.regions)} Regions: {self.regions}")
        logging.info(f"Number of FOVs per region: {self.num_fovs_per_region}")

    def parse_multipage_tiff(self) -> None:
        """Parse multi-page TIFF files - SAME LOGIC AS parse_acquisition_metadata."""
        input_path = pathlib.Path(self.parent.input_folder)
        self.acquisition_metadata = {}
        
        max_z = 0
        max_fov = 0
        
        # Iterate over each timepoint - SAME AS ORIGINAL
        for timepoint in self.timepoints:
            image_folder = input_path / str(timepoint)
            coordinates_path = image_folder / "coordinates.csv"
            
            logging.info(f"Processing timepoint {timepoint}, image folder: {image_folder}")
            
            try:
                coordinates_df = pd.read_csv(coordinates_path)
            except FileNotFoundError:
                logging.warning(f"coordinates.csv not found for timepoint {timepoint}")
                continue
            
            # Find all multi-page TIFF files (replaces individual image files)
            tiff_files = sorted([
                f.resolve() for f in image_folder.iterdir()
                if f.suffix.lower() in (".tiff", ".tif") and "_stack" in f.name
            ])
            
            # Process each multi-page TIFF file
            for tiff_file in tiff_files:
                # Parse filename to get region and fov - SAME PARSING LOGIC
                filename_parts = tiff_file.stem.split("_")
                if len(filename_parts) >= 3 and filename_parts[-1] == "stack":
                    region = "_".join(filename_parts[:-2])  # Handle multi-word regions
                    fov = int(filename_parts[-2])
                    
                    try:
                        with tifffile.TiffFile(tiff_file) as tif:
                            # Get all coordinate entries for this region/fov combination
                            fov_coords = coordinates_df[
                                (coordinates_df['region'] == region) & 
                                (coordinates_df['fov'] == fov)
                            ].sort_values('z_level')
                            
                            if fov_coords.empty:
                                logging.warning(f"No coordinates for {tiff_file}")
                                continue
                            
                            # Determine channels and z-levels from TIFF structure and coordinates
                            unique_z_levels = sorted(fov_coords['z_level'].unique())
                            num_z_levels = len(unique_z_levels)
                            num_pages = len(tif.pages)
                            
                            # Infer number of channels
                            if num_z_levels > 0:
                                num_channels = num_pages // num_z_levels
                            else:
                                num_channels = 1
                            
                            # Process each page in the TIFF - REPLACES FILE ITERATION
                            for page_idx, page in enumerate(tif.pages):
                                # Determine z_level and channel from page structure
                                if num_channels == 1:
                                    # Single channel case
                                    if page_idx < len(unique_z_levels):
                                        z_level = unique_z_levels[page_idx]
                                    else:
                                        z_level = page_idx
                                    channel = "BF"  # Default channel name
                                else:
                                    # Multiple channels case - cycle through z-levels for each channel
                                    z_level = unique_z_levels[page_idx % num_z_levels]
                                    channel_idx = page_idx // num_z_levels
                                    
                                    # Try to extract channel name from TIFF metadata
                                    try:
                                        if hasattr(page, 'tags') and 'ImageDescription' in page.tags:
                                            description = page.tags['ImageDescription'].value
                                            if isinstance(description, str) and 'channel' in description.lower():
                                                import re
                                                channel_match = re.search(r'channel["\']?\s*:\s*["\']?([^"\',:}]+)', description, re.IGNORECASE)
                                                if channel_match:
                                                    channel = channel_match.group(1).strip()
                                                else:
                                                    channel = f"channel_{channel_idx}"
                                            else:
                                                channel = f"channel_{channel_idx}"
                                        else:
                                            channel = f"channel_{channel_idx}"
                                    except:
                                        channel = f"channel_{channel_idx}"
                                
                                # Apply same channel name processing as original
                                channel = channel.replace("_", " ").replace("full ", "full_")
                                
                                # Find coordinates - SAME LOGIC AS ORIGINAL
                                coord_rows = coordinates_df[
                                    (coordinates_df["region"] == region)
                                    & (coordinates_df["fov"] == fov)
                                    & (coordinates_df["z_level"] == z_level)
                                ]
                                
                                if coord_rows.empty:
                                    logging.warning(f"No coordinates for {tiff_file}, page {page_idx}")
                                    continue
                                
                                coord_row = coord_rows.iloc[0]
                                
                                # Create metadata object - SAME AS ORIGINAL
                                meta = AcquisitionMetadata(
                                    filepath=tiff_file,  # Points to the multi-page TIFF
                                    x=coord_row["x (mm)"],
                                    y=coord_row["y (mm)"],
                                    z=coord_row["z (um)"],
                                    channel=channel,
                                    z_level=z_level,
                                    region=region,
                                    fov_idx=fov,
                                    t=timepoint,
                                    frame_idx=page_idx
                                )
                                
                                self.acquisition_metadata[meta.key] = meta
                                
                                # Add region and channel names to the sets - SAME AS ORIGINAL
                                self.regions.append(region)
                                self.channel_names.append(channel)
                                
                                # Update max_z and max_fov values - SAME AS ORIGINAL
                                max_z = max(max_z, z_level)
                                max_fov = max(max_fov, fov)
                    
                    except Exception as e:
                        logging.warning(f"Error reading {tiff_file}: {e}")
                        continue
        
        # After processing all timepoints, finalize the list of regions and channels - SAME AS ORIGINAL
        self.regions = sorted(set(self.regions))
        self.channel_names = sorted(set(self.channel_names))
        
        # Calculate number of timepoints (t), Z levels, and FOVs per region - SAME AS ORIGINAL
        self.num_t = len(self.timepoints)
        self.num_z = max_z + 1
        self.num_fovs_per_region = max_fov + 1
        
        if not self.acquisition_metadata:
            logging.warning("No acquisition metadata found")
            return
        
        # Set up image parameters based on the first image - SAME AS ORIGINAL
        first_meta = list(self.acquisition_metadata.values())[0]
        
        # Read the first page of the first TIFF to get image dimensions
        try:
            with tifffile.TiffFile(first_meta.filepath) as tif:
                first_image = tif.pages[first_meta.frame_idx].asarray()
        except:
            logging.warning("Could not read first image for dimensions")
            return
        
        self.dtype = first_image.dtype
        if len(first_image.shape) == 2:
            self.input_height, self.input_width = first_image.shape
        elif len(first_image.shape) == 3:
            self.input_height, self.input_width = first_image.shape[:2]
        else:
            raise ValueError(f"Unexpected image shape: {first_image.shape}")
        
        # Set up chunks - SAME AS ORIGINAL
        self.chunks = (
            1,
            1,
            1,
            min(self.input_height, self.CHUNK_SIZE_LIMIT_PX),
            min(self.input_width, self.CHUNK_SIZE_LIMIT_PX),
        )
        
        # Set up final monochrome channels - SAME AS ORIGINAL
        self.monochrome_channels = []
        for channel in self.channel_names:
            (t, region, fov, z_level, _) = first_meta.key
            channel_key = MetaKey(t, region, fov, z_level, channel)  # Use MetaKey instead of tuple
            if channel_key in self.acquisition_metadata:
                # Read the channel image to check if it's RGB
                try:
                    with tifffile.TiffFile(self.acquisition_metadata[channel_key].filepath) as tif:
                        channel_image = tif.pages[self.acquisition_metadata[channel_key].frame_idx].asarray()
                except:
                    channel_image = first_image  # Fallback
                
                if len(channel_image.shape) == 3 and channel_image.shape[2] == 3:
                    channel = channel.split("_")[0]
                    self.monochrome_channels.extend(
                        [f"{channel}_R", f"{channel}_G", f"{channel}_B"]
                    )
                else:
                    self.monochrome_channels.append(channel)
            else:
                self.monochrome_channels.append(channel)
        
        self.num_c = len(self.monochrome_channels)
        self.monochrome_colors = [
            self.get_channel_color(name) for name in self.monochrome_channels
        ]
        
        # Print out information about the dataset - SAME AS ORIGINAL
        logging.info(f"Regions: {self.regions}, Channels: {self.channel_names}")
        logging.info(f"FOV dimensions: {self.input_height}x{self.input_width}")
        logging.info(f"{self.num_z} Z levels, {self.num_t} Time points")
        logging.info(f"{self.num_c} Channels: {self.monochrome_channels}")
        logging.info(f"{len(self.regions)} Regions: {self.regions}")
        logging.info(f"Number of FOVs per region: {self.num_fovs_per_region}")

    @staticmethod
    def get_channel_color(channel_name: str) -> int:
        """Compute the color for display of a given channel name."""
        color_map = {
            "405": 0x0000FF,  # Blue
            "488": 0x00FF00,  # Green
            "561": 0xFFCF00,  # Yellow
            "638": 0xFF0000,  # Red
            "730": 0x770000,  # Dark Red"
            "_B": 0x0000FF,  # Blue
            "_G": 0x00FF00,  # Green
            "_R": 0xFF0000,  # Red
        }
        for key, value in color_map.items():
            if key in channel_name:
                return value
        return 0xFFFFFF  # Default to white if no match found

    def get_region_metadata(
        self, t: int, region: str
    ) -> dict[MetaKey, AcquisitionMetadata]:
        """Helper method to get region metadata with consistent filtering."""

        # Filter data with explicit type matching
        metadata = {
            key: value
            for key, value in self.acquisition_metadata.items()
            if key.t == t and key.region == region
        }

        if not metadata:
            available_t = sorted(set(k[0] for k in self.acquisition_metadata.keys()))
            available_r = sorted(set(k[1] for k in self.acquisition_metadata.keys()))
            logging.info(f"\nAvailable timepoints in data: {available_t}")
            logging.info(f"Available regions in data: {available_r}")
            raise ValueError(f"No data found for timepoint {t}, region {region}")

        return metadata

    def calculate_output_dimensions(
        self, timepoint: int, region: str
    ) -> ImagePlaneDims:
        """Calculate dimensions for the output image.

        Args:
            timepoint: The timepoint to process
            region: The region identifier

        TODO(colin): it's weird that this depends on a timepoint and region yet
        sets global properties of `self`. Refactor to avoid this?

        Returns:
            tuple: (width_pixels, height_pixels)
        """
        region_data = self.get_region_metadata(timepoint, region)
        # Extract positions
        self.xy_positions = sorted(
            (tile_info.x, tile_info.y) for tile_info in region_data.values()
        )
        x_positions = self.x_positions
        y_positions = self.y_positions

        # Calculate dimensions based on physical coordinates
        width_mm = (
            max(x_positions)
            - min(x_positions)
            + (self.input_width * self.pixel_size_um / 1000)
        )
        height_mm = (
            max(y_positions)
            - min(y_positions)
            + (self.input_height * self.pixel_size_um / 1000)
        )

        width_pixels = int(np.ceil(width_mm * 1000 / self.pixel_size_um))
        height_pixels = int(np.ceil(height_mm * 1000 / self.pixel_size_um))

        # Calculate pyramid levels based on dimensions and number of regions.
        if (
            self.parent.num_pyramid_levels is not None
            and self.parent.num_pyramid_levels > 0
        ):
            self.num_pyramid_levels = self.parent.num_pyramid_levels
        else:
            if len(self.regions) > 1:
                rows, columns = self.get_rows_and_columns()
                max_dimension = max(len(rows), len(columns))
            else:
                max_dimension = 1

            self.num_pyramid_levels = max(
                1,
                math.ceil(
                    np.log2(max(width_pixels, height_pixels) / 1024 * max_dimension)
                ),
            )

        return ImagePlaneDims(width_px=width_pixels, height_px=height_pixels)

    def get_rows_and_columns(self) -> tuple[list[str], list[str]]:
        rows = sorted(set(region[0] for region in self.regions))
        columns = sorted(set(region[1:] for region in self.regions))
        return rows, columns

    @property
    def x_positions(self) -> list[float]:
        """Get the unique x positions in the acquired images.

        The existing code only supports grid-aligned scan patterns, but in the
        future this will have to be replaced in favor of an altenate method for
        scans that are not perfectly aligned to a grid.
        """
        return sorted(set(x for x, _ in self.xy_positions))

    @property
    def y_positions(self) -> list[float]:
        """Get the unique y positions in the acquired images.

        The existing code only supports grid-aligned scan patterns, but in the
        future this will have to be replaced in favor of an altenate method for
        scans that are not perfectly aligned to a grid.
        """
        return sorted(set(y for _, y in self.xy_positions))


class ZLayerSelector(BaseModel):
    """Base class for z-layer selection strategies."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def select_layers(self, num_layers: int) -> list[int]:
        """Select which z-layers to use from the stack."""
        raise NotImplementedError("Subclasses must implement select_layers")
