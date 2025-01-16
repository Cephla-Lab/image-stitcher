import enum
import functools
import json
import logging
import math
import os
import pathlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, ClassVar, NamedTuple, assert_never

import numpy as np
import pandas as pd
import psutil
from dask_image.imread import imread as dask_imread
from pydantic import AfterValidator, BaseModel

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S.%f"


class OutputFormat(enum.Enum):
    ome_zarr = ".ome.zarr"
    ome_tiff = ".ome.tiff"


class ScanPattern(enum.Enum):
    unidirectional = "Unidirectional"
    s_pattern = "S-Pattern"


def input_path_exists(path: str) -> str:
    """Pydantic validator to check the path exists."""
    if not os.path.exists(path):
        raise ValueError(f"Input folder does not exist: {path}")

    return path


def z_non_negative(num: int) -> int:
    """Pydantic validator that the z-level is >= 0."""
    if num < 0:
        raise ValueError("Registration Z-level must be non-negative")

    return num


class StitchingParameters(
    BaseModel,
    use_attribute_docstrings=True,
):
    """Parameters for microscopy image stitching operations."""

    input_folder: Annotated[str, AfterValidator(input_path_exists)]
    """A folder on the local machine containing an image acqusition.

    This should be in the latest format output by the Cephla microscopes. 
    """

    # Output configuration
    output_format: OutputFormat = OutputFormat.ome_zarr
    """Output format for the stitched data."""

    # Image processing options
    apply_flatfield: bool = False
    """Whether to apply a flatfield correction to the images prior to stitching."""

    # Registration options
    use_registration: bool = False
    """Whether to register the images using their content, rather than based on x/y stage positions."""
    registration_channel: str | None = None
    """Channel name to use for registration. None -> first available channel."""
    registration_z_level: Annotated[int, AfterValidator(z_non_negative)] = 0
    """Z level to use for registration."""
    dynamic_registration: bool = False
    """Use dynamic registration for improved accuracy.

    TODO(colin): describe what this actually does and when you would want to use it or not.
    """

    # Scanning and stitching configuration
    scan_pattern: ScanPattern = ScanPattern.unidirectional
    """The scan pattern used for the acquisition.

    unidirectional means all rows go the same direction; S-pattern indicates every
    other row goes the other direction.
    """
    merge_timepoints: bool = False
    """Merge timepoints to create time series output."""

    merge_hcs_regions: bool = False
    """Merge HCS regions (wells) to create full wellplate HCS output."""

    verbose: bool = False
    """Show debug-level logging."""

    def __post_init__(self) -> None:
        """Validate and process parameters after initialization."""
        # Convert relative path to absolute
        self.input_folder = os.path.abspath(self.input_folder)

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
        match self:
            case ReverseRows.even:
                return row_idx % 2 == 0
            case ReverseRows.odd:
                return row_idx % 2 == 1
            case _ as unreachable:
                assert_never(unreachable)


@dataclass
class SPatternScanParams:
    """Computed parameters for an S-pattern scan pattern."""

    h_shift: tuple[int, int] = (0, 0)
    v_shift: tuple[int, int] = (0, 0)
    h_shift_rev: tuple[int, int] = (0, 0)
    h_shift_rev_rows: ReverseRows = ReverseRows.even


ScanParams = UnidirectionalScanPatternParams | SPatternScanParams


def default_scan_params(pattern: ScanPattern) -> ScanParams:
    match pattern:
        case ScanPattern.unidirectional:
            return UnidirectionalScanPatternParams()
        case ScanPattern.s_pattern:
            return SPatternScanParams()
        case _ as unreachable:
            assert_never(unreachable)


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
    # TODO(colin): fill in correct type when importing the flatfield code.
    flatfields: dict[None, None] = field(init=False)
    acquisition_metadata: dict[MetaKey, AcquisitionMetadata] = field(init=False)
    dtype: np.dtype = field(init=False)
    chunks: dict[tuple[int, int], tuple[int, int, int, int, int]] = field(init=False)
    """A map from (timepoint, region) to dask/zarr array chunk sizes (t, c, z, y, x)."""
    xy_positions: list[tuple[float, float]] = field(init=False)
    regions: list[str] = field(default_factory=list)
    channel_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.scan_params = default_scan_params(self.parent.scan_pattern)
        self.init_timepoints()
        self.init_acquisition_parameters()
        self.init_pixel_size()
        self.parse_acquisition_metadata()

    def init_timepoints(self) -> None:
        self.timepoints = [
            int(d, 10)
            for d in os.listdir(self.parent.input_folder)
            if os.path.isdir(os.path.join(self.parent.input_folder, d)) and d.isdigit()
        ]
        self.timepoints.sort()

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

    def compute_chunks(self) -> dict[tuple[int, int], tuple[int, int, int, int, int]]:
        """Compute the optimial chunk size for dask storage.

        Empirically it seems that larger chunks are faster (so long as we can
        keep them in memory) up to 2**31-1 bytes at which point we start to run
        into problems (hard errors from some zarr operations; occasional hanging
        from dask).

        Returns a map from (timepoint, region) to chunks sizes (t, c, z, y, x)
        """
        # psutil docs say the available item of this function's output tries to
        # be a cross-platform measure of how much memory could be used before
        # swap is required
        estimated_available_memory = psutil.virtual_memory().available
        chunks = {}

        for timepoint in self.timepoints:
            for region in self.regions:
                width, height = self.calculate_output_dimensions(timepoint, region)

                full_in_memory_output_shape = (
                    1,
                    self.num_c,
                    self.num_z,
                    height,
                    width,
                )
                output_estimated_memory_required = (
                    functools.reduce(lambda a, b: a * b, full_in_memory_output_shape)
                    * self.dtype.itemsize
                )
                if output_estimated_memory_required < 0.8 * estimated_available_memory:
                    if output_estimated_memory_required > 2**31 - 1:
                        chunks[(timepoint, region)] = (
                            1,
                            1,
                            1,
                            StitchingComputedParameters.CHUNK_SIZE_LIMIT_PX,
                            StitchingComputedParameters.CHUNK_SIZE_LIMIT_PX,
                        )
                    else:
                        chunks[(timepoint, region)] = full_in_memory_output_shape
                else:
                    # TODO(colin): this might be too small in many cases; we should try
                    # setting this chunk size based on available memory too?
                    # Need a benchmark for this case to test that out.
                    chunks[(timepoint, region)] = (
                        1,
                        1,
                        1,
                        self.input_height,
                        self.input_width,
                    )

        return chunks

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
        self.chunks = self.compute_chunks()

        # Print out information about the dataset
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

        if self.parent.use_registration:
            # Calculate dimensions with registration shifts
            num_cols = len(x_positions)
            num_rows = len(y_positions)

            # Handle different scanning patterns
            if isinstance(self.scan_params, SPatternScanParams):
                max_h_shift = (
                    max(
                        abs(self.scan_params.h_shift[0]),
                        abs(self.scan_params.h_shift_rev[0]),
                    ),
                    max(
                        abs(self.scan_params.h_shift[1]),
                        abs(self.scan_params.h_shift_rev[1]),
                    ),
                )
            else:
                max_h_shift = (
                    abs(self.scan_params.h_shift[0]),
                    abs(self.scan_params.h_shift[1]),
                )

            # Calculate dimensions including overlaps and shifts
            width_pixels = int(
                self.input_width
                + ((num_cols - 1) * (self.input_width - max_h_shift[1]))
            )
            width_pixels += abs(
                (num_rows - 1) * self.scan_params.v_shift[1]
            )  # Add horizontal shift from vertical registration

            height_pixels = int(
                self.input_height
                + ((num_rows - 1) * (self.input_height - self.scan_params.v_shift[0]))
            )
            height_pixels += abs(
                (num_cols - 1) * max_h_shift[0]
            )  # Add vertical shift from horizontal registration

        else:
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

        # Calculate pyramid levels based on dimensions and number of regions
        if len(self.regions) > 1:
            rows, columns = self.get_rows_and_columns()
            max_dimension = max(len(rows), len(columns))
        else:
            max_dimension = 1

        self.num_pyramid_levels = max(
            1,
            math.ceil(np.log2(max(width_pixels, height_pixels) / 1024 * max_dimension)),
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
