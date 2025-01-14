import enum
import os
from datetime import datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel


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


class StitchingParameters(BaseModel):
    """Parameters for microscopy image stitching operations."""

    input_folder: Annotated[str, AfterValidator(input_path_exists)]
    """A folder on the local machine containing an image acqusition.

    This should be in the latest format output by the Cephla microscopes. 
    """

    # Output configuration
    output_format: OutputFormat

    # Image processing options
    apply_flatfield: bool = False
    """Whether to apply a flatfield correction to the images prior to stitching."""

    # Registration options
    use_registration: bool = False
    registration_channel: str | None = None  # Will use first available channel if None
    registration_z_level: Annotated[int, AfterValidator(z_non_negative)] = 0
    dynamic_registration: bool = False

    # Scanning and stitching configuration
    scan_pattern: ScanPattern = ScanPattern.unidirectional
    merge_timepoints: bool = False
    merge_hcs_regions: bool = False

    def __post_init__(self) -> None:
        """Validate and process parameters after initialization."""
        # Convert relative path to absolute
        self.input_folder = os.path.abspath(self.input_folder)

    @property
    def stitched_folder(self) -> str:
        """Path to folder containing stitched outputs."""
        return os.path.join(
            self.input_folder
            + "_stitched_"
            + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
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
