"""Stage model module for microscope tile registration.

This module provides functionality for modeling and validating stage positions and
translations between microscope image tiles. It includes:

- Translation validation and filtering
- Overlap computation
- Outlier detection
- Translation refinement

The module uses statistical methods to ensure robust registration results.
"""
import itertools
from typing import Callable, Dict, List, Optional, Tuple
from enum import Enum, auto
import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._typing_utils import Float, FloatArray, Int

# Configure logger
logger = logging.getLogger(__name__)

# Constants for validation
MIN_VALID_TRANSLATIONS = 3  # Minimum number of valid translations needed for reliable statistics
MAX_OUTLIER_FACTOR = 3.0  # Maximum factor for outlier detection
MIN_OVERLAP_PERCENT = 5.0  # Minimum required overlap percentage

class TranslationSource(Enum):
    """Source of translation values in the registration process.
    
    Attributes:
        MEASURED: Translation directly measured from image registration
        ESTIMATED: Translation estimated from other valid translations
        INVALID: Translation is invalid and cannot be used
    """
    MEASURED = auto()
    ESTIMATED = auto()
    INVALID = auto()

def validate_grid_dataframe(grid: pd.DataFrame, direction: str) -> None:
    """Validate grid DataFrame structure and content.
    
    Args:
        grid: DataFrame to validate
        direction: Direction to validate ('left' or 'top')
        
    Raises:
        ValueError: If DataFrame is invalid or missing required columns
    """
    required_cols = [
        f"{direction}_x_first",
        f"{direction}_y_first",
        f"{direction}_ncc_first"
    ]
    
    missing_cols = [col for col in required_cols if col not in grid.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
        
    if grid.empty:
        raise ValueError("Empty grid DataFrame")
        
    if not grid[required_cols].dtypes.apply(np.issubdtype, args=(np.number,)).all():
        raise ValueError("Non-numeric values in translation columns")

def compute_image_overlap2(
    grid: pd.DataFrame,
    direction: str,
    sizeY: Int,
    sizeX: Int
) -> Tuple[Float, Float]:
    """Compute median image overlap in both dimensions.

    Args:
        grid: DataFrame with tile positions and translations
        direction: Direction to compute overlap for ('left' or 'top')
        sizeY: Image height in pixels
        sizeX: Image width in pixels

    Returns:
        Tuple of (y_overlap, x_overlap) as fractions of image size

    Raises:
        ValueError: If input data is invalid or insufficient
    """
    validate_grid_dataframe(grid, direction)
    
    # Get normalized translations
    translation: NDArray = np.array([
        grid[f"{direction}_y_first"].values / sizeY,
        grid[f"{direction}_x_first"].values / sizeX,
    ])
    
    # Remove NaN values
    valid_mask = np.all(np.isfinite(translation), axis=0)
    translation = translation[:, valid_mask]
    
    if translation.shape[1] < MIN_VALID_TRANSLATIONS:
        raise ValueError(
            f"Insufficient valid translations ({translation.shape[1]}). "
            f"Need at least {MIN_VALID_TRANSLATIONS}."
        )
    
    # Compute median overlap
    overlap = np.median(translation, axis=1)
    
    # Log overlap values for debugging
    logger.debug(f"Computed {direction} overlap: y={overlap[0]:.3f}, x={overlap[1]:.3f}")
    
    # Only validate that overlaps are not more than 100%
    if np.any(np.abs(overlap) > 1.0):
        raise ValueError("Invalid overlap values > 100% detected")
        
    return tuple(overlap)

def filter_by_overlap_and_correlation(
    T: pd.Series,
    ncc: pd.Series,
    overlap: Float,
    size: Int,
    pou: Float = 3,
    ncc_threshold: Float = 0.1,
) -> pd.Series:
    """Filter translations by estimated overlap and correlation values.

    Args:
        T: Translation values in pixels
        ncc: Normalized cross correlation values
        overlap: Expected overlap percentage
        size: Image dimension size in pixels
        pou: Percent overlap uncertainty (default: 3)
        ncc_threshold: Minimum acceptable NCC value (default: 0.1)

    Returns:
        Boolean series indicating valid translations

    Raises:
        ValueError: If input parameters are invalid
    """
    if not (0 < overlap < 100):
        raise ValueError("Overlap must be between 0 and 100 percent")
    if pou < 0 or pou >= overlap:
        raise ValueError("Invalid percent overlap uncertainty (pou)")
    if ncc_threshold < 0 or ncc_threshold > 1:
        raise ValueError("NCC threshold must be between 0 and 1")
        
    # Calculate overlap range
    min_overlap = size * (100 - overlap - pou) / 100
    max_overlap = size * (100 - overlap + pou) / 100
    
    return (T.between(min_overlap, max_overlap)) & (ncc > ncc_threshold)

def filter_outliers(
    T: pd.Series,
    isvalid: pd.Series,
    w: Float = 1.5
) -> pd.Series:
    """Filter translation outliers using IQR method.

    Args:
        T: Translation values
        isvalid: Initial validity mask
        w: IQR multiplier for outlier detection (default: 1.5)

    Returns:
        Boolean series with outliers marked as False

    Notes:
        Uses the standard interquartile range (IQR) method where values
        outside Q1 - w*IQR to Q3 + w*IQR are considered outliers.
    """
    if w <= 0:
        raise ValueError("IQR multiplier must be positive")
        
    valid_T = T[isvalid].values
    if len(valid_T) < MIN_VALID_TRANSLATIONS:
        return isvalid
        
    q1, _, q3 = np.quantile(valid_T, (0.25, 0.5, 0.75))
    iqr = max(1, abs(q3 - q1))  # Use at least 1 pixel range

    # Detect outliers
    lower_bound = q1 - w * iqr
    upper_bound = q3 + w * iqr
    
    return isvalid & T.between(lower_bound, upper_bound)

def filter_by_repeatability(
    grid: pd.DataFrame, r: Float, ncc_threshold: Float
) -> pd.DataFrame:
    """Filter the stage translation by repeatability.

    Parameters
    ----------
    grid : pd.DataFrame
        the dataframe for the grid position, with columns "{left|top}_{x|y|ncc}_first"
    r : Float
        the repeatability value
    ncc_threshold : Float
        the threshold for ncc values, only values higher will be considered

    Returns
    -------
    grid : pd.DataFrame
        the updated dataframe for the grid position
    """
    for _, grp in grid.groupby("col"):
        isvalid = grp["left_valid2"].astype(bool)
        if not any(isvalid):
            grid.loc[grp.index, "left_valid3"] = False
        else:
            medx = grp[isvalid]["left_y_first"].median()
            medy = grp[isvalid]["left_x_first"].median()
            grid.loc[grp.index, "left_valid3"] = (
                grp["left_y_first"].between(medx - r, medx + r)
                & grp["left_x_first"].between(medy - r, medy + r)
                & (grp["left_ncc_first"] > ncc_threshold)
            )
    for _, grp in grid.groupby("row"):
        isvalid = grp["top_valid2"]
        if not any(isvalid):
            grid.loc[grp.index, "top_valid3"] = False
        else:
            medx = grp[isvalid]["top_y_first"].median()
            medy = grp[isvalid]["top_x_first"].median()
            grid.loc[grp.index, "top_valid3"] = (
                grp["top_y_first"].between(medx - r, medx + r)
                & grp["top_x_first"].between(medy - r, medy + r)
                & (grp["top_ncc_first"] > ncc_threshold)
            )
    return grid

def replace_invalid_translations(grid: pd.DataFrame) -> pd.DataFrame:
    """Replace invalid translations by estimated values.

    Parameters
    ----------
    grid : pd.DataFrame
        the dataframe for the grid position,
        with columns "{left|top}_{x|y}_second" and "{left|top}_valid3"

    Returns
    -------
    grid : pd.DataFrame
        the updated dataframe for the grid position
    """
    # First, copy valid translations to second columns and mark their source
    for direction in ["left", "top"]:
        for key in ["x", "y", "ncc"]:
            isvalid = grid[f"{direction}_valid3"]
            grid.loc[isvalid, f"{direction}_{key}_second"] = grid.loc[
                isvalid, f"{direction}_{key}_first"
            ]
            # Add source column if it doesn't exist
            if f"{direction}_source" not in grid.columns:
                grid[f"{direction}_source"] = TranslationSource.INVALID
            grid.loc[isvalid, f"{direction}_source"] = TranslationSource.MEASURED

    # Replace invalid translations with median of valid ones
    for direction, rowcol in zip(["left", "top"], ["col", "row"]):
        for _, grp in grid.groupby(rowcol):
            isvalid = grp[f"{direction}_valid3"].astype(bool)
            if any(isvalid):
                # Replace invalid translations with median of valid ones
                grid.loc[grp.index[~isvalid], f"{direction}_y_second"] = grp[isvalid][
                    f"{direction}_y_first"
                ].median()
                grid.loc[grp.index[~isvalid], f"{direction}_x_second"] = grp[isvalid][
                    f"{direction}_x_first"
                ].median()
                grid.loc[grp.index[~isvalid], f"{direction}_ncc_second"] = 0.0  # No correlation for estimated translations
                grid.loc[grp.index[~isvalid], f"{direction}_source"] = TranslationSource.ESTIMATED

    # Handle any remaining NaN values
    for direction, xy in itertools.product(["left", "top"], ["x", "y"]):
        key = f"{direction}_{xy}_second"
        isna = pd.isna(grid[key])
        if any(isna):
            grid.loc[isna, key] = grid.loc[~isna, key].median()
            grid.loc[isna, f"{direction}_ncc_second"] = 0.0
            grid.loc[isna, f"{direction}_source"] = TranslationSource.ESTIMATED

    # Verify all translations are finite
    for direction, xy in itertools.product(["left", "top"], ["x", "y"]):
        assert np.all(np.isfinite(grid[f"{direction}_{xy}_second"]))

    return grid
