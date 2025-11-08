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
from dataclasses import dataclass
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
MIN_VALID_TRANSLATIONS = 1  # Minimum number of valid translations needed for reliable statistics
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
    # Check for both naming conventions to maintain compatibility
    required_cols_v1 = [
        f"{direction}_x_first",
        f"{direction}_y_first",
        f"{direction}_ncc_first"
    ]
    
    required_cols_v2 = [
        f"{direction}_x",
        f"{direction}_y", 
        f"{direction}_ncc"
    ]
    
    # Try the new naming convention first (with _first suffix)
    if all(col in grid.columns for col in required_cols_v1):
        required_cols = required_cols_v1
    # Fall back to old naming convention (without _first suffix)
    elif all(col in grid.columns for col in required_cols_v2):
        required_cols = required_cols_v2
    else:
        # Show what columns are actually available for debugging
        available_cols = [col for col in grid.columns if direction in col]
        raise ValueError(f"Missing required columns for direction '{direction}'. "
                       f"Expected either {required_cols_v1} or {required_cols_v2}. "
                       f"Available columns with '{direction}': {available_cols}")
        
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
    
    # Get normalized translations (divide by image dimensions to get fraction)
    translation: NDArray = np.array([
        grid[f"{direction}_y_first"].values / sizeY,
        grid[f"{direction}_x_first"].values / sizeX,
    ])
    
    # Remove NaN values and track which rows are valid
    valid_mask = np.all(np.isfinite(translation), axis=0)
    translation = translation[:, valid_mask]
    
    # Get filenames for valid translations
    if 'filename' in grid.columns:
        valid_filenames = grid['filename'].values[valid_mask]
    else:
        valid_filenames = [f"Index_{grid.index[i]}" for i in np.where(valid_mask)[0]]
    
    if translation.shape[1] < MIN_VALID_TRANSLATIONS:
        raise ValueError(
            f"Insufficient valid translations ({translation.shape[1]}). "
            f"Need at least {MIN_VALID_TRANSLATIONS}."
        )
    
    # Print all individual overlap values for manual analysis
    print(f"\n{'='*80}")
    print(f"Direction: {direction}")
    print(f"Number of valid translations: {translation.shape[1]}")
    print(f"{'='*80}")
    print(f"Individual overlap values (y, x) as fractions:")
    for i in range(translation.shape[1]):
        y_overlap = translation[0, i]
        x_overlap = translation[1, i]
        filename = valid_filenames[i] if i < len(valid_filenames) else "UNKNOWN"
        print(f"  Pair {i+1:3d} [{filename}]: y={y_overlap:7.4f} ({y_overlap*100:6.2f}%), x={x_overlap:7.4f} ({x_overlap*100:6.2f}%)")
    
    # Compute median overlap
    overlap = np.median(translation, axis=1)
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Y-overlap: median={overlap[0]:7.4f} ({overlap[0]*100:6.2f}%), "
          f"mean={np.mean(translation[0]):7.4f} ({np.mean(translation[0])*100:6.2f}%), "
          f"std={np.std(translation[0]):7.4f} ({np.std(translation[0])*100:6.2f}%)")
    print(f"  X-overlap: median={overlap[1]:7.4f} ({overlap[1]*100:6.2f}%), "
          f"mean={np.mean(translation[1]):7.4f} ({np.mean(translation[1])*100:6.2f}%), "
          f"std={np.std(translation[1]):7.4f} ({np.std(translation[1])*100:6.2f}%)")
    print(f"{'='*80}\n")
    
    # Log overlap values for debugging
    logger.debug(f"Computed {direction} overlap: y={overlap[0]:.3f}, x={overlap[1]:.3f}")
    
    # Only validate that overlaps are not more than 100%
    if np.any(np.abs(overlap) > 1.0):
        raise ValueError("Invalid overlap values > 100% detected")
        
    return tuple(overlap)

@dataclass
class FilterConfig:
    """Configuration for translation filtering pipeline."""
    overlap_left: Float
    overlap_top: Float
    size_x: Int
    size_y: Int
    pou: Float = 3.0
    ncc_threshold: Float = 0.1
    iqr_multiplier: Float = 1.5
    repeatability: Float = 0.0


def filter_translations(
    grid: pd.DataFrame,
    config: FilterConfig
) -> pd.DataFrame:
    """Unified translation filtering pipeline.
    
    Applies all filtering steps in sequence:
    1. Filter by overlap and correlation (valid1)
    2. Filter outliers using IQR (valid2)
    3. Filter by repeatability (valid3)
    4. Replace invalid translations with estimates
    
    Args:
        grid: DataFrame with translation data
        config: Filter configuration parameters
        
    Returns:
        Filtered DataFrame with valid3 columns and second columns
    """
    # Check which directions have data
    has_top = "top_y_first" in grid.columns or "top_y" in grid.columns
    has_left = "left_x_first" in grid.columns or "left_x" in grid.columns
    
    # Apply overlap and correlation filter (valid1)
    if has_top:
        top_y_col = "top_y_first" if "top_y_first" in grid.columns else "top_y"
        top_ncc_col = "top_ncc_first" if "top_ncc_first" in grid.columns else "top_ncc"
        grid["top_valid1"] = filter_by_overlap_and_correlation(
            grid[top_y_col], grid[top_ncc_col],
            config.overlap_top, config.size_y, config.pou, config.ncc_threshold
        )
    else:
        grid["top_valid1"] = pd.Series(False, index=grid.index)
    
    if has_left:
        left_x_col = "left_x_first" if "left_x_first" in grid.columns else "left_x"
        left_ncc_col = "left_ncc_first" if "left_ncc_first" in grid.columns else "left_ncc"
        grid["left_valid1"] = filter_by_overlap_and_correlation(
            grid[left_x_col], grid[left_ncc_col],
            config.overlap_left, config.size_x, config.pou, config.ncc_threshold
        )
    else:
        grid["left_valid1"] = pd.Series(False, index=grid.index)
    
    # Apply IQR outlier filter (valid2)
    if has_top:
        top_y_col = "top_y_first" if "top_y_first" in grid.columns else "top_y"
        grid["top_valid2"] = filter_outliers(
            grid[top_y_col], grid["top_valid1"], config.iqr_multiplier
        )
    else:
        grid["top_valid2"] = pd.Series(False, index=grid.index)
    
    if has_left:
        left_x_col = "left_x_first" if "left_x_first" in grid.columns else "left_x"
        grid["left_valid2"] = filter_outliers(
            grid[left_x_col], grid["left_valid1"], config.iqr_multiplier
        )
    else:
        grid["left_valid2"] = pd.Series(False, index=grid.index)
    
    # Apply repeatability filter (valid3)
    grid = filter_by_repeatability(grid, config.repeatability, config.ncc_threshold)
    
    # Replace invalid translations
    grid = replace_invalid_translations(grid)
    
    return grid


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
            # Handle both naming conventions
            y_col = "left_y_first" if "left_y_first" in grid.columns else "left_y"
            x_col = "left_x_first" if "left_x_first" in grid.columns else "left_x"
            ncc_col = "left_ncc_first" if "left_ncc_first" in grid.columns else "left_ncc"
            
            medy = grp[isvalid][y_col].median()
            medx = grp[isvalid][x_col].median()
            grid.loc[grp.index, "left_valid3"] = (
                grp[y_col].between(medy - r, medy + r)
                & grp[x_col].between(medx - r, medx + r)
                & (grp[ncc_col] > ncc_threshold)
            )
    for _, grp in grid.groupby("row"):
        isvalid = grp["top_valid2"]
        if not any(isvalid):
            grid.loc[grp.index, "top_valid3"] = False
        else:
            # Handle both naming conventions
            y_col = "top_y_first" if "top_y_first" in grid.columns else "top_y"
            x_col = "top_x_first" if "top_x_first" in grid.columns else "top_x"
            ncc_col = "top_ncc_first" if "top_ncc_first" in grid.columns else "top_ncc"
            
            medy = grp[isvalid][y_col].median()
            medx = grp[isvalid][x_col].median()
            grid.loc[grp.index, "top_valid3"] = (
                grp[y_col].between(medy - r, medy + r)
                & grp[x_col].between(medx - r, medx + r)
                & (grp[ncc_col] > ncc_threshold)
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
            # Handle both naming conventions
            first_col = f"{direction}_{key}_first"
            second_col = f"{direction}_{key}_second"
            
            if first_col in grid.columns:
                grid.loc[isvalid, second_col] = grid.loc[isvalid, first_col]
            else:
                # Fallback to non-suffixed columns if _first columns don't exist
                fallback_col = f"{direction}_{key}"
                if fallback_col in grid.columns:
                    grid.loc[isvalid, second_col] = grid.loc[isvalid, fallback_col]
                else:
                    # Create empty column if neither exists
                    grid[second_col] = np.nan
                    
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
                # Handle both naming conventions
                y_first_col = f"{direction}_y_first"
                x_first_col = f"{direction}_x_first"
                
                if y_first_col in grid.columns:
                    grid.loc[grp.index[~isvalid], f"{direction}_y_second"] = grp[isvalid][y_first_col].median()
                else:
                    fallback_y_col = f"{direction}_y"
                    if fallback_y_col in grid.columns:
                        grid.loc[grp.index[~isvalid], f"{direction}_y_second"] = grp[isvalid][fallback_y_col].median()
                    else:
                        grid.loc[grp.index[~isvalid], f"{direction}_y_second"] = 0.0
                        
                if x_first_col in grid.columns:
                    grid.loc[grp.index[~isvalid], f"{direction}_x_second"] = grp[isvalid][x_first_col].median()
                else:
                    fallback_x_col = f"{direction}_x"
                    if fallback_x_col in grid.columns:
                        grid.loc[grp.index[~isvalid], f"{direction}_x_second"] = grp[isvalid][fallback_x_col].median()
                    else:
                        grid.loc[grp.index[~isvalid], f"{direction}_x_second"] = 0.0
                        
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
