"""Tile registration module for microscope image stitching.

This module provides functionality for registering microscope image tiles and updating
their stage coordinates without performing full image stitching. It includes:

- Tile position extraction and validation
- Row and column clustering of tiles
- Translation computation between adjacent tiles
- Global optimization of tile positions
- Stage coordinate updates

The module is designed to work with TIFF images and CSV coordinate files from
microscope acquisition systems.
"""
import os
from dataclasses import dataclass
from pathlib import Path
import re
import logging
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import json

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

from typing import Dict, List, Optional, Tuple, Union, Pattern
from ._typing_utils import BoolArray
from ._typing_utils import Float
from ._typing_utils import NumArray

from ._constrained_refinement import refine_translations
from ._global_optimization import compute_final_position
from ._global_optimization import compute_maximum_spanning_tree
from ._stage_model import compute_image_overlap2
from ._stage_model import filter_by_overlap_and_correlation
from ._stage_model import filter_by_repeatability
from ._stage_model import filter_outliers
from ._stage_model import replace_invalid_translations
from ._translation_computation import interpret_translation
from ._translation_computation import multi_peak_max
from ._translation_computation import pcm

# Set start method to 'spawn' for CUDA safety in multiprocessing
# 'fork' (default on linux) can cause issues with CUDA context
# initialization in child processes.
# This must be done once, before any pools are created.
try:
    mp.set_start_method('spawn', force=True)
    # print("INFO: Multiprocessing start method set to 'spawn'.")
except RuntimeError:
    pass # context already set

# Configure logger
logger = logging.getLogger(__name__)

# Constants for file handling
DEFAULT_FOV_RE = re.compile(r"(?P<region>\w+)_(?P<fov>0|[1-9]\d*)_(?P<z_level>\d+)_", re.I)
DEFAULT_FOV_COL = "fov"
DEFAULT_X_COL = "x (mm)"
DEFAULT_Y_COL = "y (mm)"

# Constants for tile registration
MIN_PITCH_FOR_FACTOR = 0.1  # Minimum pitch value to use factor-based tolerance (mm)
DEFAULT_ABSOLUTE_TOLERANCE = 0.05  # Default absolute tolerance (mm)
ROW_TOL_FACTOR = 0.20  # Tolerance factor for row clustering
COL_TOL_FACTOR = 0.20  # Tolerance factor for column clustering

@dataclass
class RowInfo:
    """Information about a row of tiles.
    
    Attributes:
        center_y: Y-coordinate of the row center in mm
        tile_indices: List of indices for tiles in this row
    """
    center_y: float
    tile_indices: List[int]

def extract_tile_indices(
    filenames: List[str],
    coords_df: pd.DataFrame,
    *,
    fov_re: Pattern[str] = DEFAULT_FOV_RE,
    fov_col_name: str = DEFAULT_FOV_COL,
    x_col_name: str = DEFAULT_X_COL,
    y_col_name: str = DEFAULT_Y_COL,
    ROW_TOL_FACTOR: float = 0.20,
    COL_TOL_FACTOR: float = 0.20
) -> Tuple[List[int], List[int], Dict[str, int]]:
    """
    Map each filename to (row, col) based on stage coordinates.
    Handles rows of different length that are centred or truncated.

    Args:
        filenames: List of filenames for the tiles.
        coords_df: DataFrame with tile coordinates. Must contain columns specified by
                   fov_col_name, x_col_name, y_col_name.
        fov_re: Compiled regular expression to extract FOV identifier from filename.
                The first capturing group should be the FOV identifier.
        fov_col_name: Name of the column in coords_df containing the FOV identifier.
        x_col_name: Name of the column for X coordinates.
        y_col_name: Name of the column for Y coordinates.
        ROW_TOL_FACTOR: Tolerance factor for row clustering (percentage of Y pitch).
        COL_TOL_FACTOR: Tolerance factor for column clustering (percentage of X pitch).

    Returns:
        A tuple: (row_assignments, col_assignments, fname_to_dfidx_map)
        - row_assignments: List of 0-indexed row numbers for each filename.
        - col_assignments: List of 0-indexed column numbers for each filename.
        - fname_to_dfidx_map: Dictionary mapping filename to its original index in coords_df.
    """
    if not filenames:
        logger.info("Received empty filenames list, returning empty results.")
        return [], [], {}

    # --- Validate DataFrame columns ---
    required_cols = [fov_col_name, x_col_name, y_col_name]
    for col in required_cols:
        if col not in coords_df.columns:
            msg = f"Required column '{col}' not found in coords_df."
            logger.error(msg)
            raise KeyError(msg)

    # 1.  Collect (x, y) per filename
    xy_coords: List[Tuple[float, float]] = []
    fname_to_dfidx_map: Dict[str, int] = {}
    valid_indices_in_filenames = []

    for i, fname in enumerate(filenames):
        m = fov_re.search(fname)
        if not m:
            logger.warning(f"{fname}: cannot extract FOV with pattern {fov_re.pattern}. Skipping this file.")
            continue
        try:
            # Use named groups if available
            if 'fov' in m.groupdict():
                fov_str = m.group('fov')
            else:
                fov_str = m.group(1)
            fov = int(fov_str)
        except (IndexError, ValueError):
            logger.warning(f"{fname}: FOV regex matched, but could not extract a valid integer FOV. Skipping.")
            continue

        # For multi-region support, also check region match if present
        region_match = True
        if 'region' in m.groupdict() and 'region' in coords_df.columns:
            file_region = m.group('region')
            # Filter by region as well
            df_row = coords_df.loc[
                (coords_df[fov_col_name].astype(int) == fov) & 
                (coords_df['region'] == file_region)
            ]
        else:
            # Original behavior without region filtering
            try:
                df_row = coords_df.loc[coords_df[fov_col_name].astype(int) == fov]
            except ValueError:
                logger.error(f"Could not convert column '{fov_col_name}' to int for comparison.")
                raise

        if df_row.empty:
            logger.warning(f"FOV {fov} (from {fname}) not in coordinates DataFrame. Skipping this file.")
            continue
        if len(df_row) > 1:
            logger.warning(f"FOV {fov} (from {fname}) has multiple entries. Using the first one that matches the z-level if possible.")
            
            # Try to find an entry that also matches the z-level from the filename
            if 'z_level' in m.groupdict():
                try:
                    z_level_fname = int(m.group('z_level'))
                    if 'z_level' in coords_df.columns:
                        matching_z_rows = df_row[df_row['z_level'].astype(int) == z_level_fname]
                        if not matching_z_rows.empty:
                            df_row = matching_z_rows
                            logger.info(f"Refined selection for FOV {fov} to include z_level {z_level_fname}.")
                except ValueError:
                    logger.warning(f"Could not parse z_level from filename {fname}.")
            
            if len(df_row) > 1:
                logger.warning(f"Still multiple entries for FOV {fov}. Using the first.")

        idx = df_row.index[0]
        fname_to_dfidx_map[fname] = idx
        try:
            x_val = float(df_row.at[idx, x_col_name])
            y_val = float(df_row.at[idx, y_col_name])
            xy_coords.append((x_val, y_val))
            valid_indices_in_filenames.append(i)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse coordinates for FOV {fov} (from {fname}): {e}. Skipping.")
            continue

    if not xy_coords:
        logger.warning("No valid coordinates could be extracted. Returning empty results.")
        num_original_files = len(filenames)
        return [-1] * num_original_files, [-1] * num_original_files, fname_to_dfidx_map

    x_arr, y_arr = map(np.asarray, zip(*xy_coords))

    # Initialize full assignment arrays with -1 (unassigned)
    final_row_assignments = np.full(len(filenames), -1, dtype=int)
    final_col_assignments = np.full(len(filenames), -1, dtype=int)

    # 2.  Row clustering
    sorted_idx_for_xy = np.argsort(y_arr)
    processed_rows: List[RowInfo] = []

    unique_y = np.sort(np.unique(y_arr))
    pitch_y = np.min(np.diff(unique_y)) if len(unique_y) > 1 else 0.0
    if pitch_y == 0.0 and len(unique_y) > 1:
        logger.warning("Calculated Y pitch is zero despite multiple unique Y values.")

    row_tol = pitch_y * ROW_TOL_FACTOR if pitch_y > MIN_PITCH_FOR_FACTOR else DEFAULT_ABSOLUTE_TOLERANCE
    logger.debug(f"Row clustering: pitch_y={pitch_y:.4f}, row_tol={row_tol:.4f}")

    for gi_xy in sorted_idx_for_xy:
        y = y_arr[gi_xy]
        if not processed_rows or abs(y - processed_rows[-1].center_y) > row_tol:
            processed_rows.append(RowInfo(center_y=y, tile_indices=[gi_xy]))
        else:
            current_row = processed_rows[-1]
            current_row.tile_indices.append(gi_xy)
            mean_y = np.mean(y_arr[current_row.tile_indices])
            processed_rows[-1] = RowInfo(center_y=mean_y, tile_indices=current_row.tile_indices)

    # Map row assignments back
    temp_row_assignments = np.full(len(xy_coords), -1, dtype=int)
    for r_idx, row_info in enumerate(processed_rows):
        if row_info.tile_indices:
            temp_row_assignments[np.array(row_info.tile_indices)] = r_idx

    # 3.  Global column clustering
    unique_x = np.sort(np.unique(x_arr))
    pitch_x = np.min(np.diff(unique_x)) if len(unique_x) > 1 else 0.0
    if pitch_x == 0.0 and len(unique_x) > 1:
        logger.warning("Calculated X pitch is zero despite multiple unique X values.")

    col_tol = pitch_x * COL_TOL_FACTOR if pitch_x > MIN_PITCH_FOR_FACTOR else DEFAULT_ABSOLUTE_TOLERANCE
    logger.debug(f"Column clustering: pitch_x={pitch_x:.4f}, col_tol={col_tol:.4f}")

    col_centers_list: List[float] = []
    if unique_x.size > 0:
        col_centers_list.append(unique_x[0])
        for x_val in unique_x[1:]:
            if abs(x_val - col_centers_list[-1]) > col_tol:
                col_centers_list.append(x_val)
    col_centers_arr = np.asarray(col_centers_list)

    temp_col_assignments = np.full(len(xy_coords), -1, dtype=int)
    if x_arr.size > 0:
        if col_centers_arr.size == 0:
            logger.warning("No distinct column centers found. Assigning all to column 0.")
            if x_arr.size > 0:
                temp_col_assignments.fill(0)
        else:
            temp_col_assignments = np.argmin(np.abs(x_arr[:, None] - col_centers_arr[None, :]), axis=1)

    # Populate final assignments
    if valid_indices_in_filenames:
        final_row_assignments[valid_indices_in_filenames] = temp_row_assignments
        final_col_assignments[valid_indices_in_filenames] = temp_col_assignments

    logger.info(f"Processed {len(xy_coords)}/{len(filenames)} files. Found {len(processed_rows)} rows and {len(col_centers_list)} columns.")
    
    return final_row_assignments.tolist(), final_col_assignments.tolist(), fname_to_dfidx_map



DEFAULT_EDGE_WIDTH = 256  # Increased from 64 to 256 for better correlation with 2048x2048 tiles

def register_tiles(
    images: NumArray,
    rows: List[int],
    cols: List[int],
    edge_width: int = DEFAULT_EDGE_WIDTH, # New parameter
    overlap_diff_threshold: Float = 10, #10 by default
    pou: Float = 3, #3 by default
    ncc_threshold: Float = 0.5, #0.5 by default
) -> Tuple[pd.DataFrame, dict]:
    """Register tiles using edge-based correlation without full stitching.
    
    Parameters
    ----------
    images : NumArray
        Array of full images to register
    rows : List[int]
        Row indices for each image
    cols : List[int]
        Column indices for each image
    edge_width : int
        Width of the edge strips (in pixels) to use for registration.
    overlap_diff_threshold : Float
        Allowed difference from initial guess (percentage)
    pou : Float
        Percent overlap uncertainty
    ncc_threshold : Float
        Normalized cross correlation threshold
        
    Returns
    -------
    grid : pd.DataFrame
        Registration results with global pixel positions
    prop_dict : dict
        Dictionary of estimated parameters
    """
    images_arr = np.array(images) # Ensure it's a NumPy array
    if images_arr.ndim < 3: # Expecting (N, H, W) or (N, H, W, C)
        raise ValueError("Images array must be at least 3-dimensional (N, H, W).")
    if images_arr.shape[0] == 0:
        logger.warning("Empty images array provided to register_tiles.")
        # Return empty DataFrame and dict, or handle as appropriate
        return pd.DataFrame(), {}

    assert len(rows) == len(cols) == images_arr.shape[0]
    
    # Full image dimensions
    full_sizeY, full_sizeX = images_arr.shape[1:3] # Use 1:3 to be robust for (N,H,W) or (N,H,W,C)
                                                # Assuming registration on first channel if C exists,
                                                # or that images are grayscale.
                                                # If images are (N,C,H,W), then shape[2:4]

    grid = pd.DataFrame({
        "col": cols,
        "row": rows,
    }, index=np.arange(len(cols)))
    
    coord_to_idx = pd.Series(grid.index, index=pd.MultiIndex.from_arrays([grid['col'], grid['row']]))
    
    top_coords = pd.MultiIndex.from_arrays([grid['col'], grid['row'] - 1])
    grid['top'] = pd.array(np.round(coord_to_idx.reindex(top_coords).values), dtype='Int32')
    
    left_coords = pd.MultiIndex.from_arrays([grid['col'] - 1, grid['row']])
    grid['left'] = pd.array(np.round(coord_to_idx.reindex(left_coords).values), dtype='Int32')
    
    for direction in ["left", "top"]:
        for key in ["ncc", "y", "x"]:
            grid[f"{direction}_{key}_first"] = np.nan
            # Ensure _second columns exist for refine_translations if they are result of replace_invalid_translations
            grid[f"{direction}_{key}_second"] = np.nan 

    # Translation computation using edges
    with Pool(processes=min(2, cpu_count())) as pool:
        # Pass full_sizeY, full_sizeX, and edge_width
        args_for_pool = [(d, grid, images_arr, full_sizeY, full_sizeX, edge_width) for d in ["left", "top"]]
        all_results = list(tqdm(
            pool.imap(_compute_direction_translations, args_for_pool),
            total=len(args_for_pool),
            desc="Computing initial translations on edges"
        ))
        
        for results_list in all_results:
            for i2, direction, max_peak_global_translation in results_list: # max_peak is now global
                for j, key in enumerate(["ncc", "y", "x"]):
                    grid.loc[i2, f"{direction}_{key}_first"] = max_peak_global_translation[j]

    # --- The rest of the function (filtering, global optimization) remains largely the same ---
    # --- as it operates on the 'grid' DataFrame which now contains global translations ---
    # --- derived from edge computations.                                               ---

    # Example: (Ensure these functions are robust to NaNs from missing pairs)
    has_top_pairs = np.any(grid["top_ncc_first"].dropna() > ncc_threshold)
    has_left_pairs = np.any(grid["left_ncc_first"].dropna() > ncc_threshold)

    if not has_left_pairs and not has_top_pairs:
         raise ValueError("No good initial pairs found (left or top) - tiles may not have sufficient overlap or NCC threshold too high.")
    if not has_left_pairs:
        logger.warning("No good left pairs found - tiles may not have sufficient horizontal overlap or NCC threshold too high.")
        # Create dummy left displacement if no left pairs exist but top pairs do
        left_displacement = (0.0, 0.0) # (dy, dx) normalized
        overlap_left = 50.0 # Default
    else:
        left_displacement = compute_image_overlap2(
            grid[grid["left_ncc_first"].fillna(-1) > ncc_threshold], "left", full_sizeY, full_sizeX
        )
        overlap_left = np.clip(100 - left_displacement[1] * 100, pou, 100 - pou)


    if not has_top_pairs:
        logger.warning("No good top pairs found - tiles may not have sufficient vertical overlap or NCC threshold too high.")
        top_displacement = (0.0, 0.0) # (dy, dx) normalized
        overlap_top = 50.0 # Default
    else:
        top_displacement = compute_image_overlap2(
            grid[grid["top_ncc_first"].fillna(-1) > ncc_threshold], "top", full_sizeY, full_sizeX
        )
        overlap_top = np.clip(100 - top_displacement[0] * 100, pou, 100 - pou)
    
    # Filter and validate translations
    if has_top_pairs:
        grid["top_valid1"] = filter_by_overlap_and_correlation(
            grid["top_y_first"], grid["top_ncc_first"], overlap_top, full_sizeY, pou, ncc_threshold
        )
        grid["top_valid2"] = filter_outliers(grid["top_y_first"], grid["top_valid1"])
    else:
        grid["top_valid1"] = pd.Series(False, index=grid.index)
        grid["top_valid2"] = pd.Series(False, index=grid.index)

    if has_left_pairs:
        grid["left_valid1"] = filter_by_overlap_and_correlation(
            grid["left_x_first"], grid["left_ncc_first"], overlap_left, full_sizeX, pou, ncc_threshold
        )
        grid["left_valid2"] = filter_outliers(grid["left_x_first"], grid["left_valid1"])
    else:
        grid["left_valid1"] = pd.Series(False, index=grid.index)
        grid["left_valid2"] = pd.Series(False, index=grid.index)
    
    rs = []
    for direction, dims_chars, rowcol_group_key in zip(["top", "left"], ["yx", "xy"], ["col", "row"]):
        valid_key = f"{direction}_valid2"
        # Ensure 'valid_key' column exists, even if all False
        if valid_key not in grid.columns: grid[valid_key] = False
            
        valid_grid_subset = grid[grid[valid_key].fillna(False)] # Handle potential NaNs in boolean column
        
        if len(valid_grid_subset) > 0:
            # Primary dimension (e.g., 'top_y_first')
            w1s = valid_grid_subset[f"{direction}_{dims_chars[0]}_first"]
            if len(w1s.dropna()) > 0: # Check for non-NaN values before max/min
                 r1 = np.ceil((w1s.max() - w1s.min()) / 2) if len(w1s) > 1 else 0
            else:
                 r1 = 0

            # Secondary dimension (e.g., 'top_x_first'), grouped
            # Original code seems to use zip(*valid_grid.groupby...)[f"{direction}_{dims[1]}_first"]),
            # which might fail if groups are empty or structure changes.
            # Safer way:
            r2_vals = []
            for _, grp in valid_grid_subset.groupby(rowcol_group_key):
                w2_series = grp[f"{direction}_{dims_chars[1]}_first"].dropna()
                if len(w2_series) > 1:
                    r2_vals.append(np.max(w2_series) - np.min(w2_series))
            r2 = np.ceil(np.max(r2_vals) / 2) if r2_vals else 0
            rs.append(max(r1, r2))
        else:
            rs.append(0) # No valid translations for this direction
    r_repeatability = np.max(rs) if rs else 0 # Max repeatability error

    grid = filter_by_repeatability(grid, r_repeatability, ncc_threshold)
    grid = replace_invalid_translations(grid) # This populates *_second columns
    
    # Pass edge_width to the modified refine_translations
    grid = refine_translations(images_arr, grid, r_repeatability, edge_width) 
    
    tree = compute_maximum_spanning_tree(grid)
    grid = compute_final_position(grid, tree)
    
    prop_dict = {
        "W": full_sizeY, # Report full image dimensions
        "H": full_sizeX,
        "overlap_left": overlap_left,
        "overlap_top": overlap_top,
        "repeatability": r_repeatability,
        "edge_width_used": edge_width # Add this for metadata
    }
    
    return grid, prop_dict


def calculate_pixel_size_microns(
    grid: pd.DataFrame,
    coords_df: pd.DataFrame,
    filename_to_index: Dict[str, int],
    filenames: List[str],
    image_directory: Union[str, Path] = None  # Add image_directory parameter
) -> float:
    """Calculate pixel size in microns by comparing stage and pixel positions.
    
    Parameters
    ----------
    grid : pd.DataFrame
        Registration results with pixel positions
    coords_df : pd.DataFrame
        Original coordinates with stage positions in mm
    filename_to_index : Dict[str, int]
        Mapping from filename to coordinate index
    filenames : List[str]
        Ordered list of filenames
    image_directory : Union[str, Path], optional
        Directory containing the images and acquisition parameters
        
    Returns
    -------
    pixel_size_um : float
        Pixel size in microns
    """
    # Get pairs of tiles with known displacements
    pixel_distances = []
    stage_distances = []
    
    for idx, row in grid.iterrows():
        # Check left neighbor
        if not pd.isna(row['left']):
            left_idx = int(row['left'])
            
            # Get stage positions for both tiles
            curr_coord_idx = filename_to_index[filenames[idx]]
            left_coord_idx = filename_to_index[filenames[left_idx]]
            
            curr_x_mm = coords_df.loc[curr_coord_idx, 'x (mm)']
            curr_y_mm = coords_df.loc[curr_coord_idx, 'y (mm)']
            left_x_mm = coords_df.loc[left_coord_idx, 'x (mm)']
            left_y_mm = coords_df.loc[left_coord_idx, 'y (mm)']
            
            # Calculate stage distance in microns
            stage_dist_um = np.sqrt(
                ((curr_x_mm - left_x_mm) * 1000) ** 2 +
                ((curr_y_mm - left_y_mm) * 1000) ** 2
            )
            
            # Calculate pixel distance
            curr_x_px = row['x_pos']
            curr_y_px = row['y_pos']
            left_x_px = grid.loc[left_idx, 'x_pos']
            left_y_px = grid.loc[left_idx, 'y_pos']
            
            pixel_dist = np.sqrt(
                (curr_x_px - left_x_px) ** 2 +
                (curr_y_px - left_y_px) ** 2
            )
            
            if pixel_dist > 0:  # Avoid division by zero
                pixel_distances.append(pixel_dist)
                stage_distances.append(stage_dist_um)
    
    # Calculate median pixel size
    if pixel_distances:
        pixel_sizes = np.array(stage_distances) / np.array(pixel_distances)
        pixel_size_um = np.median(pixel_sizes)
        print(f"Calculated actual pixel size: {pixel_size_um:.4f} µm/pixel")
        
        # Update acquisition parameters JSON
        try:
            # Find the acquisition parameters JSON file
            if image_directory is None:
                json_path = Path('acquisition parameters.json')
            else:
                # The JSON file is always at the parent level of the image directory
                json_path = Path(image_directory) / 'acquisition parameters.json'
            
            if not json_path.exists():
                raise FileNotFoundError(f"Could not find acquisition parameters.json at {json_path}")
            
            print(f"Found acquisition parameters at: {json_path}")
            
            # Read the current JSON
            with open(json_path, 'r') as f:
                params = json.load(f)
            
            # Calculate the magnification parameters needed for downstream software
            obj_mag = params["objective"]["magnification"]
            obj_tube_lens_mm = params["objective"]["tube_lens_f_mm"]
            tube_lens_mm = params["tube_lens_mm"]
            
            obj_focal_length_mm = obj_tube_lens_mm / obj_mag
            actual_mag = tube_lens_mm / obj_focal_length_mm
            
            print(f"Optical parameters: obj_mag={obj_mag}, obj_tube_lens={obj_tube_lens_mm}mm, tube_lens={tube_lens_mm}mm")
            print(f"Calculated actual_mag: {actual_mag:.4f}")
            
            # Store original value if it exists
            if 'sensor_pixel_size_um' in params:
                original_value = params['sensor_pixel_size_um']
                params['original_sensor_pixel_size_um'] = original_value
                print(f"Preserved original sensor_pixel_size_um: {original_value}")
            
            # Calculate the sensor_pixel_size_um that will give us the correct pixel_size_um
            # downstream software calculates: pixel_size_um = sensor_pixel_size_um / actual_mag
            # So we need: sensor_pixel_size_um = pixel_size_um * actual_mag
            corrected_sensor_pixel_size_um = pixel_size_um * actual_mag
            params['sensor_pixel_size_um'] = corrected_sensor_pixel_size_um
            
            # Write back to JSON
            with open(json_path, 'w') as f:
                json.dump(params, f, indent=4)
                
            print(f"Updated sensor_pixel_size_um to {corrected_sensor_pixel_size_um:.4f} µm")
            print(f"This will result in downstream pixel_size_um = {corrected_sensor_pixel_size_um/actual_mag:.4f} µm/pixel")
            
        except Exception as e:
            print(f"Warning: Could not update acquisition parameters: {e}")
        
        return pixel_size_um
    else:
        raise ValueError("Could not calculate pixel size - no valid tile pairs found")


def update_stage_coordinates(
    grid: pd.DataFrame,
    coords_df: pd.DataFrame,
    filename_to_index: Dict[str, int],
    filenames: List[str],
    pixel_size_um: float,
    reference_idx: int = 0
) -> pd.DataFrame:
    """Update stage coordinates based on registration results.
    
    Parameters
    ----------
    grid : pd.DataFrame
        Registration results with pixel positions
    coords_df : pd.DataFrame
        Original coordinates to update
    filename_to_index : Dict[str, int]
        Mapping from filename to coordinate index
    filenames : List[str]
        Ordered list of filenames
    pixel_size_um : float
        Pixel size in microns
    reference_idx : int
        Index of reference tile (default 0)
        
    Returns
    -------
    updated_coords : pd.DataFrame
        Updated coordinates with new stage positions
    """
    updated_coords = coords_df.copy()
    
    # Get reference position
    ref_coord_idx = filename_to_index[filenames[reference_idx]]
    ref_x_mm = coords_df.loc[ref_coord_idx, 'x (mm)']
    ref_y_mm = coords_df.loc[ref_coord_idx, 'y (mm)']
    ref_x_px = grid.loc[reference_idx, 'x_pos']
    ref_y_px = grid.loc[reference_idx, 'y_pos']
    
    # Determine the FOV column name, default to DEFAULT_FOV_COL
    fov_col_name = DEFAULT_FOV_COL # Or pass as argument if it can vary beyond this
    if fov_col_name not in coords_df.columns:
        # Attempt to infer if a 'fov' like column exists if default is not present
        potential_fov_cols = [col for col in coords_df.columns if 'fov' in col.lower()]
        if potential_fov_cols:
            fov_col_name = potential_fov_cols[0]
            logger.info(f"Default FOV column '{DEFAULT_FOV_COL}' not found. Inferred '{fov_col_name}' as FOV column.")
        else:
            logger.error(f"FOV column '{DEFAULT_FOV_COL}' not found and could not be inferred. Cannot propagate z-slice updates.")
            # Fallback to original behavior: update only the specific row
            for idx, row in grid.iterrows():
                coord_idx = filename_to_index[filenames[idx]]
                delta_x_px = row['x_pos'] - ref_x_px
                delta_y_px = row['y_pos'] - ref_y_px
                delta_x_mm = (delta_x_px * pixel_size_um) / 1000.0
                delta_y_mm = (delta_y_px * pixel_size_um) / 1000.0
                updated_coords.loc[coord_idx, 'x (mm)'] = ref_x_mm + delta_x_mm
                updated_coords.loc[coord_idx, 'y (mm)'] = ref_y_mm + delta_y_mm
                updated_coords.loc[coord_idx, 'x_pos_px'] = row['x_pos']
                updated_coords.loc[coord_idx, 'y_pos_px'] = row['y_pos']
            return updated_coords


    # Update all positions
    for idx, row_data in grid.iterrows(): # Renamed `row` to `row_data` to avoid conflict
        # This coord_idx corresponds to the specific z-slice that was registered
        registered_coord_idx = filename_to_index[filenames[idx]]
        
        # Calculate position relative to reference in pixels for the registered slice
        delta_x_px = row_data['x_pos'] - ref_x_px
        delta_y_px = row_data['y_pos'] - ref_y_px
        
        # Convert to mm
        new_x_mm = ref_x_mm + (delta_x_px * pixel_size_um) / 1000.0
        new_y_mm = ref_y_mm + (delta_y_px * pixel_size_um) / 1000.0

        # Get the FOV identifier for the current registered tile
        # This FOV identifier will be used to update all z-slices belonging to this FOV
        current_fov_identifier = updated_coords.loc[registered_coord_idx, fov_col_name]

        # Find all rows in the original coords_df that belong to this FOV
        fov_rows_indices = updated_coords[updated_coords[fov_col_name] == current_fov_identifier].index
        
        # Update stage coordinates for all these rows (all z-slices of this FOV)
        updated_coords.loc[fov_rows_indices, 'x (mm)'] = new_x_mm
        updated_coords.loc[fov_rows_indices, 'y (mm)'] = new_y_mm
        
        # Add registration info (pixel positions) only to the specific registered slice's row for clarity,
        # or decide if this should also be copied or left blank for other z-slices.
        # For now, only updating the registered slice with pixel info.
        updated_coords.loc[registered_coord_idx, 'x_pos_px'] = row_data['x_pos']
        updated_coords.loc[registered_coord_idx, 'y_pos_px'] = row_data['y_pos']
        # For other z-slices of the same FOV, pixel positions might not be directly applicable
        # or could be set to NaN or a placeholder if needed.
        other_z_slice_indices = fov_rows_indices.difference([registered_coord_idx])
        if not other_z_slice_indices.empty:
            updated_coords.loc[other_z_slice_indices, 'x_pos_px'] = np.nan # Or some other indicator
            updated_coords.loc[other_z_slice_indices, 'y_pos_px'] = np.nan


    return updated_coords


def read_coordinates_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Read coordinates from CSV file.
    
    Parameters
    ----------
    csv_path : Union[str, Path]
        Path to coordinates CSV file
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing coordinates
    """
    return pd.read_csv(csv_path)

def read_tiff_images_for_region(
    directory: Union[str, Path], 
    pattern: str,
    region: str,
    fov_re: Pattern[str] = DEFAULT_FOV_RE,
    z_slice_to_keep: Optional[int] = 0
) -> Dict[str, np.ndarray]:
    """Read TIFF images from directory for a specific region.
    
    Parameters
    ----------
    directory : Union[str, Path]
        Directory containing TIFF images
    pattern : str
        Glob pattern to match TIFF files
    region : str
        Region name to filter for
    fov_re : Pattern[str]
        Compiled regular expression to extract region, FOV, and z-level
    z_slice_to_keep : Optional[int]
        The specific z-slice to use for registration
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping filenames to image arrays
    """
    directory = Path(directory)
    images = {}
    
    # Get all matching files
    all_tiff_paths = list(directory.glob(pattern))
    if not all_tiff_paths:
        print(f"No files matching pattern '{pattern}' found in {directory}")
        return images
    
    # Filter for the specific region
    region_paths = []
    for path in all_tiff_paths:
        m = fov_re.search(path.name)
        if m and 'region' in m.groupdict():
            if m.group('region') == region:
                region_paths.append(path)
    
    if not region_paths:
        print(f"No files found for region '{region}'")
        return images
    
    # Now apply z-slice filtering to region-specific files
    selected_tiff_paths = []
    if z_slice_to_keep is not None:
        fov_to_files_map: Dict[int, List[Tuple[int, Path]]] = {}
        
        for path in region_paths:
            m = fov_re.search(path.name)
            if m:
                try:
                    fov = int(m.group('fov'))
                    z_level = int(m.group('z_level'))
                    
                    if fov not in fov_to_files_map:
                        fov_to_files_map[fov] = []
                    fov_to_files_map[fov].append((z_level, path))
                except (IndexError, ValueError):
                    logger.warning(f"Could not parse FOV/Z from {path.name}")
        
        for fov, z_files in fov_to_files_map.items():
            if not z_files:
                continue
            
            # Find the file matching z_slice_to_keep
            target_slice_found = False
            for z_level, path in z_files:
                if z_level == z_slice_to_keep:
                    selected_tiff_paths.append(path)
                    target_slice_found = True
                    logger.debug(f"Selected {path.name} for FOV {fov} (z-slice {z_slice_to_keep})")
                    break
            
            if not target_slice_found:
                # Fallback: use the lowest z_level
                z_files.sort(key=lambda x: x[0])
                fallback_path = z_files[0][1]
                selected_tiff_paths.append(fallback_path)
                logger.warning(f"Z-slice {z_slice_to_keep} not found for FOV {fov}. Using {fallback_path.name} instead.")
    else:
        selected_tiff_paths = region_paths
    
    if not selected_tiff_paths:
        print(f"No files selected for region '{region}' after z-slice filtering")
        return images
    
    # Use a reasonable number of processes
    n_processes = min(cpu_count(), len(selected_tiff_paths))
    
    # Load images in parallel
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(load_single_image, selected_tiff_paths),
            total=len(selected_tiff_paths),
            desc=f"Loading images for region {region}"
        ))
    
    # Filter out failed loads and update dictionary
    images.update({k: v for k, v in results if v is not None})
    
    print(f"Successfully loaded {len(images)}/{len(selected_tiff_paths)} images for region {region}")
    
    return images



def load_single_image(path: Path) -> Tuple[str, Optional[np.ndarray]]:
    """Load a single image with error handling.
    
    Parameters
    ----------
    path : Path
        Path to the TIFF file
        
    Returns
    -------
    Tuple[str, Optional[np.ndarray]]
        Tuple of (filename, image_array) or (filename, None) if loading failed
    """
    try:
        return path.name, tifffile.imread(str(path))
    except MemoryError:
        logger.error(f"MemoryError reading {path}. File might be too large or corrupt.")
        return path.name, None
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return path.name, None

def read_tiff_images(
    directory: Union[str, Path], 
    pattern: str,
    fov_re: Pattern[str] = DEFAULT_FOV_RE,
    z_slice_to_keep: Optional[int] = 0
) -> Dict[str, np.ndarray]:
    """Read TIFF images from directory in parallel, selecting specific z-slices.
    
    This is kept for backward compatibility but calls the region-specific version
    for all regions found.
    """
    # For backward compatibility, read all regions
    directory = Path(directory)
    all_images = {}
    
    # Get all matching files to find regions
    all_tiff_paths = list(directory.glob(pattern))
    regions = set()
    
    for path in all_tiff_paths:
        m = fov_re.search(path.name)
        if m and 'region' in m.groupdict():
            regions.add(m.group('region'))
    
    # Read images for each region
    for region in regions:
        region_images = read_tiff_images_for_region(
            directory=directory,
            pattern=pattern,
            region=region,
            fov_re=fov_re,
            z_slice_to_keep=z_slice_to_keep
        )
        all_images.update(region_images)
    
    return all_images


def update_stage_coordinates_multi_z(
    grid: pd.DataFrame,
    coords_df: pd.DataFrame,
    filename_to_index: Dict[str, int],
    filenames: List[str],
    pixel_size_um: float,
    full_coords_df: pd.DataFrame,
    region: str,
    reference_idx: int = 0
) -> pd.DataFrame:
    """Update stage coordinates for all z-slices based on registration results.
    
    Parameters
    ----------
    grid : pd.DataFrame
        Registration results with pixel positions
    coords_df : pd.DataFrame
        Region-specific coordinates to update
    filename_to_index : Dict[str, int]
        Mapping from filename to coordinate index
    filenames : List[str]
        Ordered list of filenames
    pixel_size_um : float
        Pixel size in microns
    full_coords_df : pd.DataFrame
        Full coordinates DataFrame (all regions)
    region : str
        Current region being processed
    reference_idx : int
        Index of reference tile (default 0)
        
    Returns
    -------
    updated_coords : pd.DataFrame
        Updated coordinates with new stage positions
    """
    # Work with the full dataframe to update all z-slices
    updated_coords = full_coords_df.copy()
    
    # Get reference position
    ref_coord_idx = filename_to_index[filenames[reference_idx]]
    ref_x_mm = coords_df.loc[ref_coord_idx, 'x (mm)']
    ref_y_mm = coords_df.loc[ref_coord_idx, 'y (mm)']
    ref_x_px = grid.loc[reference_idx, 'x_pos']
    ref_y_px = grid.loc[reference_idx, 'y_pos']
    
    # Get FOV column name
    fov_col_name = DEFAULT_FOV_COL
    if fov_col_name not in coords_df.columns:
        potential_fov_cols = [col for col in coords_df.columns if 'fov' in col.lower()]
        if potential_fov_cols:
            fov_col_name = potential_fov_cols[0]
            logger.info(f"Using '{fov_col_name}' as FOV column.")
    
    # Process each registered tile
    for idx, row in grid.iterrows():
        filename = filenames[idx]
        coord_idx = filename_to_index[filename]
        
        # Get the FOV from this coordinate
        fov = coords_df.loc[coord_idx, fov_col_name]
        
        # Calculate the position update
        delta_x_px = row['x_pos'] - ref_x_px
        delta_y_px = row['y_pos'] - ref_y_px
        delta_x_mm = (delta_x_px * pixel_size_um) / 1000.0
        delta_y_mm = (delta_y_px * pixel_size_um) / 1000.0
        
        new_x_mm = ref_x_mm + delta_x_mm
        new_y_mm = ref_y_mm + delta_y_mm
        
        # Update all z-slices for this FOV in the same region
        mask = (updated_coords['region'] == region) & (updated_coords[fov_col_name] == fov)
        updated_coords.loc[mask, 'x (mm)'] = new_x_mm
        updated_coords.loc[mask, 'y (mm)'] = new_y_mm
        updated_coords.loc[mask, 'x_pos_px'] = row['x_pos']
        updated_coords.loc[mask, 'y_pos_px'] = row['y_pos']
    
    return updated_coords


def detect_channels(directory: Union[str, Path]) -> Dict[str, List[Path]]:
    """Detect available channels in the directory.
    
    Parameters
    ----------
    directory : Union[str, Path]
        Directory containing TIFF images
        
    Returns
    -------
    Dict[str, List[Path]]
        Dictionary mapping channel type to list of file paths
    """
    directory = Path(directory)
    tiff_files = list(directory.glob("*.tiff"))
    
    # Channel categories
    channels = {
        "fluorescence": [],
        "brightfield": []
    }
    
    # Regular expressions for channel detection
    fluor_patterns = [
        r"(\d+)_nm_Ex\.tiff$",  # Matches wavelength patterns like 405_nm_Ex.tiff
        r"Fluorescence",  # Generic fluorescence indicator
    ]
    bf_patterns = [
        r"BF",  # Brightfield indicator
        r"brightfield",
        r"bright_field"
    ]
    
    for file in tiff_files:
        filename = file.name.lower()
        
        # Check for fluorescence channels
        is_fluor = any(re.search(pattern, file.name, re.IGNORECASE) for pattern in fluor_patterns)
        if is_fluor:
            channels["fluorescence"].append(file)
            continue
            
        # Check for brightfield channels
        is_bf = any(re.search(pattern, filename) for pattern in bf_patterns)
        if is_bf:
            channels["brightfield"].append(file)
            continue
    
    return channels

def select_channel_pattern(directory: Union[str, Path]) -> str:
    """Automatically select an appropriate channel pattern.
    
    Parameters
    ----------
    directory : Union[str, Path]
        Directory containing TIFF images
        
    Returns
    -------
    str
        Glob pattern for the selected channel
    """
    channels = detect_channels(directory)
    
    # Prefer fluorescence channels if available
    if channels["fluorescence"]:
        # Group fluorescence files by wavelength
        wavelengths = {}
        for file in channels["fluorescence"]:
            match = re.search(r"(\d+)_nm_Ex\.tiff$", file.name)
            if match:
                wavelength = int(match.group(1))
                if wavelength not in wavelengths:
                    wavelengths[wavelength] = []
                wavelengths[wavelength].append(file)
        
        if wavelengths:
            # Select the shortest wavelength (typically best for registration)
            wavelength = min(wavelengths.keys())
            pattern = f"*{wavelength}_nm_Ex.tiff"
            print(f"Selected fluorescence channel: {pattern}")
            return pattern
        
        # If no wavelength pattern found, use the first fluorescence file pattern
        example_file = channels["fluorescence"][0].name
        pattern = f"*{example_file.split('_Fluorescence_')[-1]}"
        print(f"Selected fluorescence channel: {pattern}")
        return pattern
    
    # Fall back to brightfield if available
    if channels["brightfield"]:
        example_file = channels["brightfield"][0].name
        if len(channels["brightfield"]) > 1:
            print("Warning: Multiple brightfield files found, using first one as pattern")
        pattern = f"*{example_file.split('_BF_')[-1]}"
        print(f"Selected brightfield channel: {pattern}")
        return pattern
    
    # If no specific patterns found, use all TIFF files
    print("Warning: No specific channel pattern detected, using all TIFF files")
    return "*.tiff"



def _compute_direction_translations(args):
    direction, grid, images_arr, full_sizeY, full_sizeX, edge_width = args
    results = []
    
    for i2_idx, g_row in grid.iterrows():
        i1_idx_nullable = g_row[direction]
        if pd.isna(i1_idx_nullable):
            continue
        
        i1_idx = int(i1_idx_nullable) # Convert from Int32Dtype to int
        # i2_idx is already an int from iterrows() index

        image1_full = images_arr[i1_idx]
        image2_full = images_arr[i2_idx]
        
        # Extract edges
        if direction == "left": # i2 is to the right of i1
            current_w = min(edge_width, full_sizeX)
            edge1 = image1_full[:, -current_w:]
            edge2 = image2_full[:, :current_w]
            # Offset of edge1's origin within image1_full (used for converting back to global)
            offset_y_edge1_in_full = 0
            offset_x_edge1_in_full = full_sizeX - current_w
        elif direction == "top": # i2 is below i1
            current_w = min(edge_width, full_sizeY)
            edge1 = image1_full[-current_w:, :]
            edge2 = image2_full[:current_w, :]
            # Offset of edge1's origin within image1_full
            offset_y_edge1_in_full = full_sizeY - current_w
            offset_x_edge1_in_full = 0
        else:
            continue # Should not happen
        
        if edge1.size == 0 or edge2.size == 0: # Skip if edges are empty (e.g., edge_width > image_dim)
            logger.warning(f"Empty edge strip for initial translation: pair ({i1_idx}, {i2_idx}), direction {direction}. Skipping.")
            continue

        edge_sizeY, edge_sizeX = edge1.shape

        # PCM and peak finding on edges
        PCM_on_edges = pcm(edge1, edge2).real
        # Limits for multi_peak_max and interpret_translation are based on edge dimensions
        # (translation of edge2 relative to edge1)
        lims_edge_relative = np.array([[-edge_sizeY, edge_sizeY], [-edge_sizeX, edge_sizeX]])
        
        yins_edge, xins_edge, _ = multi_peak_max(PCM_on_edges)
        
        # interpret_translation works with edges and edge-based limits,
        # returns (ncc_val, y_best_edge_relative, x_best_edge_relative)
        ncc_val, y_best_edge_relative, x_best_edge_relative = interpret_translation(
            edge1, edge2, yins_edge, xins_edge, *lims_edge_relative[0], *lims_edge_relative[1]
        )
        
        # Convert edge-relative translation back to global translation
        # Global translation = (translation of edge2 w.r.t. edge1) + (translation of edge1's origin w.r.t. image1's origin)
        # This is translation of image2_full w.r.t image1_full
        y_best_global = y_best_edge_relative + offset_y_edge1_in_full
        x_best_global = x_best_edge_relative + offset_x_edge1_in_full
        
        # if direction == "left":
        #     # y_best_global = y_best_edge_relative (since edge1_origin_y is 0 in full_im1)
        #     # x_best_global = x_best_edge_relative + (full_sizeX - current_w)
        #     y_best_global = y_best_edge_relative 
        #     x_best_global = x_best_edge_relative + (full_sizeX - current_w)
        # elif direction == "top":
        #     # y_best_global = y_best_edge_relative + (full_sizeY - current_w)
        #     # x_best_global = x_best_edge_relative (since edge1_origin_x is 0 in full_im1)
        #     y_best_global = y_best_edge_relative + (full_sizeY - current_w)
        #     x_best_global = x_best_edge_relative
        # else: # Should not happen
        #     y_best_global, x_best_global = y_best_edge_relative, x_best_edge_relative


        results.append((i2_idx, direction, (ncc_val, y_best_global, x_best_global)))
    
    return results

def register_and_update_coordinates(
    image_directory: Union[str, Path],
    csv_path: Union[str, Path],
    output_csv_path: Union[str, Path],
    channel_pattern: Optional[str] = None,
    overlap_diff_threshold: float = 10,
    pou: float = 3,
    ncc_threshold: float = 0.5,
    skip_backup: bool = False,
    z_slice_for_registration: Optional[int] = 0,
    edge_width: int = DEFAULT_EDGE_WIDTH
) -> pd.DataFrame:
    """Register tiles and update stage coordinates for all regions.
    
    Parameters
    ----------
    image_directory : Union[str, Path]
        Directory containing image files
    csv_path : Union[str, Path]
        Path to coordinates CSV file
    output_csv_path : Union[str, Path]
        Path to save updated coordinates
    channel_pattern : Optional[str]
        Pattern to match image files (e.g., "*.tiff")
    overlap_diff_threshold : float
        Allowed difference from initial guess (percentage)
    pou : float
        Percent overlap uncertainty
    ncc_threshold : float
        Normalized cross correlation threshold
    skip_backup : bool
        Whether to skip creating a backup of the coordinates file
    z_slice_for_registration : Optional[int]
        Which z-slice to use for registration (default: 0)
    edge_width : int
        Width of the edge strips (in pixels) to use for registration
        
    Returns
    -------
    pd.DataFrame
        Updated coordinates DataFrame
    """
    image_directory = Path(image_directory)
    csv_path = Path(csv_path)
    output_csv_path = Path(output_csv_path)
    
    # Create original_coordinates directory if it doesn't exist
    original_coords_dir = image_directory / "original_coordinates"
    original_coords_dir.mkdir(exist_ok=True)
    
    # Look for coordinates.csv in the timepoint directory
    timepoint_dir = image_directory / "0"
    if timepoint_dir.exists():
        coords_file = timepoint_dir / "coordinates.csv"
        if coords_file.exists():
            csv_path = coords_file
            print(f"Found coordinates file at: {csv_path}")
            # Get timepoint from the subdirectory name (e.g., "0" from the path)
            timepoint = timepoint_dir.name
    
    # Create backup of original coordinates
    if not skip_backup:
        backup_path = original_coords_dir / f"original_coordinates_{timepoint}.csv"
        import shutil
        shutil.copy2(csv_path, backup_path)
        print(f"Created backup of original coordinates at: {backup_path}")
    
    # Read coordinates
    coords_df = read_coordinates_csv(csv_path)
    
    # Auto-detect channel pattern if not provided
    if channel_pattern is None:
        # Look in the 0 subdirectory for TIFF files
        tiff_dir = image_directory / "0"
        if tiff_dir.exists():
            channel_pattern = select_channel_pattern(tiff_dir)
        else:
            channel_pattern = select_channel_pattern(image_directory)
    
    # Initialize the updated coordinates with a copy
    updated_coords = coords_df.copy()
    
    # Group by region
    regions = coords_df['region'].unique()
    print(f"Found {len(regions)} regions to process: {regions}")
    
    for region in regions:
        print(f"\nProcessing region: {region}")
        
        # Filter coordinates for this region
        region_coords = coords_df[coords_df['region'] == region].copy()
        
        # Read images for this region only
        # Look in the 0 subdirectory for TIFF files
        tiff_dir = image_directory / "0"
        if tiff_dir.exists():
            region_images = read_tiff_images_for_region(
                directory=tiff_dir,
                pattern=channel_pattern,
                region=region,
                z_slice_to_keep=z_slice_for_registration
            )
        else:
            region_images = read_tiff_images_for_region(
                directory=image_directory,
                pattern=channel_pattern,
                region=region,
                z_slice_to_keep=z_slice_for_registration
            )
        
        if not region_images:
            print(f"No images found for region {region}, skipping")
            continue
        
        # Extract tile indices
        filenames = list(region_images.keys())
        rows, cols, filename_to_index = extract_tile_indices(filenames, region_coords)
        
        try:
            # Register tiles
            grid, prop_dict = register_tiles(
                images=list(region_images.values()),
                rows=rows,
                cols=cols,
                edge_width=edge_width,
                overlap_diff_threshold=overlap_diff_threshold,
                pou=pou,
                ncc_threshold=ncc_threshold
            )
            
            # Calculate pixel size
            pixel_size_um = calculate_pixel_size_microns(
                grid=grid,
                coords_df=region_coords,
                filename_to_index=filename_to_index,
                filenames=filenames,
                image_directory=image_directory
            )
            
            # Update stage coordinates for this region
            region_updated = update_stage_coordinates_multi_z(
                grid=grid,
                coords_df=region_coords,
                filename_to_index=filename_to_index,
                filenames=filenames,
                pixel_size_um=pixel_size_um,
                full_coords_df=updated_coords,
                region=region
            )
            
            # Update the main dataframe with region results
            updated_coords.update(region_updated)
            
        except Exception as e:
            print(f"Error processing region {region}: {e}")
            # Clean up GPU memory after error
            try:
                import cupy as cp
                # Reset CUDA device
                cp.cuda.Device().synchronize()
                cp.cuda.runtime.deviceReset()
                # Free memory pools
            except:
                pass
            continue
        
        # Force GPU memory cleanup between regions
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            cp.cuda.Device().synchronize()
        except:
            pass
    
    # Save updated coordinates in the timepoint directory
    timepoint_output_path = timepoint_dir / "coordinates.csv"
    updated_coords.to_csv(timepoint_output_path, index=False)
    print(f"Saved updated coordinates to: {timepoint_output_path}")
    
    return updated_coords


def process_multiple_timepoints(
    base_directory: Union[str, Path],
    overlap_diff_threshold: float = 10,
    pou: float = 3,
    ncc_threshold: float = 0.5,
    edge_width: int = DEFAULT_EDGE_WIDTH
) -> Dict[int, pd.DataFrame]:
    """Process multiple timepoints from a directory.
    
    Parameters
    ----------
    base_directory : Union[str, Path]
        Base directory containing timepoint subdirectories
    overlap_diff_threshold : float
        Allowed difference from initial guess (percentage)
    pou : float
        Percent overlap uncertainty
    ncc_threshold : float
        Normalized cross correlation threshold
    edge_width : int
        Width of the edge strips (in pixels) to use for registration
        
    Returns
    -------
    Dict[int, pd.DataFrame]
        Dictionary mapping timepoint to updated coordinates DataFrame
    """
    base_directory = Path(base_directory)
    if not base_directory.is_dir():
        raise ValueError(f"Base directory does not exist: {base_directory}")
        
    # Create original_coordinates directory
    backup_dir = base_directory / "original_coordinates"
    backup_dir.mkdir(exist_ok=True)
        
    # Find all timepoint directories (numbered subdirectories)
    timepoint_dirs = sorted([d for d in base_directory.iterdir() if d.is_dir() and d.name.isdigit()])
    if not timepoint_dirs:
        raise ValueError(f"No timepoint directories found in {base_directory}")
        
    print(f"Found {len(timepoint_dirs)} timepoint directories")
    
    # Process each timepoint
    results = {}
    for tp_dir in timepoint_dirs:
        timepoint = int(tp_dir.name)
        print(f"\nProcessing timepoint {timepoint}")
        
        # Find coordinates.csv in timepoint directory
        tp_coords = list(tp_dir.glob("coordinates.csv"))
        if not tp_coords:
            print(f"Warning: No coordinates.csv found in timepoint {timepoint}, skipping")
            continue
        coords_csv = tp_coords[0]
        
        # Create backup of this timepoint's coordinates
        backup_path = backup_dir / f"original_coordinates_{timepoint}.csv"
        if not backup_path.exists():
            print(f"Creating backup of timepoint {timepoint} coordinates at {backup_path}")
            import shutil
            shutil.copy2(coords_csv, backup_path)
            
        try:
            # Process this timepoint
            updated_coords = register_and_update_coordinates(
                image_directory=base_directory,  # Use base directory instead of timepoint directory
                csv_path=coords_csv,
                output_csv_path=coords_csv,  # Overwrite the original
                channel_pattern=None,  # Auto-detect
                overlap_diff_threshold=overlap_diff_threshold,
                pou=pou,
                ncc_threshold=ncc_threshold,
                skip_backup=True,  # Skip backup since we already made one
                z_slice_for_registration=0,
                edge_width=edge_width
            )
            results[timepoint] = updated_coords
            print(f"Successfully processed timepoint {timepoint}")
        except Exception as e:
            print(f"Error processing timepoint {timepoint}: {e}")
            continue
            
    if not results:
        raise ValueError("No timepoints were successfully processed")
    
    return results
