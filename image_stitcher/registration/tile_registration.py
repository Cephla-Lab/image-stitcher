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

KEY IMPROVEMENT: Neighborhood-aware batching ensures that tiles and their neighbors
are always loaded in the same batch, eliminating cross-batch translation failures
that occurred in previous implementations. This guarantees complete translation
computation for all valid tile pairs.

CRITICAL FIX: Fixed broken neighbor assignment that was converting NaN values to
garbage integer indices, creating fake neighbor relationships between tiles that
shouldn't be neighbors. Now properly preserves NaN for missing neighbors.

NOTE: This module now uses a single, unified registration function (register_tiles_batched)
that replaces the previous broken register_tiles function.
"""
import os
from dataclasses import dataclass
from pathlib import Path
import re
import logging
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import json
import gc
import psutil
from typing import Dict, List, Optional, Tuple, Union, Pattern, Iterator, Set
from ._typing_utils import BoolArray
from ._typing_utils import Float
from ._typing_utils import NumArray

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

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
# Additional regex for multi-page TIFF files with "_stack" suffix
MULTIPAGE_FOV_RE = re.compile(r"(?P<region>\w+)_(?P<fov>0|[1-9]\d*)_stack", re.I)
DEFAULT_FOV_COL = "fov"
DEFAULT_X_COL = "x (mm)"
DEFAULT_Y_COL = "y (mm)"

# TODO: TECHNICAL DEBT - This is a poor design pattern that should be refactored
# 
# PROBLEM: This helper function creates duplicate regex logic and inconsistent interfaces.
# Multiple functions now have two different code paths for filename parsing, making
# the codebase harder to maintain and debug.
#
# BETTER PATTERN: Create a proper FileFormat abstraction with:
#   1. FileFormat enum (REGULAR_TIFF, MULTIPAGE_TIFF)
#   2. FileFormatDetector class to identify format from filename
#   3. FileFormatHandler interface with format-specific parsing logic
#   4. Update all filename parsing to use the abstraction consistently
#
# This would:
#   - Centralize format detection logic
#   - Make adding new formats easier
#   - Eliminate duplicate regex patterns
#   - Provide consistent interfaces across all functions
#   - Enable proper unit testing of format-specific behavior
#
# For now, this helper function provides a quick fix for multi-page TIFF support
# but should be replaced with the proper abstraction in a future refactor.

def parse_filename_fov_info(filename: str) -> Optional[Dict[str, Union[str, int]]]:
    """TEMPORARY HELPER: Parse FOV info from filename, handling both regular and multi-page TIFF formats.
    
    WARNING: This function duplicates regex logic and should be replaced with a proper
    FileFormat abstraction. See TODO comment above for the recommended approach.
    
    Parameters
    ----------
    filename : str
        The filename to parse
        
    Returns
    -------
    Optional[Dict[str, Union[str, int]]]
        Dictionary with 'region', 'fov', and optionally 'z_level' keys, or None if no match
    """
    # Try regular TIFF pattern first
    m = DEFAULT_FOV_RE.search(filename)
    if m:
        return {
            'region': m.group('region'),
            'fov': int(m.group('fov')),
            'z_level': int(m.group('z_level'))
        }
    
    # Try multi-page TIFF pattern
    m = MULTIPAGE_FOV_RE.search(filename)
    if m:
        return {
            'region': m.group('region'),
            'fov': int(m.group('fov')),
            # No z_level for multi-page TIFFs - let GUI/parameters control z-selection
        }
    
    return None

# Constants for tile registration
MIN_PITCH_FOR_FACTOR = 0.1  # Minimum pitch value to use factor-based tolerance (mm)
DEFAULT_ABSOLUTE_TOLERANCE = 0.05  # Default absolute tolerance (mm)
ROW_TOL_FACTOR = 0.20  # Tolerance factor for row clustering
COL_TOL_FACTOR = 0.20  # Tolerance factor for column clustering
DEFAULT_EDGE_WIDTH = 256  # Increased from 64 to 256 for better correlation with 2048x2048 tiles

@dataclass
class Neighborhood:
    """A tile and its required neighbors for translation computation."""
    center_idx: int
    left_idx: Optional[int] = None
    top_idx: Optional[int] = None
    
    def get_all_indices(self) -> Set[int]:
        """Get all unique image indices needed for this neighborhood."""
        indices = {self.center_idx}
        if self.left_idx is not None:
            indices.add(self.left_idx)
        if self.top_idx is not None:
            indices.add(self.top_idx)
        return indices

def create_neighborhoods(master_grid: pd.DataFrame) -> List[Neighborhood]:
    """Create neighborhoods for all tiles that need translation computation."""
    neighborhoods = []
    
    for idx, row in master_grid.iterrows():
        # Only create neighborhood if tile has at least one neighbor to compute with
        left_neighbor = None if pd.isna(row['left']) else int(row['left'])
        top_neighbor = None if pd.isna(row['top']) else int(row['top'])
        
        if left_neighbor is not None or top_neighbor is not None:
            neighborhoods.append(Neighborhood(
                center_idx=idx,
                left_idx=left_neighbor,
                top_idx=top_neighbor
            ))
    
    return neighborhoods

def group_neighborhoods_into_batches(
    neighborhoods: List[Neighborhood], 
    selected_tiff_paths: List[Path], 
    max_memory_bytes: int
) -> List[List[Neighborhood]]:
    """Group neighborhoods into memory-constrained batches."""
    
    # Estimate memory per image
    first_image = tifffile.imread(str(selected_tiff_paths[0]))
    bytes_per_image = first_image.nbytes * 4  # 4x overhead
    del first_image
    gc.collect()
    
    batches = []
    current_batch = []
    current_images = set()
    
    for neighborhood in neighborhoods:
        # Calculate images needed for this neighborhood
        neighborhood_images = neighborhood.get_all_indices()
        
        # Check if adding this neighborhood would exceed memory limit
        new_images = neighborhood_images - current_images
        memory_needed = len(current_images | neighborhood_images) * bytes_per_image
        
        if memory_needed > max_memory_bytes and current_batch:
            # Start new batch
            batches.append(current_batch)
            current_batch = [neighborhood]
            current_images = neighborhood_images.copy()
        else:
            # Add to current batch
            current_batch.append(neighborhood)
            current_images |= neighborhood_images
    
    # Add final batch
    if current_batch:
        batches.append(current_batch)
    
    return batches

def load_batch_for_neighborhoods(
    neighborhood_batch: List[Neighborhood],
    selected_tiff_paths: List[Path],
    all_filenames: List[str]
) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """Load all unique images needed for a batch of neighborhoods."""
    
    # Get all unique image indices needed
    all_needed_indices = set()
    for neighborhood in neighborhood_batch:
        all_needed_indices.update(neighborhood.get_all_indices())
    
    needed_indices_list = sorted(list(all_needed_indices))
    
    # Load all needed images
    needed_paths = [selected_tiff_paths[i] for i in needed_indices_list]
    batch_images = load_batch_images(needed_paths)
    
    return batch_images, needed_indices_list

def process_neighborhood_batch(
    neighborhood_batch: List[Neighborhood],
    batch_images: Dict[str, np.ndarray],
    needed_indices_list: List[int],
    selected_tiff_paths: List[Path],
    all_filenames: List[str],
    master_grid: pd.DataFrame,
    edge_width: int
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Process translations for a batch of neighborhoods."""
    
    # Create index mapping: master_idx -> batch_array_idx
    master_to_batch_map = {master_idx: batch_idx 
                          for batch_idx, master_idx in enumerate(needed_indices_list)}
    
    # Create batch image array
    batch_image_array = np.array([batch_images[selected_tiff_paths[i].name] 
                                 for i in needed_indices_list])
    
    # Get image dimensions
    full_sizeY, full_sizeX = batch_image_array.shape[1:3]
    
    # Results storage: {tile_idx: {direction: {ncc:, y:, x:}}}
    results = {}
    
    # Process each neighborhood
    for neighborhood in neighborhood_batch:
        center_idx = neighborhood.center_idx
        results[center_idx] = {}
        
        # Process left translation if exists
        if neighborhood.left_idx is not None:
            left_batch_idx = master_to_batch_map[neighborhood.left_idx]
            center_batch_idx = master_to_batch_map[center_idx]
            
            translation_result = compute_single_translation(
                batch_image_array[left_batch_idx],  # i1 (left image)
                batch_image_array[center_batch_idx],  # i2 (center image)
                "left",
                full_sizeY, full_sizeX, edge_width
            )
            
            if translation_result is not None:
                results[center_idx]["left"] = {
                    "ncc": translation_result[0],
                    "y": translation_result[1], 
                    "x": translation_result[2]
                }
        
        # Process top translation if exists
        if neighborhood.top_idx is not None:
            top_batch_idx = master_to_batch_map[neighborhood.top_idx]
            center_batch_idx = master_to_batch_map[center_idx]
            
            translation_result = compute_single_translation(
                batch_image_array[top_batch_idx],   # i1 (top image)
                batch_image_array[center_batch_idx], # i2 (center image)  
                "top",
                full_sizeY, full_sizeX, edge_width
            )
            
            if translation_result is not None:
                results[center_idx]["top"] = {
                    "ncc": translation_result[0],
                    "y": translation_result[1],
                    "x": translation_result[2]
                }
    
    return results

def compute_single_translation(
    image1_full: np.ndarray,
    image2_full: np.ndarray, 
    direction: str,
    full_sizeY: int, full_sizeX: int, edge_width: int
) -> Optional[Tuple[float, float, float]]:
    """Compute translation between two images for given direction."""
    
    try:
        # Extract edges based on direction
        if direction == "left":  # i2 is to the right of i1
            current_w = min(edge_width, full_sizeX)
            edge1 = image1_full[:, -current_w:]
            edge2 = image2_full[:, :current_w]
            offset_y_edge1_in_full = 0
            offset_x_edge1_in_full = full_sizeX - current_w
        elif direction == "top":  # i2 is below i1
            current_w = min(edge_width, full_sizeY)
            edge1 = image1_full[-current_w:, :]
            edge2 = image2_full[:current_w, :]
            offset_y_edge1_in_full = full_sizeY - current_w
            offset_x_edge1_in_full = 0
        else:
            return None
        
        if edge1.size == 0 or edge2.size == 0:
            return None
            
        edge_sizeY, edge_sizeX = edge1.shape
        
        # PCM and peak finding on edges
        PCM_on_edges = pcm(edge1, edge2).real
        lims_edge_relative = np.array([[-edge_sizeY, edge_sizeY], [-edge_sizeX, edge_sizeX]])
        
        yins_edge, xins_edge, _ = multi_peak_max(PCM_on_edges)
        
        # interpret_translation
        ncc_val, y_best_edge_relative, x_best_edge_relative = interpret_translation(
            edge1, edge2, yins_edge, xins_edge, 
            *lims_edge_relative[0], *lims_edge_relative[1]
        )
        
        # Convert to global coordinates
        y_best_global = y_best_edge_relative + offset_y_edge1_in_full
        x_best_global = x_best_edge_relative + offset_x_edge1_in_full
        
        return (ncc_val, y_best_global, x_best_global)
        
    except Exception as e:
        logger.warning(f"Failed to compute translation for direction {direction}: {e}")
        return None

def calculate_safe_batch_size(first_image_path: Path, memory_fraction: float = 0.75) -> int:
    """Calculate safe batch size based on available memory and first image size."""
    first_image = tifffile.imread(str(first_image_path))
    image_memory = first_image.nbytes
    del first_image
    gc.collect()
    
    processing_overhead = 4  # Conservative 4x multiplier
    total_per_image = image_memory * processing_overhead
    
    available_memory = psutil.virtual_memory().available * memory_fraction
    max_batch_size = max(1, int(available_memory // total_per_image))
    
    print(f"Image memory: {image_memory/1e6:.1f}MB, "
          f"Available: {available_memory/1e9:.1f}GB, "
          f"Batch size: {max_batch_size}")
    
    return max_batch_size

def batch_paths(paths: List[Path], batch_size: int) -> Iterator[List[Path]]:
    """Split paths into batches of specified size."""
    for i in range(0, len(paths), batch_size):
        yield paths[i:i + batch_size]

def load_batch_images(batch_paths: List[Path]) -> Dict[str, np.ndarray]:
    """Load a batch of images using multiprocessing."""
    n_processes = min(cpu_count(), len(batch_paths))
    
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(load_single_image, batch_paths),
            total=len(batch_paths),
            desc="Loading batch"
        ))
    
    return {k: v for k, v in results if v is not None}

def register_tiles_batched(
    selected_tiff_paths: List[Path],
    region_coords: pd.DataFrame,
    edge_width: int = DEFAULT_EDGE_WIDTH,
    overlap_diff_threshold: float = 10,
    pou: float = 3,
    ncc_threshold: float = 0.5,
    z_slice_to_keep: Optional[int] = 0
) -> Tuple[pd.DataFrame, dict]:
    """Register tiles using memory-constrained batched processing."""
    
    if not selected_tiff_paths:
        return pd.DataFrame(), {}
    
    # Calculate safe batch size
    batch_size = calculate_safe_batch_size(selected_tiff_paths[0])
    
    # DEBUG: Force small batch size for testing
    batch_size = min(batch_size, 20)
    print(f"DEBUG: Using batch size: {batch_size}")
    
    # Create complete grid structure BEFORE batching
    all_filenames = [p.name for p in selected_tiff_paths]
    rows, cols, filename_to_index = extract_tile_indices(all_filenames, region_coords)
    
    # DEBUG: Show grid structure
    print(f"DEBUG: Grid dimensions: {len(rows)} tiles, {len(set(rows))} unique rows, {len(set(cols))} unique columns")
    print(f"DEBUG: Row range: {min(rows)} to {max(rows)}")
    print(f"DEBUG: Col range: {min(cols)} to {max(cols)}")
    
    # DEBUG: Show detailed FOV to grid position mapping
    print("\nDEBUG: FOV to Grid Position Mapping:")
    print("=" * 80)
    for i, (filename, row, col) in enumerate(zip(all_filenames[:20], rows[:20], cols[:20])):  # Show first 20
        # Extract FOV from filename
        m = DEFAULT_FOV_RE.search(filename)
        fov_num = "UNKNOWN"
        if m:
            try:
                if 'fov' in m.groupdict():
                    fov_num = m.group('fov')
                else:
                    fov_num = m.group(1)
            except:
                pass
        
        # Get stage coordinates
        coord_idx = filename_to_index.get(filename, "NOT_FOUND")
        if coord_idx != "NOT_FOUND" and coord_idx in region_coords.index:
            stage_x = region_coords.loc[coord_idx, 'x (mm)']
            stage_y = region_coords.loc[coord_idx, 'y (mm)']
            print(f"Index {i:2d}: FOV {fov_num:3s} -> Grid(row={row:2d}, col={col:2d}) | Stage({stage_x:7.3f}, {stage_y:7.3f}) | {filename}")
        else:
            print(f"Index {i:2d}: FOV {fov_num:3s} -> Grid(row={row:2d}, col={col:2d}) | Stage(NOT_FOUND) | {filename}")
    
    if len(all_filenames) > 20:
        print(f"... (showing first 20 of {len(all_filenames)} total)")
    
    # Initialize master grid with complete structure
    master_grid = pd.DataFrame({
        "col": cols,
        "row": rows,
    }, index=np.arange(len(cols)))
    
    # DEBUG: Show grid occupancy matrix
    print("\nDEBUG: Grid Occupancy Matrix:")
    print("=" * 50)
    max_row = max(rows)
    max_col = max(cols)
    print(f"Grid size: {max_row + 1} rows × {max_col + 1} columns")
    
    # Create occupancy matrix
    occupancy = {}
    for i, (row, col) in enumerate(zip(rows, cols)):
        if (row, col) not in occupancy:
            occupancy[(row, col)] = []
        occupancy[(row, col)].append(i)
    
    # Show occupancy conflicts
    conflicts = [(pos, indices) for pos, indices in occupancy.items() if len(indices) > 1]
    if conflicts:
        print(f"WARNING: Found {len(conflicts)} grid position conflicts:")
        for (row, col), indices in conflicts:
            print(f"  Position (row={row}, col={col}) has tiles: {indices}")
    
    # Show grid pattern (first 10x10)
    print("\nDEBUG: Grid Pattern (showing up to 10x10, 'X' = occupied, '.' = empty):")
    display_rows = min(10, max_row + 1)
    display_cols = min(10, max_col + 1)
    for r in range(display_rows):
        row_str = ""
        for c in range(display_cols):
            if (r, c) in occupancy:
                row_str += "X "
            else:
                row_str += ". "
        print(f"Row {r:2d}: {row_str}")
    
    print("DEBUG: First 10 rows of master grid:")
    print(master_grid.head(10))
    
    coord_to_idx = pd.Series(master_grid.index, index=pd.MultiIndex.from_arrays([master_grid['col'], master_grid['row']]))
    top_coords = pd.MultiIndex.from_arrays([master_grid['col'], master_grid['row'] - 1])
    # FIXED: Proper neighbor assignment that preserves NaN for missing neighbors
    top_neighbor_values = coord_to_idx.reindex(top_coords).values
    top_neighbors = []
    for val in top_neighbor_values:
        if pd.isna(val):
            top_neighbors.append(pd.NA)
        else:
            top_neighbors.append(int(val))
    master_grid['top'] = pd.array(top_neighbors, dtype='Int64')  # Int64 supports NA
    left_coords = pd.MultiIndex.from_arrays([master_grid['col'] - 1, master_grid['row']])
    # FIXED: Proper neighbor assignment that preserves NaN for missing neighbors
    left_neighbor_values = coord_to_idx.reindex(left_coords).values
    left_neighbors = []
    for val in left_neighbor_values:
        if pd.isna(val):
            left_neighbors.append(pd.NA)
        else:
            left_neighbors.append(int(val))
    master_grid['left'] = pd.array(left_neighbors, dtype='Int64')  # Int64 supports NA
    
    # DEBUG: Show neighbor assignments
    print("DEBUG: Neighbor assignments (first 20 rows):")
    neighbor_debug = master_grid[['col', 'row', 'left', 'top']].head(20)
    print(neighbor_debug)
    

    
    # DEBUG: Show sample tiles for verification
    print("DEBUG: Sample tile details:")
    grid_size = len(master_grid)
    sample_indices = [0, min(10, grid_size-1), min(20, grid_size-1), min(50, grid_size-1)]
    for idx in sample_indices:
        if idx < grid_size:
            print(f"DEBUG: Tile {idx}: {dict(master_grid.loc[idx])}")
    
    # Initialize translation columns
    for direction in ["left", "top"]:
        for key in ["ncc", "y", "x"]:
            master_grid[f"{direction}_{key}_first"] = np.nan
            master_grid[f"{direction}_{key}_second"] = np.nan
    
    # Create neighborhoods and group into memory-constrained batches
    neighborhoods = create_neighborhoods(master_grid)
    
    # Calculate memory constraints
    first_image = tifffile.imread(str(selected_tiff_paths[0]))
    max_memory = calculate_safe_batch_size(selected_tiff_paths[0]) * first_image.nbytes * 4
    del first_image
    gc.collect()
    
    neighborhood_batches = group_neighborhoods_into_batches(neighborhoods, selected_tiff_paths, max_memory)
    
    print(f"DEBUG: Created {len(neighborhoods)} neighborhoods in {len(neighborhood_batches)} batches")
    
    # Process each neighborhood batch
    for batch_idx, neighborhood_batch in enumerate(neighborhood_batches):
        print(f"Processing neighborhood batch {batch_idx + 1}/{len(neighborhood_batches)} with {len(neighborhood_batch)} neighborhoods")
        
        # Load all images needed for this batch
        batch_images, needed_indices = load_batch_for_neighborhoods(neighborhood_batch, selected_tiff_paths, all_filenames)
        
        print(f"DEBUG: Loaded {len(batch_images)} unique images for {len(needed_indices)} indices")
        
        # Process all neighborhoods in this batch
        translation_results = process_neighborhood_batch(
            neighborhood_batch, batch_images, needed_indices, selected_tiff_paths, 
            all_filenames, master_grid, edge_width
        )
        
        print(f"DEBUG: Computed translations for {len(translation_results)} neighborhoods")
        
        # Update master grid with results
        for center_idx, directions in translation_results.items():
            for direction, translation_data in directions.items():
                for key, value in translation_data.items():
                    master_grid.loc[center_idx, f"{direction}_{key}_first"] = value
        
        # Force memory cleanup
        del batch_images
        gc.collect()
    
    # Run global optimization on complete grid (no images needed)
    print("Running global optimization...")
    
    # DEBUG: Show master grid state before filtering
    print("DEBUG: Master grid state before filtering:")
    print(f"DEBUG: Grid shape: {master_grid.shape}")
    print(f"DEBUG: Columns: {list(master_grid.columns)}")
    
    # DEBUG: Show translation data summary
    print("DEBUG: Translation data summary:")
    for direction in ["left", "top"]:
        ncc_col = f"{direction}_ncc_first"
        y_col = f"{direction}_y_first"
        x_col = f"{direction}_x_first"
        
        if ncc_col in master_grid.columns:
            valid_count = master_grid[ncc_col].notna().sum()
            nan_count = master_grid[ncc_col].isna().sum()
            print(f"DEBUG: {direction}: {valid_count} valid, {nan_count} NaN")
            
            if valid_count > 0:
                ncc_values = master_grid[ncc_col].dropna()
                y_values = master_grid[y_col].dropna()
                x_values = master_grid[x_col].dropna()
                print(f"DEBUG: {direction} NCC range: {ncc_values.min():.3f} to {ncc_values.max():.3f}")
                print(f"DEBUG: {direction} Y range: {y_values.min():.1f} to {y_values.max():.1f}")
                print(f"DEBUG: {direction} X range: {x_values.min():.1f} to {x_values.max():.1f}")
    

    
    # Continue with existing filtering and optimization logic
    has_top_pairs = np.any(master_grid["top_ncc_first"].dropna() > ncc_threshold)
    has_left_pairs = np.any(master_grid["left_ncc_first"].dropna() > ncc_threshold)
    
    if not has_left_pairs and not has_top_pairs:
        raise ValueError("No good initial pairs found - tiles may not have sufficient overlap")
    
    # Get image dimensions from first batch for overlap calculation
    first_image = tifffile.imread(str(selected_tiff_paths[0]))
    full_sizeY, full_sizeX = first_image.shape[:2]
    del first_image
    gc.collect()
    
    # Compute overlaps and continue with existing logic
    if has_left_pairs:
        left_displacement = compute_image_overlap2(
            master_grid[master_grid["left_ncc_first"].fillna(-1) > ncc_threshold], 
            "left", full_sizeY, full_sizeX
        )
        overlap_left = np.clip(100 - left_displacement[1] * 100, pou, 100 - pou)
    else:
        overlap_left = 50.0
    
    if has_top_pairs:
        top_displacement = compute_image_overlap2(
            master_grid[master_grid["top_ncc_first"].fillna(-1) > ncc_threshold], 
            "top", full_sizeY, full_sizeX
        )
        overlap_top = np.clip(100 - top_displacement[0] * 100, pou, 100 - pou)
    else:
        overlap_top = 50.0
    
    # Apply filtering (existing logic)
    if has_top_pairs:
        master_grid["top_valid1"] = filter_by_overlap_and_correlation(
            master_grid["top_y_first"], master_grid["top_ncc_first"], 
            overlap_top, full_sizeY, pou, ncc_threshold
        )
        master_grid["top_valid2"] = filter_outliers(master_grid["top_y_first"], master_grid["top_valid1"])
    else:
        master_grid["top_valid1"] = pd.Series(False, index=master_grid.index)
        master_grid["top_valid2"] = pd.Series(False, index=master_grid.index)
    
    if has_left_pairs:
        master_grid["left_valid1"] = filter_by_overlap_and_correlation(
            master_grid["left_x_first"], master_grid["left_ncc_first"], 
            overlap_left, full_sizeX, pou, ncc_threshold
        )
        master_grid["left_valid2"] = filter_outliers(master_grid["left_x_first"], master_grid["left_valid1"])
    else:
        master_grid["left_valid1"] = pd.Series(False, index=master_grid.index)
        master_grid["left_valid2"] = pd.Series(False, index=master_grid.index)
    
    # Compute repeatability and apply remaining filters
    rs = []
    for direction, dims_chars, rowcol_group_key in zip(["top", "left"], ["yx", "xy"], ["col", "row"]):
        valid_key = f"{direction}_valid2"
        valid_grid_subset = master_grid[master_grid[valid_key].fillna(False)]
        
        if len(valid_grid_subset) > 0:
            w1s = valid_grid_subset[f"{direction}_{dims_chars[0]}_first"]
            r1 = np.ceil((w1s.max() - w1s.min()) / 2) if len(w1s.dropna()) > 1 else 0
            
            r2_vals = []
            for _, grp in valid_grid_subset.groupby(rowcol_group_key):
                w2_series = grp[f"{direction}_{dims_chars[1]}_first"].dropna()
                if len(w2_series) > 1:
                    r2_vals.append(np.max(w2_series) - np.min(w2_series))
            r2 = np.ceil(np.max(r2_vals) / 2) if r2_vals else 0
            rs.append(max(r1, r2))
        else:
            rs.append(0)
    
    r_repeatability = np.max(rs) if rs else 0
    
    master_grid = filter_by_repeatability(master_grid, r_repeatability, ncc_threshold)
    master_grid = replace_invalid_translations(master_grid)
    
    # DEBUG: Show master grid state before spanning tree computation
    print("DEBUG: Master grid state before spanning tree computation:")
    print(f"DEBUG: Grid shape: {master_grid.shape}")
    print(f"DEBUG: Columns: {list(master_grid.columns)}")
    
    # DEBUG: Count edges and check for NaN values
    edge_count = 0
    nan_edge_count = 0
    for idx, row in master_grid.iterrows():
        for direction in ["left", "top"]:
            if not pd.isna(row[direction]):
                ncc_col = f"{direction}_ncc_first"
                y_col = f"{direction}_y_first"
                x_col = f"{direction}_x_first"
                
                ncc_val = row.get(ncc_col, np.nan)
                y_val = row.get(y_col, np.nan)
                x_val = row.get(x_col, np.nan)
                
                if pd.isna(ncc_val) or pd.isna(y_val) or pd.isna(x_val):
                    nan_edge_count += 1
                
                edge_count += 1
    
    print(f"DEBUG: Total edges to be processed: {edge_count} ({nan_edge_count} with NaN values)")
    
    # Global optimization
    print("DEBUG: Calling compute_maximum_spanning_tree...")
    tree = compute_maximum_spanning_tree(master_grid)
    print("DEBUG: Spanning tree computation successful!")
    print(f"DEBUG: Tree has {len(tree.nodes)} nodes and {len(tree.edges)} edges")
    
    master_grid = compute_final_position(master_grid, tree)
    
    prop_dict = {
        "W": full_sizeY,
        "H": full_sizeX,
        "overlap_left": overlap_left,
        "overlap_top": overlap_top,
        "repeatability": r_repeatability,
        "edge_width_used": edge_width
    }
    
    return master_grid, prop_dict

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
        # TECHNICAL DEBT: This dual-path parsing is a poor pattern - see TODO above parse_filename_fov_info()
        # The function now has inconsistent behavior depending on filename format
        # TODO: Replace with proper FileFormat abstraction
        fov_info = parse_filename_fov_info(fname)
        m = None  # Initialize m
        if fov_info:
            fov = fov_info['fov']
        else:
            # Fall back to provided regex pattern
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
        file_region = None
        if fov_info and 'region' in coords_df.columns:
            file_region = fov_info['region']
        elif m and 'region' in m.groupdict() and 'region' in coords_df.columns:
            file_region = m.group('region')
        
        # Get coordinates for this FOV, filtering by region if available
        try:
            if file_region:
                df_row = coords_df.loc[
                    (coords_df[fov_col_name].astype(int) == fov) & 
                    (coords_df['region'] == file_region)
                ]
            else:
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
            # For multi-page TIFFs, skip z-level refinement and let GUI/parameters control z-selection
            z_level_fname = None
            if fov_info and 'z_level' in fov_info:
                z_level_fname = fov_info['z_level']
            elif m and 'z_level' in m.groupdict():
                try:
                    z_level_fname = int(m.group('z_level'))
                except ValueError:
                    logger.warning(f"Could not parse z_level from filename {fname}.")
            
            if z_level_fname is not None and 'z_level' in coords_df.columns:
                matching_z_rows = df_row[df_row['z_level'].astype(int) == z_level_fname]
                if not matching_z_rows.empty:
                    df_row = matching_z_rows
                    logger.info(f"Refined selection for FOV {fov} to include z_level {z_level_fname}.")
            # For multi-page TIFFs (no z_level in filename), use first entry and let z-slice filtering handle selection
            
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
        fov_info = parse_filename_fov_info(path.name)
        if fov_info and fov_info['region'] == region:
            region_paths.append(path)
    
    if not region_paths:
        print(f"No files found for region '{region}'")
        return images
    
    # Now apply z-slice filtering to region-specific files
    selected_tiff_paths = []
    if z_slice_to_keep is not None:
        fov_to_files_map: Dict[int, List[Tuple[int, Path]]] = {}
        
        for path in region_paths:
            fov_info = parse_filename_fov_info(path.name)
            if fov_info:
                try:
                    fov = fov_info['fov']
                    # For multi-page TIFFs, z_level is not in filename
                    if 'z_level' in fov_info:
                        z_level = fov_info['z_level']
                    else:
                        # Multi-page TIFF: use z_slice_to_keep as the target z-level
                        z_level = z_slice_to_keep if z_slice_to_keep is not None else 0
                    
                    if fov not in fov_to_files_map:
                        fov_to_files_map[fov] = []
                    fov_to_files_map[fov].append((z_level, path))
                except (KeyError, ValueError):
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
    
    For multi-page TIFF files (containing "_stack" in name), extracts the middle z-slice
    to match the default stitching behavior.
    
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
        # Check if this is a multi-page TIFF
        if "_stack" in path.name:
            # Multi-page TIFF: extract middle z-slice to match stitching behavior
            with tifffile.TiffFile(path) as tif:
                num_pages = len(tif.pages)
                if num_pages > 1:
                    # Use middle z-slice (same logic as stitcher's middle_layer strategy)
                    middle_z = num_pages // 2
                    image = tif.pages[middle_z].asarray()
                    logger.debug(f"Loaded middle z-slice {middle_z} from {path.name} ({num_pages} total slices)")
                    return path.name, image
                else:
                    # Single page, load normally
                    return path.name, tif.pages[0].asarray()
        else:
            # Regular TIFF file
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
        
        # Get image paths for this region only
        # Look in the 0 subdirectory for TIFF files
        tiff_dir = image_directory / "0"
        if tiff_dir.exists():
            selected_tiff_paths = list(tiff_dir.glob(channel_pattern))
        else:
            selected_tiff_paths = list(image_directory.glob(channel_pattern))
        
        # Filter for the specific region
        region_paths = []
        for path in selected_tiff_paths:
            fov_info = parse_filename_fov_info(path.name)
            if fov_info and fov_info['region'] == region:
                region_paths.append(path)
        
        if not region_paths:
            print(f"No images found for region {region}, skipping")
            continue
        
        # Apply z-slice filtering to region-specific files
        if z_slice_for_registration is not None:
            fov_to_files_map: Dict[int, List[Tuple[int, Path]]] = {}
            
            for path in region_paths:
                fov_info = parse_filename_fov_info(path.name)
                if fov_info:
                    try:
                        fov = fov_info['fov']
                        # For multi-page TIFFs, z_level is not in filename - use the file as-is
                        # The z-slice selection will happen during image loading
                        if 'z_level' in fov_info:
                            z_level = fov_info['z_level']
                        else:
                            # Multi-page TIFF: use z_slice_for_registration as the target z-level
                            z_level = z_slice_for_registration
                        
                        if fov not in fov_to_files_map:
                            fov_to_files_map[fov] = []
                        fov_to_files_map[fov].append((z_level, path))
                    except (KeyError, ValueError):
                        logger.warning(f"Could not parse FOV/Z from {path.name}")
            
            selected_tiff_paths = []
            for fov, z_files in fov_to_files_map.items():
                if not z_files:
                    continue
                
                # Find the file matching z_slice_for_registration
                target_slice_found = False
                for z_level, path in z_files:
                    if z_level == z_slice_for_registration:
                        selected_tiff_paths.append(path)
                        target_slice_found = True
                        break
                
                if not target_slice_found:
                    # Fallback: use the lowest z_level
                    z_files.sort(key=lambda x: x[0])
                    fallback_path = z_files[0][1]
                    selected_tiff_paths.append(fallback_path)
                    logger.warning(f"Z-slice {z_slice_for_registration} not found for FOV {fov}. Using {fallback_path.name} instead.")
        else:
            selected_tiff_paths = region_paths
        
        if not selected_tiff_paths:
            print(f"No files selected for region {region} after z-slice filtering")
            continue
        
        # Extract tile indices from filenames
        filenames = [p.name for p in selected_tiff_paths]
        rows, cols, filename_to_index = extract_tile_indices(filenames, region_coords)
        
        try:
            # Register tiles using batched processing
            grid, prop_dict = register_tiles_batched(
                selected_tiff_paths=selected_tiff_paths,
                region_coords=region_coords,
                edge_width=edge_width,
                overlap_diff_threshold=overlap_diff_threshold,
                pou=pou,
                ncc_threshold=ncc_threshold,
                z_slice_to_keep=z_slice_for_registration
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
