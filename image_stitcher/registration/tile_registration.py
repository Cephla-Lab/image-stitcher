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
from enum import Enum
from pathlib import Path
import re
import logging
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import json
import gc
import psutil
from typing import Dict, List, Optional, Tuple, Union, Pattern, Set
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
from ._stage_model import filter_translations, FilterConfig
from ._translation_computation import compute_translation, TranslationStrategy
from ._translation_computation import multi_peak_max
from ._translation_computation import pcm
from ._translation_computation import OptimizationConfig, SubpixelConfig, RANSACConfig
from ._tensor_backend import TensorBackend, create_tensor_backend

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

# Global tensor backend instance
_tensor_backend: Optional[TensorBackend] = None

def get_tensor_backend() -> TensorBackend:
    """Get or create the global tensor backend instance."""
    global _tensor_backend
    if _tensor_backend is None:
        _tensor_backend = create_tensor_backend()
    return _tensor_backend

def set_tensor_backend(engine: Optional[str] = None) -> TensorBackend:
    """Set the global tensor backend to use a specific engine.
    
    Parameters
    ----------
    engine : Optional[str]
        Preferred engine ('cupy', 'torch', 'numpy'), None for auto
        
    Returns
    -------
    TensorBackend
        The created backend instance
    """
    global _tensor_backend
    _tensor_backend = create_tensor_backend(engine)
    return _tensor_backend

# Constants for file handling
DEFAULT_FOV_RE = re.compile(r"(?P<region>\w+)_(?P<fov>0|[1-9]\d*)_(?P<z_level>\d+)_", re.I)
# Additional regex for multi-page TIFF files with "_stack" suffix
MULTIPAGE_FOV_RE = re.compile(r"(?P<region>\w+)_(?P<fov>0|[1-9]\d*)_stack", re.I)
# Additional regex for OME-TIFF files (region_fov.ome.tif)
OME_TIFF_FOV_RE = re.compile(r"(?P<region>\w+)_(?P<fov>0|[1-9]\d*)\.ome\.tiff?$", re.I)
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
    """TEMPORARY HELPER: Parse FOV info from filename, handling regular, multi-page, and OME-TIFF formats.
    
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
    # Try OME-TIFF pattern first (most specific)
    m = OME_TIFF_FOV_RE.search(filename)
    if m:
        return {
            'region': m.group('region'),
            'fov': int(m.group('fov'))
            # No z_level in OME-TIFF filenames - it's stored inside the file
        }
    
    # Try regular TIFF pattern
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


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class RegistrationMode(Enum):
    """Mode for tile registration operations."""
    SINGLE = "single"
    AUTO = "auto"
    MULTI_TIMEPOINT = "multi"


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


@dataclass
class RowInfo:
    """Information about a row of tiles.
    
    Attributes:
        center_y: Y-coordinate of the row center in mm
        tile_indices: List of indices for tiles in this row
    """
    center_y: float
    tile_indices: List[int]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
    bytes_per_image = first_image.nbytes * 8  # 8x overhead for float64 operations
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
    all_filenames: List[str],
    flatfield: Optional[np.ndarray] = None,
    fov_to_z_level: Optional[Dict[int, int]] = None,
    channel: int = 0
) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """Load all unique images needed for a batch of neighborhoods."""
    
    # Get all unique image indices needed
    all_needed_indices = set()
    for neighborhood in neighborhood_batch:
        all_needed_indices.update(neighborhood.get_all_indices())
    
    needed_indices_list = sorted(list(all_needed_indices))
    
    # Load all needed images
    needed_paths = [selected_tiff_paths[i] for i in needed_indices_list]
    batch_images = load_batch_images(needed_paths, flatfield=flatfield, fov_to_z_level=fov_to_z_level, channel=channel)
    
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

def select_best_registration_channel(
    selected_tiff_paths: List[Path],
    region_coords: pd.DataFrame,
    edge_width: int = DEFAULT_EDGE_WIDTH,
    num_sample_fovs: int = 3,
    fov_to_z_level: Optional[Dict[int, int]] = None
) -> int:
    """Select the channel with highest edge variance for registration.
    
    Samples random FOVs, loads all available channels, and computes edge variance
    to determine which channel has the most features for reliable registration.
    
    Works with both OME-TIFF files (multiple channels per file) and individual
    TIFF files (one file per channel).
    
    Args:
        selected_tiff_paths: List of image file paths
        region_coords: Coordinates DataFrame
        edge_width: Width of edge strips to check
        num_sample_fovs: Number of random FOVs to sample (default 3)
        fov_to_z_level: Optional z-level mapping for multi-z datasets
        
    Returns:
        Best channel index (0-based) to use for registration
    """
    from ..image_loaders import ChannelSource
    
    # Get unique FOVs and regions
    fov_region_pairs = set()
    for path in selected_tiff_paths:
        fov_info = parse_filename_fov_info(path.name)
        if fov_info and 'fov' in fov_info:
            region = fov_info.get('region', 'Unknown')
            fov = fov_info['fov']
            fov_region_pairs.add((region, fov))
    
    if not fov_region_pairs:
        logger.warning("Could not parse any FOV information from filenames - using default channel 0")
        return 0
    
    # Sample random FOVs
    fov_region_list = list(fov_region_pairs)
    if len(fov_region_list) <= num_sample_fovs:
        sample_pairs = fov_region_list
    else:
        sample_indices = np.random.choice(len(fov_region_list), num_sample_fovs, replace=False)
        sample_pairs = [fov_region_list[i] for i in sample_indices]
    
    logger.info(f"Checking edge variance across channels for {len(sample_pairs)} random FOVs: {sample_pairs}")
    
    # Compute edge variance per channel across sampled FOVs
    channel_variances = {}
    all_channel_names = None
    
    for region, fov in sample_pairs:
        try:
            # Determine z-level for this FOV
            z_level = fov_to_z_level.get(fov) if fov_to_z_level else None
            
            # Discover channels for this FOV (format-agnostic)
            descriptors = ChannelSource.discover(
                files=selected_tiff_paths,
                region=region,
                fov=fov,
                z_level=z_level
            )
            
            if not descriptors:
                logger.warning(f"No channels found for FOV {region}_{fov}")
                continue
            
            # Store channel names from first FOV for display
            if all_channel_names is None:
                all_channel_names = [desc.name for desc in descriptors]
            
            # Compute variance for each channel
            for desc in descriptors:
                img = desc.loader.read_slice(
                    channel=desc.channel_param,
                    z=desc.z_param
                )
                
                # Extract right edge (most common overlap direction)
                current_w = min(edge_width, img.shape[1])
                edge = img[:, -current_w:]
                
                # Compute variance (same as is_edge_featureless)
                downsampled = edge[::16, ::16]
                if downsampled.size > 0:
                    variance = np.var(downsampled.astype(np.float32))
                    
                    if desc.index not in channel_variances:
                        channel_variances[desc.index] = []
                    channel_variances[desc.index].append(variance)
        
        except Exception as e:
            logger.warning(f"Failed to check channels for FOV {region}_{fov}: {e}")
            continue
    
    if not channel_variances:
        logger.warning("Could not compute channel variances - using default channel 0")
        return 0
    
    # Compute mean variance per channel and select best
    channel_mean_variance = {ch: np.mean(vars) for ch, vars in channel_variances.items()}
    best_channel = max(channel_mean_variance.items(), key=lambda x: x[1])[0]
    
    # Log results
    logger.warning(f"Channel variance analysis (mean across {len(sample_pairs)} FOVs):")
    for ch in sorted(channel_mean_variance.keys()):
        status = " ← SELECTED" if ch == best_channel else ""
        ch_name = all_channel_names[ch] if all_channel_names and ch < len(all_channel_names) else f"Channel {ch}"
        logger.warning(f"  {ch_name}: variance={channel_mean_variance[ch]:.2e}{status}")
    
    return best_channel


def is_edge_featureless(edge: np.ndarray, variance_threshold: float = 1e3) -> bool:
    """Check if edge strip is featureless (dust speckles only, no tissue).
    
    Checks only the edge region being registered, not the full tile.
    This prevents false positives where a tile has tissue in center but empty edges.
    
    Args:
        edge: 2D edge strip array (e.g., 2048x195)
        variance_threshold: Variance threshold (default 1e3 - lowered to accept more edges)
        
    Returns:
        True if featureless (skip registration), False otherwise
    """
    # Downsample 16x for speed
    downsampled = edge[::16, ::16]
    if downsampled.size == 0:
        return True  # Empty edge is featureless
    
    variance = np.var(downsampled.astype(np.float32))
    is_featureless = variance < variance_threshold
    
    if is_featureless:
        logger.warning(
            f"Edge rejected as featureless: variance={variance:.2e} < threshold={variance_threshold:.2e}. "
            f"Edge shape={edge.shape}, intensity range=[{edge.min()}, {edge.max()}]. "
            f"Consider using a different channel for registration if multiple channels are available."
        )
    
    return is_featureless

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
        
        # Check if edge strips are featureless (dust only, no tissue)
        if is_edge_featureless(edge1) or is_edge_featureless(edge2):
            return None  # Skip registration, use stage coordinates only
            
        edge_sizeY, edge_sizeX = edge1.shape
        
        # PCM and peak finding on edges
        PCM_on_edges = pcm(edge1, edge2).real
        lims_edge_relative = np.array([[-edge_sizeY, edge_sizeY], [-edge_sizeX, edge_sizeX]])
        
        yins_edge, xins_edge, _ = multi_peak_max(PCM_on_edges)
        
        # Compute translation with RANSAC strategy (includes subpixel + optimization)
        ncc_val, y_best_edge_relative, x_best_edge_relative = compute_translation(
            edge1, edge2, yins_edge, xins_edge, 
            *lims_edge_relative[0], *lims_edge_relative[1],
            strategy=TranslationStrategy.RANSAC,
            n=10,
            ransac_config=RANSACConfig(use_ransac=True, residual_threshold=2.0, min_inlier_ratio=0.3),
            subpixel_config=SubpixelConfig(upsample_factor=100, use_subpixel=True),
            optimization_config=OptimizationConfig(method='trf', max_nfev=50, verbose=0)
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
    
    processing_overhead = 8  # Conservative 8x multiplier for float64 operations
    total_per_image = image_memory * processing_overhead
    
    available_memory = psutil.virtual_memory().available * memory_fraction
    max_batch_size = max(1, int(available_memory // total_per_image))
    
    print(f"Image memory: {image_memory/1e6:.1f}MB, "
          f"Available: {available_memory/1e9:.1f}GB, "
          f"Batch size: {max_batch_size}")
    
    return max_batch_size


def _load_image_wrapper(args: Tuple[Path, Optional[Dict[int, int]], int]) -> Tuple[str, Optional[np.ndarray]]:
    """Load image without flatfield correction (for multiprocessing)."""
    path, fov_to_z_level, channel = args
    return load_single_image(path, flatfield=None, fov_to_z_level=fov_to_z_level, channel=channel)

def load_batch_images(
    batch_paths: List[Path], 
    flatfield: Optional[np.ndarray] = None,
    fov_to_z_level: Optional[Dict[int, int]] = None,
    channel: int = 0
) -> Dict[str, np.ndarray]:
    """Load a batch of images using multiprocessing, apply flatfield serially."""
    n_processes = min(cpu_count(), len(batch_paths))
    
    # Load images in parallel without flatfield
    # Pack fov_to_z_level and channel with each path for multiprocessing
    args_list = [(path, fov_to_z_level, channel) for path in batch_paths]
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(_load_image_wrapper, args_list),
            total=len(batch_paths),
            desc="Loading batch"
        ))
    
    images = {k: v for k, v in results if v is not None}
    
    # Apply flatfield correction serially if provided
    if flatfield is not None:
        for filename, image in images.items():
            images[filename] = apply_flatfield_to_image(image, flatfield, dtype=image.dtype)
    
    return images

def register_tiles_batched(
    selected_tiff_paths: List[Path],
    region_coords: pd.DataFrame,
    edge_width: int = DEFAULT_EDGE_WIDTH,
    overlap_diff_threshold: float = 10,
    pou: float = 3,
    ncc_threshold: float = 0.5,
    z_slice_to_keep: Optional[int] = 0,
    tensor_backend_engine: Optional[str] = None,
    flatfield_corrections: Optional[Dict[int, np.ndarray]] = None,
    registration_channel: int = 0
) -> Tuple[pd.DataFrame, dict]:
    """[DEPRECATED] Use register_tiles(mode=RegistrationMode.SINGLE) instead.
    
    Register tiles using memory-constrained batched processing.
    
    Parameters
    ----------
    selected_tiff_paths : List[Path]
        List of paths to TIFF files to register
    region_coords : pd.DataFrame
        DataFrame containing tile coordinates
    edge_width : int
        Width of edge strips for correlation
    overlap_diff_threshold : float
        Allowed difference from initial guess (percentage)
    pou : float
        Percent overlap uncertainty
    ncc_threshold : float
        Normalized cross correlation threshold
    z_slice_to_keep : Optional[int]
        Z-slice to use for registration
    tensor_backend_engine : Optional[str]
        Preferred tensor backend engine ('cupy', 'torch', 'numpy', None for auto)
    flatfield_corrections : Optional[Dict[int, np.ndarray]]
        Precomputed flatfield corrections indexed by channel
    flatfield_manifest : Optional[Path]
        Path to flatfield manifest file to load corrections
        
    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Registration results and properties dictionary
    """
    
    if not selected_tiff_paths:
        return pd.DataFrame(), {}
    
    # Set tensor backend if specified
    if tensor_backend_engine is not None:
        backend = set_tensor_backend(tensor_backend_engine)
        print(f"Using tensor backend: {backend.name} ({'GPU' if backend.is_gpu else 'CPU'})")
    else:
        backend = get_tensor_backend()
        print(f"Using tensor backend: {backend.name} ({'GPU' if backend.is_gpu else 'CPU'})")
    
    # Calculate safe batch size
    batch_size = calculate_safe_batch_size(selected_tiff_paths[0])
    
    # DEBUG: Force small batch size for testing
    batch_size = min(batch_size, 20)
    print(f"DEBUG: Using batch size: {batch_size}")
    
    # Create complete grid structure BEFORE batching
    all_filenames = [p.name for p in selected_tiff_paths]
    rows, cols, filename_to_index, fov_to_z_level = extract_tile_indices(all_filenames, region_coords)
    
    # DEBUG: Show grid structure
    print(f"DEBUG: Grid dimensions: {len(rows)} tiles, {len(set(rows))} unique rows, {len(set(cols))} unique columns")
    print(f"DEBUG: Row range: {min(rows)} to {max(rows)}")
    print(f"DEBUG: Col range: {min(cols)} to {max(cols)}")
    
    # DEBUG: Show detailed FOV to grid position mapping
    print("\nDEBUG: FOV to Grid Position Mapping:")
    print("=" * 80)
    for i, (filename, row, col) in enumerate(zip(all_filenames[:20], rows[:20], cols[:20])):  # Show first 20
        # Extract FOV from filename using the proper parser
        fov_info = parse_filename_fov_info(filename)
        fov_num = "UNKNOWN"
        if fov_info and 'fov' in fov_info:
            fov_num = str(fov_info['fov'])
        
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
    
    # Initialize master grid with complete structure including stage coordinates
    # Use existing filename_to_index mapping to get coordinates
    stage_x_list = []
    stage_y_list = []
    
    for filename in all_filenames:
        coord_idx = filename_to_index.get(filename)
        if coord_idx is not None and coord_idx in region_coords.index:
            stage_x_list.append(float(region_coords.loc[coord_idx, 'x (mm)']))
            stage_y_list.append(float(region_coords.loc[coord_idx, 'y (mm)']))
        else:
            # Fallback to 0,0 if no match (shouldn't happen)
            stage_x_list.append(0.0)
            stage_y_list.append(0.0)
    
    master_grid = pd.DataFrame({
        "col": cols,
        "row": rows,
        "stage_x": stage_x_list,
        "stage_y": stage_y_list,
        "filename": all_filenames,
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
    
    if not neighborhoods:
        raise ValueError(
            f"No valid neighborhoods found. Grid has {len(master_grid)} tiles but no valid "
            f"neighbor relationships. Check that tiles have proper row/col assignments."
        )
    
    # Auto-select best registration channel for OME-TIFF (channels inside file)
    # For individual files, channel selection happens upstream before this function
    is_ome_tiff = any(p.name.lower().endswith(('.ome.tif', '.ome.tiff')) for p in selected_tiff_paths)
    
    if registration_channel == 0 and is_ome_tiff:
        best_channel = select_best_registration_channel(
            selected_tiff_paths, 
            region_coords,
            edge_width=edge_width,
            num_sample_fovs=3,
            fov_to_z_level=fov_to_z_level
        )
        if best_channel != 0:
            logger.warning(f"Auto-selected channel {best_channel} for registration")
            registration_channel = best_channel
    
    # Load first image once for both memory estimation and flatfield validation
    _, first_image = load_single_image(selected_tiff_paths[0], flatfield=None, fov_to_z_level=fov_to_z_level)
    
    # Get flatfield for registration (use specified channel)
    flatfield_for_registration = None
    if flatfield_corrections:
        # Try specified registration channel first
        flatfield_for_registration = flatfield_corrections.get(registration_channel, None)
        
        if flatfield_for_registration is None:
            available_channels = list(flatfield_corrections.keys())
            
            # If non-default channel explicitly requested but not found, raise error
            if registration_channel != 0 and available_channels:
                raise ValueError(
                    f"Registration channel {registration_channel} not found in flatfield_corrections. "
                    f"Available channels: {available_channels}. "
                    f"Please specify a valid registration_channel parameter."
                )
            
            # For default channel (0), fall back to first available
            if available_channels:
                fallback_channel = available_channels[0]
                flatfield_for_registration = flatfield_corrections[fallback_channel]
                logging.warning(
                    f"Default registration channel {registration_channel} not found. "
                    f"Using channel {fallback_channel} instead. Available channels: {available_channels}"
                )
        
        # Validate flatfield shape matches image dimensions before processing
        if flatfield_for_registration is not None:
            if flatfield_for_registration.shape != first_image.shape:
                raise ValueError(
                    f"Flatfield shape {flatfield_for_registration.shape} does not match "
                    f"image shape {first_image.shape}. Cannot apply flatfield correction."
                )
            logging.info(
                f"Using flatfield correction for registration from channel {registration_channel} "
                f"(shape: {flatfield_for_registration.shape})"
            )
    
    # Calculate memory constraints using the already-loaded first_image
    max_memory = calculate_safe_batch_size(selected_tiff_paths[0]) * first_image.nbytes * 8
    
    del first_image
    gc.collect()
    
    neighborhood_batches = group_neighborhoods_into_batches(neighborhoods, selected_tiff_paths, max_memory)
    
    print(f"DEBUG: Created {len(neighborhoods)} neighborhoods in {len(neighborhood_batches)} batches")
    
    # Process each neighborhood batch
    for batch_idx, neighborhood_batch in enumerate(neighborhood_batches):
        print(f"Processing neighborhood batch {batch_idx + 1}/{len(neighborhood_batches)} with {len(neighborhood_batch)} neighborhoods")
        
        # Load all images needed for this batch
        batch_images, needed_indices = load_batch_for_neighborhoods(
            neighborhood_batch, selected_tiff_paths, all_filenames,
            flatfield=flatfield_for_registration,
            fov_to_z_level=fov_to_z_level,
            channel=registration_channel
        )
        
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
    
    # Get standardized image dimensions (always returns Y, X regardless of format)
    from ..image_loaders import get_image_dimensions
    full_sizeY, full_sizeX = get_image_dimensions(selected_tiff_paths[0])
    print(f"DEBUG: Image dimensions: Y={full_sizeY}, X={full_sizeX}")
    
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
    
    # Compute repeatability (needed for filter config)
    rs = []
    for direction, dims_chars, rowcol_group_key in zip(["top", "left"], ["yx", "xy"], ["col", "row"]):
        # Calculate valid translations for repeatability computation
        if direction == "top" and has_top_pairs:
            temp_valid = (
                master_grid["top_y_first"].between(
                    full_sizeY * (100 - overlap_top - pou) / 100,
                    full_sizeY * (100 - overlap_top + pou) / 100
                ) & (master_grid["top_ncc_first"] > ncc_threshold)
            )
            valid_grid_subset = master_grid[temp_valid.fillna(False)]
        elif direction == "left" and has_left_pairs:
            temp_valid = (
                master_grid["left_x_first"].between(
                    full_sizeX * (100 - overlap_left - pou) / 100,
                    full_sizeX * (100 - overlap_left + pou) / 100
                ) & (master_grid["left_ncc_first"] > ncc_threshold)
            )
            valid_grid_subset = master_grid[temp_valid.fillna(False)]
        else:
            valid_grid_subset = pd.DataFrame()
        
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
    
    # Apply unified filtering pipeline
    filter_config = FilterConfig(
        overlap_left=overlap_left,
        overlap_top=overlap_top,
        size_x=full_sizeX,
        size_y=full_sizeY,
        pou=pou,
        ncc_threshold=ncc_threshold,
        iqr_multiplier=1.5,
        repeatability=r_repeatability
    )
    master_grid = filter_translations(master_grid, filter_config)
    
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
    
    # COMPREHENSIVE DEBUG OUTPUT
    print(f"\n=== REGISTRATION DEBUG REPORT ===")
    print(f"Parameters used:")
    print(f"  - edge_width: {edge_width}")
    print(f"  - ncc_threshold: {ncc_threshold}")
    print(f"  - pou: {pou}")
    print(f"  - overlap_diff_threshold: {overlap_diff_threshold}")
    
    # Analyze NCC distributions for each direction
    for direction in ["left", "top"]:
        ncc_col = f"{direction}_ncc_first"
        if ncc_col in master_grid.columns:
            ncc_values = master_grid[ncc_col].dropna()
            if len(ncc_values) > 0:
                print(f"\n{direction.upper()} direction NCC statistics:")
                print(f"  - Count: {len(ncc_values)}")
                print(f"  - Mean: {ncc_values.mean():.4f}")
                print(f"  - Std: {ncc_values.std():.4f}")
                print(f"  - Min: {ncc_values.min():.4f}")
                print(f"  - Max: {ncc_values.max():.4f}")
                print(f"  - Above threshold ({ncc_threshold}): {(ncc_values > ncc_threshold).sum()}")
                
                # Show distribution in bins
                bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
                hist, _ = np.histogram(ncc_values, bins=bins)
                print(f"  - NCC distribution:")
                for i in range(len(bins)-1):
                    print(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]} tiles")
    
    # Show edge width impact
    print(f"\nEdge width analysis:")
    print(f"  - edge_width = {edge_width}")
    print(f"  - Image dimensions: {full_sizeY} x {full_sizeX}")
    print(f"  - Actual edge width used (left): {min(edge_width, full_sizeX)} ({100*min(edge_width, full_sizeX)/full_sizeX:.1f}% of image width)")
    print(f"  - Actual edge width used (top): {min(edge_width, full_sizeY)} ({100*min(edge_width, full_sizeY)/full_sizeY:.1f}% of image height)")
    
    if edge_width >= full_sizeX * 0.5:
        print(f"  WARNING: edge_width ({edge_width}) is >=50% of image width ({full_sizeX})")
        print(f"     This may cause unstable registration as overlapping regions become too large!")
    if edge_width >= full_sizeY * 0.5:
        print(f"  WARNING: edge_width ({edge_width}) is >=50% of image height ({full_sizeY})")
        print(f"     This may cause unstable registration as overlapping regions become too large!")
    
    tree = compute_maximum_spanning_tree(master_grid)
    
    # Analyze spanning tree edges
    print(f"\nSpanning tree analysis:")
    print(f"  - Total edges in tree: {len(tree.edges)}")
    
    edge_weights = []
    edge_directions = []
    for u, v, data in tree.edges(data=True):
        edge_weights.append(data['weight'])
        edge_directions.append(data['direction'])
    
    if edge_weights:
        edge_weights = np.array(edge_weights)
        print(f"  - Edge weights (NCC + bonus): mean={edge_weights.mean():.4f}, min={edge_weights.min():.4f}, max={edge_weights.max():.4f}")
        print(f"  - Directions in tree: left={edge_directions.count('left')}, top={edge_directions.count('top')}")
        
        # Show which tiles are connected
        print(f"  - Connected tile pairs:")
        for u, v, data in tree.edges(data=True):
            weight = data['weight']
            direction = data['direction']
            # Extract the actual NCC (subtract bonus if present)
            actual_ncc = weight - 10 if weight > 10 else weight
            print(f"    Tile {u} -> {v} ({direction}): NCC={actual_ncc:.4f}, weight={weight:.4f}")
    
    print(f"=== END DEBUG REPORT ===\n")
    print("DEBUG: Spanning tree computation successful!")
    print(f"DEBUG: Tree has {len(tree.nodes)} nodes and {len(tree.edges)} edges")
    
    # Calculate pixel size from stage coordinates for isolated tile positioning
    pixel_size_um = None
    try:
        # Quick estimate from first adjacent pair
        for idx in range(len(master_grid) - 1):
            left_idx = master_grid.loc[idx, 'left']
            if pd.notna(left_idx):
                dx_mm = abs(master_grid.loc[idx, 'stage_x'] - master_grid.loc[int(left_idx), 'stage_x'])
                if dx_mm > 0:
                    # Estimate from stage spacing (will be refined later)
                    pixel_size_um = dx_mm * 1000.0 / 2084.0  # rough: mm * 1000 / image_width
                    break
    except:
        pass
    
    master_grid = compute_final_position(master_grid, tree, pixel_size_um=pixel_size_um)
    
    # ADDITIONAL DEBUG: Show final position changes
    print(f"\nFinal position analysis:")
    if 'x_pos' in master_grid.columns and 'y_pos' in master_grid.columns:
        # Compare with original stage coordinates if available
        if 'stage_x' in master_grid.columns and 'stage_y' in master_grid.columns:
            x_shifts = master_grid['x_pos'] - master_grid['stage_x']
            y_shifts = master_grid['y_pos'] - master_grid['stage_y']
            
            print(f"  - Position shifts from original stage coordinates:")
            print(f"    X shifts: mean={x_shifts.mean():.1f}, std={x_shifts.std():.1f}, range=[{x_shifts.min():.1f}, {x_shifts.max():.1f}]")
            print(f"    Y shifts: mean={y_shifts.mean():.1f}, std={y_shifts.std():.1f}, range=[{y_shifts.min():.1f}, {y_shifts.max():.1f}]")
            
            # Flag extreme movements
            large_x_shifts = np.abs(x_shifts) > 1000  # More than 1000 pixels
            large_y_shifts = np.abs(y_shifts) > 1000
            if large_x_shifts.any() or large_y_shifts.any():
                print(f"  WARNING: Large position shifts detected!")
                print(f"    Tiles with large X shifts: {master_grid.index[large_x_shifts].tolist()}")
                print(f"    Tiles with large Y shifts: {master_grid.index[large_y_shifts].tolist()}")
        
        # Show final positions
        print(f"  - Final positions:")
        print(f"    X range: [{master_grid['x_pos'].min():.1f}, {master_grid['x_pos'].max():.1f}]")
        print(f"    Y range: [{master_grid['y_pos'].min():.1f}, {master_grid['y_pos'].max():.1f}]")
    
    print(f"=== REGISTRATION COMPLETE ===\n")
    
    prop_dict = {
        "W": full_sizeY,
        "H": full_sizeX,
        "overlap_left": overlap_left,
        "overlap_top": overlap_top,
        "repeatability": r_repeatability,
        "edge_width_used": edge_width
    }
    
    return master_grid, prop_dict


def register_tiles(
    path: Union[str, Path],
    mode: RegistrationMode = RegistrationMode.AUTO,
    csv_path: Optional[Union[str, Path]] = None,
    output_csv_path: Optional[Union[str, Path]] = None,
    channel_pattern: Optional[str] = None,
    overlap_diff_threshold: float = 10,
    pou: float = 3,
    ncc_threshold: float = 0.5,
    edge_width: int = DEFAULT_EDGE_WIDTH,
    z_slice_to_keep: Optional[int] = 0,
    tensor_backend_engine: Optional[str] = None,
    flatfield_corrections: Optional[Dict[int, np.ndarray]] = None,
    registration_channel: int = 0,
    skip_backup: bool = False
) -> Union[Tuple[pd.DataFrame, dict], pd.DataFrame, Dict[int, pd.DataFrame]]:
    """Unified entry point for tile registration.
    
    Args:
        path: Path to directory or files
        mode: Registration mode (SINGLE, AUTO, MULTI_TIMEPOINT)
        csv_path: Path to coordinates CSV (for SINGLE/AUTO modes)
        output_csv_path: Path to save output (for AUTO mode)
        channel_pattern: Pattern to match image files
        overlap_diff_threshold: Allowed difference from initial guess (%)
        pou: Percent overlap uncertainty
        ncc_threshold: Normalized cross correlation threshold
        edge_width: Width of edge strips for correlation
        z_slice_to_keep: Z-slice to use for registration
        tensor_backend_engine: Preferred backend ('cupy', 'torch', 'numpy')
        flatfield_corrections: Precomputed flatfield corrections
        registration_channel: Channel to use for registration
        skip_backup: Skip creating backup of coordinates file
        
    Returns:
        Depends on mode:
        - SINGLE: Tuple[pd.DataFrame, dict] (results, properties)
        - AUTO: pd.DataFrame (updated coordinates)
        - MULTI_TIMEPOINT: Dict[int, pd.DataFrame] (timepoint -> coordinates)
    """
    path = Path(path)
    
    if mode == RegistrationMode.SINGLE:
        # Direct batched processing of provided files
        if csv_path is None:
            raise ValueError("csv_path required for SINGLE mode")
        
        coords_df = read_coordinates_csv(csv_path)
        
        # Get image files
        if path.is_dir():
            if channel_pattern is None:
                channel_pattern = select_channel_pattern(path)
            all_image_paths = sorted(path.glob(channel_pattern))
        else:
            all_image_paths = [path]
        
        # Auto-select best channel if registration_channel is default (0)
        if registration_channel == 0 and len(all_image_paths) > 0:
            # Check format to determine if channel selection is needed
            is_ome_tiff = any(p.name.lower().endswith(('.ome.tif', '.ome.tiff')) for p in all_image_paths)
            
            if not is_ome_tiff:
                # Individual files: Need to filter to best channel
                from ..image_loaders import ChannelSource
                
                # Try to select best channel
                try:
                    best_channel = select_best_registration_channel(
                        all_image_paths,
                        coords_df,
                        edge_width=edge_width,
                        num_sample_fovs=3,
                        fov_to_z_level=None
                    )
                    
                    # Get sample FOV to discover channels
                    sample_fov_info = parse_filename_fov_info(all_image_paths[0].name)
                    if sample_fov_info:
                        region = sample_fov_info.get('region', 'Unknown')
                        fov = sample_fov_info['fov']
                        z_level = sample_fov_info.get('z_level')
                        
                        # Discover channels
                        descriptors = ChannelSource.discover(
                            files=all_image_paths,
                            region=region,
                            fov=fov,
                            z_level=z_level
                        )
                        
                        if descriptors and best_channel < len(descriptors):
                            # Get the channel name for the selected channel
                            selected_channel_name = descriptors[best_channel].name
                            
                            # Filter files to only those matching the selected channel
                            filtered_paths = []
                            for p in all_image_paths:
                                channel_name = ChannelSource._extract_channel_name(p.name)
                                if channel_name == selected_channel_name:
                                    filtered_paths.append(p)
                            
                            if filtered_paths:
                                all_image_paths = filtered_paths
                                logger.info(f"Filtered to {len(all_image_paths)} files for channel '{selected_channel_name}'")
                                # Individual files use channel 0
                                registration_channel = 0
                except Exception as e:
                    logger.warning(f"Channel selection failed: {e}. Using all files.")
            else:
                # OME-TIFF: Channel selection happens inside register_tiles_batched
                pass
        
        return register_tiles_batched(
            selected_tiff_paths=all_image_paths,
            region_coords=coords_df,
            edge_width=edge_width,
            overlap_diff_threshold=overlap_diff_threshold,
            pou=pou,
            ncc_threshold=ncc_threshold,
            z_slice_to_keep=z_slice_to_keep,
            tensor_backend_engine=tensor_backend_engine,
            flatfield_corrections=flatfield_corrections,
            registration_channel=registration_channel
        )
    
    elif mode == RegistrationMode.AUTO:
        # Auto-detect and update coordinates
        if csv_path is None or output_csv_path is None:
            raise ValueError("csv_path and output_csv_path required for AUTO mode")
        
        return register_and_update_coordinates(
            image_directory=path,
            csv_path=csv_path,
            output_csv_path=output_csv_path,
            channel_pattern=channel_pattern,
            overlap_diff_threshold=overlap_diff_threshold,
            pou=pou,
            ncc_threshold=ncc_threshold,
            skip_backup=skip_backup,
            z_slice_for_registration=z_slice_to_keep,
            edge_width=edge_width,
            tensor_backend_engine=tensor_backend_engine,
            flatfield_corrections=flatfield_corrections
        )
    
    elif mode == RegistrationMode.MULTI_TIMEPOINT:
        # Process multiple timepoints
        return process_multiple_timepoints(
            base_directory=path,
            overlap_diff_threshold=overlap_diff_threshold,
            pou=pou,
            ncc_threshold=ncc_threshold,
            edge_width=edge_width,
            tensor_backend_engine=tensor_backend_engine,
            flatfield_corrections=flatfield_corrections
        )
    
    else:
        raise ValueError(f"Unknown registration mode: {mode}")


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
) -> Tuple[List[int], List[int], Dict[str, int], Dict[int, int]]:
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
        A tuple: (row_assignments, col_assignments, fname_to_dfidx_map, fov_to_z_level)
        fov_to_z_level is a dict mapping FOV number to z-level index for OME-TIFF files
        - row_assignments: List of 0-indexed row numbers for each filename.
        - col_assignments: List of 0-indexed column numbers for each filename.
        - fname_to_dfidx_map: Dictionary mapping filename to its original index in coords_df.
    """
    if not filenames:
        logger.info("Received empty filenames list, returning empty results.")
        return [], [], {}, {}

    # --- Validate DataFrame columns ---
    required_cols = [fov_col_name, x_col_name, y_col_name]
    for col in required_cols:
        if col not in coords_df.columns:
            msg = f"Required column '{col}' not found in coords_df."
            logger.error(msg)
            raise KeyError(msg)

    # 1.  Collect (x, y) per filename
    # 0. PRE-CALCULATE z-level mapping for OME-TIFFs (before coordinate extraction)
    fov_to_z_level: Dict[int, int] = {}
    if 'z (um)' in coords_df.columns and 'z_level' in coords_df.columns:
        is_ome_tiff = any(fn.lower().endswith('.ome.tiff') or fn.lower().endswith('.ome.tif') 
                         for fn in filenames)
        if is_ome_tiff:
            # Get all FOVs from filenames
            fovs_in_use = set()
            for fname in filenames:
                fov_info = parse_filename_fov_info(fname)
                if fov_info and 'fov' in fov_info:
                    fovs_in_use.add(fov_info['fov'])
            
            # Force selection of middle z-layer by index for all FOVs
            print("Using index-based middle z-layer selection for registration")
            logger.info("Using index-based middle z-layer selection for registration")
            for fov in sorted(fovs_in_use):
                fov_rows = coords_df[coords_df[fov_col_name].astype(int) == fov].sort_values('z_level')
                if len(fov_rows) > 0:
                    middle_idx = len(fov_rows) // 2
                    z_level_to_use = fov_rows.iloc[middle_idx]['z_level']
                    z_position = fov_rows.iloc[middle_idx]['z (um)']
                    fov_to_z_level[fov] = int(z_level_to_use)
                    print(f"  → FOV {fov}: using middle z_level={z_level_to_use} (index {middle_idx}/{len(fov_rows)}, z={z_position:.2f}µm)")
                    logger.info(f"  → FOV {fov}: using middle z_level={z_level_to_use} (index {middle_idx}/{len(fov_rows)}, z={z_position:.2f}µm)")

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
            # Try to find an entry that matches the pre-calculated z-level (for OME-TIFFs)
            z_level_to_use = None
            if fov in fov_to_z_level:
                z_level_to_use = fov_to_z_level[fov]
                logger.debug(f"FOV {fov}: using pre-calculated z_level={z_level_to_use} for coordinate extraction")
            # Otherwise try z-level from filename (for multi-page TIFFs)
            elif fov_info and 'z_level' in fov_info:
                z_level_to_use = fov_info['z_level']
            elif m and 'z_level' in m.groupdict():
                try:
                    z_level_to_use = int(m.group('z_level'))
                except ValueError:
                    logger.warning(f"Could not parse z_level from filename {fname}.")
            
            if z_level_to_use is not None and 'z_level' in coords_df.columns:
                matching_z_rows = df_row[df_row['z_level'].astype(int) == z_level_to_use]
                if not matching_z_rows.empty:
                    df_row = matching_z_rows
                else:
                    logger.warning(f"FOV {fov}: z_level {z_level_to_use} not found in coords, using first entry")
            
            if len(df_row) > 1:
                logger.warning(f"FOV {fov}: Still multiple entries after z-level filtering. Using first.")

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
        return [-1] * num_original_files, [-1] * num_original_files, fname_to_dfidx_map, {}

    # fov_to_z_level already calculated at line 1225 (before coordinate extraction)
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
    
    return final_row_assignments.tolist(), final_col_assignments.tolist(), fname_to_dfidx_map, fov_to_z_level




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


def update_coordinates(
    grid: pd.DataFrame,
    coords_df: pd.DataFrame,
    filename_to_index: Dict[str, int],
    filenames: List[str],
    pixel_size_um: float,
    reference_idx: int = 0,
    full_coords_df: Optional[pd.DataFrame] = None,
    region: Optional[str] = None
) -> pd.DataFrame:
    """Unified coordinate update function for single and multi-z workflows.
    
    Replaces update_stage_coordinates and update_stage_coordinates_multi_z.
    
    Parameters
    ----------
    grid : pd.DataFrame
        Registration results with pixel positions
    coords_df : pd.DataFrame
        Original coordinates to update (region-specific or full)
    filename_to_index : Dict[str, int]
        Mapping from filename to coordinate index
    filenames : List[str]
        Ordered list of filenames
    pixel_size_um : float
        Pixel size in microns
    reference_idx : int
        Index of reference tile (default 0)
    full_coords_df : Optional[pd.DataFrame]
        Full coordinates DataFrame for multi-region processing (optional)
    region : Optional[str]
        Current region being processed (optional, for multi-region)
        
    Returns
    -------
    updated_coords : pd.DataFrame
        Updated coordinates with new stage positions
    """
    # Multi-region mode: work with full DataFrame
    if full_coords_df is not None and region is not None:
        return _update_coords_multi_region(
            grid, coords_df, filename_to_index, filenames, 
            pixel_size_um, full_coords_df, region, reference_idx
        )
    
    # Single region mode: work with provided DataFrame
    return _update_coords_single_region(
        grid, coords_df, filename_to_index, filenames,
        pixel_size_um, reference_idx
    )


def _update_coords_single_region(
    grid: pd.DataFrame,
    coords_df: pd.DataFrame,
    filename_to_index: Dict[str, int],
    filenames: List[str],
    pixel_size_um: float,
    reference_idx: int
) -> pd.DataFrame:
    """Update coordinates for single region (internal helper).
    
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


def _update_coords_multi_region(
    grid: pd.DataFrame,
    coords_df: pd.DataFrame,
    filename_to_index: Dict[str, int],
    filenames: List[str],
    pixel_size_um: float,
    full_coords_df: pd.DataFrame,
    region: str,
    reference_idx: int
) -> pd.DataFrame:
    """Update coordinates for multi-region processing (internal helper)."""
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
    
    # Process each registered tile
    for idx, row in grid.iterrows():
        filename = filenames[idx]
        coord_idx = filename_to_index[filename]
        fov = coords_df.loc[coord_idx, fov_col_name]
        
        # Calculate position update
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


def update_stage_coordinates(
    grid: pd.DataFrame,
    coords_df: pd.DataFrame,
    filename_to_index: Dict[str, int],
    filenames: List[str],
    pixel_size_um: float,
    reference_idx: int = 0
) -> pd.DataFrame:
    """[DEPRECATED] Use update_coordinates() instead.
    
    Legacy function for backward compatibility.
    """
    return update_coordinates(
        grid, coords_df, filename_to_index, filenames,
        pixel_size_um, reference_idx
    )


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
    
    Checks both the given directory and parent's ome_tiff/ subdirectory.
    
    Parameters
    ----------
    directory : Union[str, Path]
        Directory containing TIFF images (typically a timepoint directory like '0/')
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
    
    # Check for new OME-TIFF structure first
    parent_ome_dir = directory.parent / "ome_tiff"
    if parent_ome_dir.exists() and parent_ome_dir.is_dir():
        search_dir = parent_ome_dir
        logger.info(f"Reading images from ome_tiff/ directory: {search_dir}")
    else:
        search_dir = directory
    
    # Get all matching files
    all_tiff_paths = list(search_dir.glob(pattern))
    if not all_tiff_paths:
        print(f"No files matching pattern '{pattern}' found in {search_dir}")
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
                    
                    # Check if this is an OME-TIFF or multi-page TIFF
                    is_ome_tiff = path.name.lower().endswith('.ome.tif') or path.name.lower().endswith('.ome.tiff')
                    is_multipage = '_stack' in path.name.lower()
                    
                    if is_ome_tiff or is_multipage:
                        # For OME-TIFF and multi-page TIFF, z-selection happens during loading
                        # Just use the target z_slice_to_keep as a placeholder
                        z_level = z_slice_to_keep if z_slice_to_keep is not None else 0
                    elif 'z_level' in fov_info:
                        # Regular TIFF with z_level in filename
                        z_level = fov_info['z_level']
                    else:
                        # Shouldn't happen, but default to 0
                        z_level = 0
                    
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
            pool.imap(_load_image_wrapper, selected_tiff_paths),
            total=len(selected_tiff_paths),
            desc=f"Loading images for region {region}"
        ))
    
    # Filter out failed loads and update dictionary
    images.update({k: v for k, v in results if v is not None})
    
    print(f"Successfully loaded {len(images)}/{len(selected_tiff_paths)} images for region {region}")
    
    return images



def apply_flatfield_to_image(
    image: np.ndarray, 
    flatfield: np.ndarray, 
    dtype: Optional[np.dtype] = None
) -> np.ndarray:
    """Apply flatfield correction to a single image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image to correct
    flatfield : np.ndarray
        Flatfield correction array (same shape as image)
    dtype : Optional[np.dtype]
        Output dtype, defaults to input dtype
        
    Returns
    -------
    np.ndarray
        Flatfield-corrected image
        
    Raises
    ------
    ValueError
        If flatfield shape doesn't match image shape
    """
    if dtype is None:
        dtype = image.dtype
    
    # Validate shapes match
    if image.shape != flatfield.shape:
        raise ValueError(
            f"Flatfield shape {flatfield.shape} doesn't match image shape {image.shape}"
        )
    
    corrected = (image / flatfield).clip(
        min=np.iinfo(dtype).min if np.issubdtype(dtype, np.integer) else 0,
        max=np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1
    ).astype(dtype)
    return corrected


def load_single_image(
    path: Path, 
    flatfield: Optional[np.ndarray] = None,
    fov_to_z_level: Optional[Dict[int, int]] = None,
    channel: int = 0
) -> Tuple[str, Optional[np.ndarray]]:
    """Load a single image with error handling and optional flatfield correction.
    
    Handles regular TIFF, multi-page TIFF, and OME-TIFF files.
    For multi-page TIFF files (containing "_stack" in name), extracts the middle z-slice.
    For OME-TIFF files, extracts specified z-slice and channel, or uses z-position override.
    
    Parameters
    ----------
    path : Path
        Path to the TIFF file
    flatfield : Optional[np.ndarray]
        Flatfield correction to apply (same shape as image)
    fov_to_z_level : Optional[Dict[int, int]]
        Mapping of FOV number to z-level index for OME-TIFF z-position matching
    channel : int
        Channel index to load for multi-channel images (default 0)
        
    Returns
    -------
    Tuple[str, Optional[np.ndarray]]
        Tuple of (filename, image_array) or (filename, None) if loading failed
    """
    try:
        image = None
        
        # Check if this is an OME-TIFF file
        if path.name.lower().endswith('.ome.tif') or path.name.lower().endswith('.ome.tiff'):
            # Use the image loader for OME-TIFF
            try:
                from ..image_loaders import create_image_loader
                loader = create_image_loader(path, format_hint='ome_tiff')
                meta = loader.metadata
                
                # Use specified channel, determine z-slice
                channel_idx = channel
                z_idx = meta['num_z'] // 2 if meta['num_z'] > 1 else 0
                
                # Check if there's a z-position-aware override
                if fov_to_z_level:
                    fov_info = parse_filename_fov_info(path.name)
                    if fov_info and 'fov' in fov_info and fov_info['fov'] in fov_to_z_level:
                        z_idx = fov_to_z_level[fov_info['fov']]
                        logger.debug(f"Loading {path.name}: using z-override z_level={z_idx} for FOV {fov_info['fov']}")
                
                image = loader.read_slice(channel=channel_idx, z=z_idx)
                logger.debug(f"Loaded OME-TIFF {path.name}: channel {channel_idx}, z-slice {z_idx}/{meta['num_z']}")
            except Exception as e:
                logger.warning(f"Failed to load OME-TIFF with image_loaders, trying tifffile fallback: {e}")
                # Fallback to tifffile with explicit middle z-slice selection
                try:
                    with tifffile.TiffFile(path) as tif:
                        num_pages = len(tif.pages)
                        if num_pages > 1:
                            middle_z = num_pages // 2
                            image = tif.pages[middle_z].asarray()
                            logger.info(f"OME-TIFF fallback: loaded z-slice {middle_z}/{num_pages} from {path.name}")
                        else:
                            image = tif.pages[0].asarray()
                except Exception as fallback_error:
                    logger.error(f"OME-TIFF fallback also failed for {path.name}: {fallback_error}")
                    image = None
        
        # Check if this is a multi-page TIFF
        if image is None and "_stack" in path.name:
            # Multi-page TIFF: extract middle z-slice to match stitching behavior
            with tifffile.TiffFile(path) as tif:
                num_pages = len(tif.pages)
                if num_pages > 1:
                    # Use middle z-slice (same logic as stitcher's middle_layer strategy)
                    middle_z = num_pages // 2
                    image = tif.pages[middle_z].asarray()
                    logger.debug(f"Loaded middle z-slice {middle_z} from {path.name} ({num_pages} total slices)")
                else:
                    # Single page, load normally
                    image = tif.pages[0].asarray()
        elif image is None:
            # Regular TIFF file
            image = tifffile.imread(str(path))
        
        # Apply flatfield correction if provided
        if image is not None and flatfield is not None:
            image = apply_flatfield_to_image(image, flatfield, dtype=image.dtype)
            
        return path.name, image
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
    """[DEPRECATED] Use update_coordinates() instead.
    
    Legacy function for backward compatibility with multi-z workflows.
    """
    return update_coordinates(
        grid, coords_df, filename_to_index, filenames,
        pixel_size_um, reference_idx, full_coords_df, region
    )


def detect_channels(directory: Union[str, Path]) -> Dict[str, List[Path]]:
    """Detect available channels in the directory.
    
    Checks both the given directory and parent's ome_tiff/ subdirectory.
    For OME-TIFF files, reads internal channel metadata.
    
    Parameters
    ----------
    directory : Union[str, Path]
        Directory containing TIFF images
        
    Returns
    -------
    Dict[str, List[Path]]
        Dictionary mapping channel type to list of file paths or channel info
    """
    directory = Path(directory)
    
    # Check for new OME-TIFF structure first
    parent_ome_dir = directory.parent / "ome_tiff"
    if parent_ome_dir.exists() and parent_ome_dir.is_dir():
        tiff_files = list(parent_ome_dir.glob("*.tiff")) + list(parent_ome_dir.glob("*.tif"))
    else:
        tiff_files = list(directory.glob("*.tiff")) + list(directory.glob("*.tif"))
    
    # Channel categories
    channels = {
        "fluorescence": [],
        "brightfield": [],
        "ome_tiff_channels": {}  # Maps channel names to (file, channel_idx)
    }
    
    # Check if we have OME-TIFF files
    ome_tiff_files = [f for f in tiff_files if '.ome.' in f.name.lower()]
    if ome_tiff_files:
        # Read channel info from first OME-TIFF file
        try:
            from ..image_loaders import create_image_loader
            loader = create_image_loader(ome_tiff_files[0], format_hint='ome_tiff')
            meta = loader.metadata
            num_channels = meta['num_channels']
            channel_names = meta.get('channel_names') or [f"channel_{i}" for i in range(num_channels)]
            
            logger.info(f"OME-TIFF: detected {num_channels} channels in {ome_tiff_files[0].name}: {channel_names}")
            
            # Categorize channels by name
            fluor_patterns = [r"Fluorescence", r"\d+ nm", r"GFP", r"RFP", r"DAPI", r"Cy\d"]
            bf_patterns = [r"^BF$", r"brightfield", r"bright.?field", r"phase"]
            
            for idx, ch_name in enumerate(channel_names):
                is_fluor = any(re.search(pattern, ch_name, re.IGNORECASE) for pattern in fluor_patterns)
                is_bf = any(re.search(pattern, ch_name, re.IGNORECASE) for pattern in bf_patterns)
                
                # Store channel info with file path and index
                channels["ome_tiff_channels"][ch_name] = (ome_tiff_files[0], idx)
                
                if is_fluor:
                    channels["fluorescence"].append((ome_tiff_files[0], idx, ch_name))
                elif is_bf:
                    channels["brightfield"].append((ome_tiff_files[0], idx, ch_name))
                else:
                    # Default to fluorescence if unclear
                    channels["fluorescence"].append((ome_tiff_files[0], idx, ch_name))
            
            return channels
        except Exception as e:
            logger.warning(f"Failed to read OME-TIFF channel metadata: {e}")
            # Fall through to filename-based detection
    
    # Regular expressions for channel detection (for non-OME-TIFF files)
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
    """Select file pattern based on image format (not channel-specific).
    
    Returns a glob pattern that matches the image format in the directory.
    Channel selection happens downstream via variance analysis in 
    select_best_registration_channel().
    
    Parameters
    ----------
    directory : Union[str, Path]
        Directory containing TIFF images (typically a timepoint directory like '0/')
        
    Returns
    -------
    str
        Glob pattern like "*.ome.tiff", "*.tiff", etc. (format-only, not channel-specific)
    """
    directory = Path(directory)
    
    # Determine which directory to check for files
    parent_ome_dir = directory.parent / "ome_tiff"
    if parent_ome_dir.exists() and parent_ome_dir.is_dir():
        search_dir = parent_ome_dir
        logger.info(f"Detected ome_tiff/ directory, using {search_dir} for file detection")
    else:
        search_dir = directory
    
    # Check for OME-TIFF files first (most specific format)
    ome_tiff_files = list(search_dir.glob("*.ome.tiff")) + list(search_dir.glob("*.ome.tif"))
    if ome_tiff_files:
        # Check which extension is actually used (.tif vs .tiff)
        example_file = ome_tiff_files[0].name
        if example_file.lower().endswith('.ome.tiff'):
            logger.info("Detected OME-TIFF format with .ome.tiff extension")
            return "*.ome.tiff"
        else:
            logger.info("Detected OME-TIFF format with .ome.tif extension")
            return "*.ome.tif"
    
    # Check for regular TIFF files
    tiff_files = list(search_dir.glob("*.tiff"))
    tif_files = list(search_dir.glob("*.tif"))
    
    if tiff_files:
        logger.info("Detected regular TIFF format with .tiff extension")
        return "*.tiff"
    elif tif_files:
        logger.info("Detected regular TIFF format with .tif extension")
        return "*.tif"
    
    # Default fallback
    logger.warning("No TIFF files found, defaulting to *.tiff pattern")
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
        
        # Compute translation with RANSAC strategy (includes subpixel + optimization)
        ncc_val, y_best_edge_relative, x_best_edge_relative = compute_translation(
            edge1, edge2, yins_edge, xins_edge, *lims_edge_relative[0], *lims_edge_relative[1],
            strategy=TranslationStrategy.RANSAC,
            n=10,
            ransac_config=RANSACConfig(use_ransac=True, residual_threshold=2.0, min_inlier_ratio=0.3),
            subpixel_config=SubpixelConfig(upsample_factor=100, use_subpixel=True),
            optimization_config=OptimizationConfig(method='trf', max_nfev=50, verbose=0)
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
    edge_width: int = DEFAULT_EDGE_WIDTH,
    tensor_backend_engine: Optional[str] = None,
    flatfield_corrections: Optional[Dict[int, np.ndarray]] = None
) -> pd.DataFrame:
    """[DEPRECATED] Use register_tiles(mode=RegistrationMode.AUTO) instead.
    
    Register tiles and update stage coordinates for all regions.
    
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
    tensor_backend_engine : Optional[str]
        Preferred tensor backend engine ('cupy', 'torch', 'numpy', None for auto)
    flatfield_corrections : Optional[Dict[int, np.ndarray]]
        Precomputed flatfield corrections indexed by channel
    flatfield_manifest : Optional[Path]
        Path to flatfield manifest file to load corrections
        
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
    
    # Initialize timepoint for backup filename (fixes the original bug)
    timepoint = "unknown"
    timepoint_dir = None  # Initialize for later use
    
    # Look for coordinates.csv in the timepoint directory
    timepoint_dir = image_directory / "0"
    if timepoint_dir.exists():
        coords_file = timepoint_dir / "coordinates.csv"
        if coords_file.exists():
            csv_path = coords_file
            print(f"Found coordinates file at: {csv_path}")
            # Get timepoint from the subdirectory name (e.g., "0" from the path)
            timepoint = timepoint_dir.name
    
    # Validate that the coordinates file actually exists
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Stage coordinates file not found.\n\n"
            f"Expected location: {csv_path}\n\n"
            f"Registration requires a coordinates.csv file containing the stage positions "
            f"for each field of view. This file is typically generated by your microscope "
            f"acquisition software.\n\n"
            f"Please ensure the coordinates file is present in your dataset directory."
        )
    
    print(f"Using coordinates file: {csv_path}")
    
    # Create backup of original coordinates
    if not skip_backup:
        backup_path = original_coords_dir / f"original_coordinates_{timepoint}.csv"
        import shutil
        shutil.copy2(csv_path, backup_path)
        print(f"Created backup of original coordinates at: {backup_path}")
    
    # Read coordinates
    coords_df = read_coordinates_csv(csv_path)
    
    # Validate coordinates DataFrame structure
    if coords_df.empty:
        raise ValueError(
            f"Coordinates file is empty or contains no data.\n\n"
            f"File location: {csv_path}\n\n"
            f"The coordinates file exists but contains no coordinate data. "
            f"Please check that the file contains valid stage position information."
        )
    
    required_columns = ['region']
    missing_columns = [col for col in required_columns if col not in coords_df.columns]
    if missing_columns:
        raise ValueError(
            f"Coordinates file is missing required information.\n\n"
            f"Missing columns: {missing_columns}\n"
            f"Available columns: {list(coords_df.columns)}\n\n"
            f"The coordinates file must contain a 'region' column to identify "
            f"different acquisition regions. Please check your microscope's "
            f"coordinate file format or contact your facility manager."
        )
    
    # Auto-detect channel pattern if not provided
    if channel_pattern is None:
        try:
            # Look in the 0 subdirectory for TIFF files
            tiff_dir = image_directory / "0"
            if tiff_dir.exists():
                channel_pattern = select_channel_pattern(tiff_dir)
            else:
                channel_pattern = select_channel_pattern(image_directory)
        except Exception as e:
            logging.error(f"Failed to auto-detect channel pattern: {e}")
            # Fallback to generic TIFF pattern
            channel_pattern = "*.tiff"
            logging.warning(f"Using fallback channel pattern: {channel_pattern}")
    
    # Initialize the updated coordinates with a copy
    updated_coords = coords_df.copy()
    
    # Group by region
    regions = coords_df['region'].unique()
    print(f"Found {len(regions)} regions to process: {regions}")
    
    # Track failed regions
    failed_regions = []
    successful_regions = []
    
    for region in regions:
        print(f"\nProcessing region: {region}")
        
        # Filter coordinates for this region
        region_coords = coords_df[coords_df['region'] == region].copy()
        
        # Get image paths for this region only
        # Check for new OME-TIFF structure first: ome_tiff/ directory
        ome_tiff_dir = image_directory / "ome_tiff"
        if ome_tiff_dir.exists() and ome_tiff_dir.is_dir():
            selected_tiff_paths = list(ome_tiff_dir.glob(channel_pattern))
            logger.info(f"Reading images from ome_tiff/ directory: {ome_tiff_dir}")
        else:
            # Look in the 0 subdirectory for TIFF files (legacy structure)
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
        
        # Auto-select best channel for individual files (BEFORE z-slice filtering)
        is_ome_tiff = any(p.name.lower().endswith(('.ome.tif', '.ome.tiff')) for p in region_paths)
        
        if not is_ome_tiff and len(region_paths) > 0:
            # Individual files: Need to filter to best channel
            from ..image_loaders import ChannelSource
            
            try:
                best_channel = select_best_registration_channel(
                    region_paths,
                    region_coords,
                    edge_width=edge_width,
                    num_sample_fovs=3,
                    fov_to_z_level=None
                )
                
                # Get sample FOV to discover channels
                sample_fov_info = parse_filename_fov_info(region_paths[0].name)
                if sample_fov_info:
                    fov = sample_fov_info['fov']
                    z_level = sample_fov_info.get('z_level')
                    
                    # Discover channels
                    descriptors = ChannelSource.discover(
                        files=region_paths,
                        region=region,
                        fov=fov,
                        z_level=z_level
                    )
                    
                    if descriptors and best_channel < len(descriptors):
                        # Get the channel name for the selected channel
                        selected_channel_name = descriptors[best_channel].name
                        
                        # Filter files to only those matching the selected channel
                        filtered_paths = []
                        for p in region_paths:
                            channel_name = ChannelSource._extract_channel_name(p.name)
                            if channel_name == selected_channel_name:
                                filtered_paths.append(p)
                        
                        if filtered_paths:
                            region_paths = filtered_paths
                            logger.info(f"Region {region}: Filtered to {len(region_paths)} files for channel '{selected_channel_name}'")
            except Exception as e:
                logger.warning(f"Channel selection failed for region {region}: {e}. Using all files.")
        
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
        rows, cols, filename_to_index, fov_to_z_level = extract_tile_indices(filenames, region_coords)
        
        try:
            # Register tiles using batched processing
            grid, prop_dict = register_tiles_batched(
                selected_tiff_paths=selected_tiff_paths,
                region_coords=region_coords,
                edge_width=edge_width,
                overlap_diff_threshold=overlap_diff_threshold,
                pou=pou,
                ncc_threshold=ncc_threshold,
                z_slice_to_keep=z_slice_for_registration,
                tensor_backend_engine=tensor_backend_engine,
                flatfield_corrections=flatfield_corrections,
                registration_channel=0
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
            successful_regions.append(region)
            
        except Exception as e:
            logging.error(f"Error processing region {region}: {e}")
            failed_regions.append((region, str(e)))
            # Clean up memory after error
            try:
                backend = get_tensor_backend()
                backend.cleanup_memory()
            except Exception:
                pass
            continue
        
        # Force memory cleanup between regions
        try:
            backend = get_tensor_backend()
            backend.cleanup_memory()
        except:
            pass
    
    # Report registration summary
    print(f"\n=== Registration Summary ===")
    print(f"Total regions: {len(regions)}")
    print(f"Successful: {len(successful_regions)}")
    print(f"Failed: {len(failed_regions)}")
    
    if failed_regions:
        print(f"\nFailed regions:")
        for region, error in failed_regions:
            print(f"  - {region}: {error}")
        
        # Warn if too many failures
        failure_rate = len(failed_regions) / len(regions)
        if failure_rate > 0.5:
            logging.warning(
                f"High failure rate: {failure_rate:.1%} of regions failed. "
                f"Check NCC threshold, image quality, or tile overlap."
            )
    
    # Save updated coordinates to the output path
    updated_coords.to_csv(output_csv_path, index=False)
    print(f"Saved updated coordinates to: {output_csv_path}")
    
    return updated_coords


def process_multiple_timepoints(
    base_directory: Union[str, Path],
    overlap_diff_threshold: float = 10,
    pou: float = 3,
    ncc_threshold: float = 0.5,
    edge_width: int = DEFAULT_EDGE_WIDTH,
    tensor_backend_engine: Optional[str] = None,
    flatfield_corrections: Optional[Dict[int, np.ndarray]] = None
) -> Dict[int, pd.DataFrame]:
    """[DEPRECATED] Use register_tiles(mode=RegistrationMode.MULTI_TIMEPOINT) instead.
    
    Process multiple timepoints from a directory.
    
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
    tensor_backend_engine : Optional[str]
        Preferred tensor backend engine ('cupy', 'torch', 'numpy', None for auto)
    flatfield_corrections : Optional[Dict[int, np.ndarray]]
        Precomputed flatfield corrections indexed by channel
    flatfield_manifest : Optional[Path]
        Path to flatfield manifest file to load corrections
        
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
                edge_width=edge_width,
                tensor_backend_engine=tensor_backend_engine,
                flatfield_corrections=flatfield_corrections
            )
            results[timepoint] = updated_coords
            print(f"Successfully processed timepoint {timepoint}")
        except Exception as e:
            print(f"Error processing timepoint {timepoint}: {e}")
            continue
            
    if not results:
        raise ValueError("No timepoints were successfully processed")
    
    return results
