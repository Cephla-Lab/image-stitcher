"""Constrained refinement module for microscope tile registration.

This module provides functionality for refining tile translations under constraints.
It implements a grid-based optimization approach to find the best translation values
while respecting physical constraints of the microscope stage movement.

The module includes:
- Integer-constrained local maximum finding
- Translation refinement using normalized cross-correlation
- Robust handling of boundary conditions
"""
import itertools
from typing import Callable, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from ._translation_computation import extract_overlap_subregion, ncc
from ._typing_utils import Float, FloatArray, Int, NumArray

# Configure logger
logger = logging.getLogger(__name__)

# Constants
MAX_ITERATIONS = 100
MIN_IMPROVEMENT = 1e-6  # Minimum improvement to continue optimization

def validate_optimization_inputs(
    func: Callable[[FloatArray], Float],
    init_x: FloatArray,
    limits: FloatArray
) -> None:
    """Validate inputs for optimization function.
    
    Args:
        func: Objective function to optimize
        init_x: Initial parameter values
        limits: Parameter limits as [[min1, max1], [min2, max2], ...]
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not callable(func):
        raise ValueError("func must be callable")
        
    init_x = np.asarray(init_x)
    limits = np.asarray(limits)
    
    if init_x.ndim != 1:
        raise ValueError("init_x must be 1-dimensional")
    if limits.ndim != 2 or limits.shape[1] != 2:
        raise ValueError("limits must be a 2D array with shape (n_params, 2)")
    if init_x.shape[0] != limits.shape[0]:
        raise ValueError("init_x and limits must have compatible shapes")
        
    # Check if initial point is within limits
    if not (np.all(limits[:, 0] <= init_x) and np.all(init_x <= limits[:, 1])):
        raise ValueError("Initial point must be within limits")

def find_local_max_integer_constrained(
    func: Callable[[FloatArray], Float],
    init_x: FloatArray,
    limits: FloatArray,
    max_iter: Int = MAX_ITERATIONS,
) -> Tuple[FloatArray, Float]:
    """Find local maximum of a function with integer-constrained parameters.

    Uses a grid-based search around the current point, moving to better solutions
    until no improvement is found or max_iter is reached.

    Args:
        func: Objective function to maximize
        init_x: Initial parameter values
        limits: Parameter limits as [[min1, max1], [min2, max2], ...]
        max_iter: Maximum number of iterations (default: 100)

    Returns:
        Tuple of (optimal_parameters, optimal_value)

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If optimization fails
    """
    validate_optimization_inputs(func, init_x, limits)
    
    init_x = np.asarray(init_x, dtype=np.float64)
    limits = np.asarray(limits, dtype=np.float64)
    dim = init_x.shape[0]
    
    try:
        value = func(init_x)
    except Exception as e:
        raise RuntimeError(f"Failed to evaluate objective function: {e}")
        
    if not np.isfinite(value):
        raise ValueError("Initial function value is not finite")
    
    x = init_x.copy()
    prev_value = float('-inf')
    
    for iteration in range(max_iter):
        # Generate neighboring points with integer steps
        neighbors = [x + np.array(dxs) for dxs in itertools.product([-1, 0, 1], repeat=dim)]
        valid_neighbors = [
            n for n in neighbors
            if np.all(limits[:, 0] <= n) and np.all(n <= limits[:, 1])
        ]
        
        if not valid_neighbors:
            logger.warning("No valid neighboring points found")
            break
            
        # Evaluate all valid neighbors
        try:
            neighbor_values = np.array([func(n) for n in valid_neighbors])
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate neighbors at iteration {iteration}: {e}")
            
        if not np.any(np.isfinite(neighbor_values)):
            raise RuntimeError(f"All neighbor evaluations returned non-finite values at iteration {iteration}")
            
        # Find best neighbor
        best_idx = np.argmax(neighbor_values)
        best_value = neighbor_values[best_idx]
        
        # Check for improvement
        if best_value <= value or abs(best_value - prev_value) < MIN_IMPROVEMENT:
            break
            
        x = np.array(valid_neighbors[best_idx])
        prev_value = value
        value = best_value
        
        logger.debug(f"Iteration {iteration}: value = {value:.6f}, x = {x}")
    
    return x, value

def refine_translations(
    images: NumArray,
    grid: pd.DataFrame,
    r: Float,
    edge_width: Int # New parameter
) -> pd.DataFrame:
    """Refine tile translations using normalized cross-correlation on edge strips.

    For each tile pair, optimizes the translation values within a constrained
    region around the initial estimate to maximize the normalized cross-correlation
    on their corresponding edges.

    Args:
        images: Array of full tile images
        grid: DataFrame with initial global translations
        r: Maximum allowed refinement distance in pixels (for global translation)
        edge_width: Width of the edge strips to use for NCC computation

    Returns:
        DataFrame with refined global translations and correlation values
    """
    if not isinstance(grid, pd.DataFrame):
        raise ValueError("grid must be a pandas DataFrame")
    if not isinstance(r, (int, float)) or r < 0:
        raise ValueError("r must be a non-negative number")
    if not isinstance(edge_width, int) or edge_width <= 0:
        raise ValueError("edge_width must be a positive integer")

    required_cols = ['left', 'top'] # Add more if other _second columns are used
    for direction in ["left", "top"]:
        required_cols.extend([f"{direction}_y_second", f"{direction}_x_second"])
    
    missing_cols = [col for col in required_cols if col not in grid.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in grid: {missing_cols}")

    # Assuming all images have the same shape for consistent edge extraction
    # If not, image shapes should be fetched per image pair.
    # This is fetched once assuming homogeneity, can be moved inside loop if images vary.
    if images.shape[0] > 0:
        # These are full image dimensions
        img_shape_y, img_shape_x = images[0].shape[-2:] # Handle potential multi-channel later if necessary
    else:
        logger.warning("No images provided to refine_translations. Returning grid as is.")
        return grid


    for direction in ["left", "top"]:
        for i2, g in tqdm(grid.iterrows(), total=len(grid), desc=f"Refining {direction} translations on edges"):
            i1 = g[direction]
            if pd.isna(i1):
                continue

            try:
                image1_full = images[int(i1)]
                image2_full = images[int(i2)]
            except IndexError as e:
                raise ValueError(f"Invalid image index: {e}")

            if image1_full.shape != image2_full.shape:
                # This check should ideally be done earlier or shapes handled more dynamically
                logger.warning(f"Inconsistent image shapes for pair ({i1}, {i2}). Skipping refinement for this pair.")
                # Fallback: Keep initial values for this pair if shapes differ
                grid.loc[i2, f"{direction}_y"] = g[f"{direction}_y_second"]
                grid.loc[i2, f"{direction}_x"] = g[f"{direction}_x_second"]
                grid.loc[i2, f"{direction}_ncc"] = float('nan')
                continue

            original_sizeY, original_sizeX = image1_full.shape # Full dimensions of this specific image

            # Extract edges based on direction
            if direction == "left":  # i2 is to the right of i1
                current_edge_w = min(edge_width, original_sizeX)
                image1_edge = image1_full[:, -current_edge_w:]
                image2_edge = image2_full[:, :current_edge_w]
            elif direction == "top":  # i2 is below i1
                current_edge_w = min(edge_width, original_sizeY)
                image1_edge = image1_full[-current_edge_w:, :]
                image2_edge = image2_full[:current_edge_w, :]
            else:
                continue # Should not happen

            if image1_edge.size == 0 or image2_edge.size == 0:
                logger.warning(f"Empty edge strip for pair ({i1}, {i2}), direction {direction}. Skipping.")
                grid.loc[i2, f"{direction}_y"] = g[f"{direction}_y_second"]
                grid.loc[i2, f"{direction}_x"] = g[f"{direction}_x_second"]
                grid.loc[i2, f"{direction}_ncc"] = float('nan')
                continue

            def overlap_ncc_on_edges(params: FloatArray) -> Float:
                """
                Compute NCC on edges for given GLOBAL translation parameters.
                `params` are (global_ty, global_tx)
                """
                global_ty_param, global_tx_param = params.astype(int)

                # Convert global translation to edge-relative translation
                # This is the translation of image2_edge relative to image1_edge's origin
                if direction == "left":
                    ty_edge_relative = global_ty_param
                    # offset of image1_edge's origin within image1_full (horizontal)
                    offset_x_image1_edge = original_sizeX - current_edge_w
                    tx_edge_relative = global_tx_param - offset_x_image1_edge
                elif direction == "top":
                    # offset of image1_edge's origin within image1_full (vertical)
                    offset_y_image1_edge = original_sizeY - current_edge_w
                    ty_edge_relative = global_ty_param - offset_y_image1_edge
                    tx_edge_relative = global_tx_param
                else: # Should not happen
                    logger.error(f"Unexpected direction '{direction}' in overlap_ncc_on_edges.")
                    return float('-inf')
                
                try:
                    # Parameters for extract_overlap_subregion are (image, y_translation, x_translation)
                    subI1 = extract_overlap_subregion(image1_edge, ty_edge_relative, tx_edge_relative)
                    subI2 = extract_overlap_subregion(image2_edge, -ty_edge_relative, -tx_edge_relative)
                    
                    if subI1.size == 0 or subI2.size == 0 : # No overlap for these edge translations
                        return float('-inf') # Or a very small number / nan, depending on desired behavior
                    
                    return ncc(subI1, subI2)
                except Exception as e:
                    # Log details for debugging
                    logger.error(f"NCC computation failed for global_params ({global_ty_param},{global_tx_param}) "
                                 f"-> edge_relative ({ty_edge_relative},{tx_edge_relative}) "
                                 f"on edges of shape {image1_edge.shape} & {image2_edge.shape}. Error: {e}")
                    return float('-inf')

            # Initial guess for GLOBAL translation (from previous steps)
            init_values_global = np.array([
                int(g[f"{direction}_y_second"]),
                int(g[f"{direction}_x_second"])
            ])

            # Limits for find_local_max_integer_constrained are for GLOBAL translations
            # So, use original_sizeY, original_sizeX (full image dimensions)
            limits_global = np.array([
                [max(-original_sizeY + 1, init_values_global[0] - r), min(original_sizeY - 1, init_values_global[0] + r)],
                [max(-original_sizeX + 1, init_values_global[1] - r), min(original_sizeX - 1, init_values_global[1] + r)]
            ])
            
            try:
                # find_local_max_integer_constrained optimizes in GLOBAL translation space
                refined_global_values, ncc_value = find_local_max_integer_constrained(
                    overlap_ncc_on_edges, init_values_global, limits_global
                )
                
                grid.loc[i2, f"{direction}_y"] = refined_global_values[0]
                grid.loc[i2, f"{direction}_x"] = refined_global_values[1]
                grid.loc[i2, f"{direction}_ncc"] = ncc_value
                
            except Exception as e:
                logger.error(f"Refinement failed for tile pair ({i1}, {i2}), direction {direction}: {e}")
                grid.loc[i2, f"{direction}_y"] = init_values_global[0]
                grid.loc[i2, f"{direction}_x"] = init_values_global[1]
                grid.loc[i2, f"{direction}_ncc"] = float('nan')
    
    for direction in ["left", "top"]:
        for dim_char in "yx":
            key = f"{direction}_{dim_char}"
            if key in grid.columns:
                grid[key] = pd.array(np.round(grid[key]), dtype='Int32')
            key_second = f"{direction}_{dim_char}_second" # Also ensure _second columns are correct type if used for init
            if key_second in grid.columns:
                grid[key_second] = pd.array(np.round(grid[key_second]), dtype='Int32')


    return grid