import itertools
import warnings
from typing import Tuple, Any, List, Optional
from contextlib import contextmanager
import time

import numpy as np
import numpy.typing as npt
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    cp = np  # Use numpy as fallback

# Assuming _typing_utils are (or are compatible with):
# Float = float
# Int = int
# FloatArray = npt.NDArray[np.float_] 
# IntArray = npt.NDArray[np.int_]   
# NumArray = npt.NDArray[Any]       

from ._typing_utils import Float
from ._typing_utils import FloatArray
from ._typing_utils import Int
from ._typing_utils import IntArray
from ._typing_utils import NumArray

# Constants for GPU memory management
MAX_ARRAY_SIZE_GB = 2 # Maximum single array size in GB
if HAS_CUDA:
    MEMORY_POOL = cp.get_default_memory_pool()
    PINNED_MEMORY_POOL = cp.get_default_pinned_memory_pool()

@contextmanager
def managed_gpu_memory():
    """Context manager for GPU memory management.
    
    Ensures GPU memory is properly managed and freed after use.
    """
    if not HAS_CUDA:
        yield
        return
        
    try:
        yield
    finally:
        MEMORY_POOL.free_all_blocks()
        PINNED_MEMORY_POOL.free_all_blocks()

def validate_image_pair(image1: NumArray, image2: NumArray) -> None:
    """Validate a pair of images for translation computation.
    
    Args:
        image1: First image
        image2: Second image
        
    Raises:
        ValueError: If images are invalid or incompatible
    """
    if image1.ndim != 2 or image2.ndim != 2:
        raise ValueError("Images must be 2-dimensional")
    if image1.shape != image2.shape:
        raise ValueError(f"Images must have same shape. Got {image1.shape} and {image2.shape}")
    if not np.isfinite(image1).all() or not np.isfinite(image2).all():
        raise ValueError("Images contain non-finite values")
    
    array_size_gb = image1.nbytes / (1024**3)
    if array_size_gb > MAX_ARRAY_SIZE_GB:
        warnings.warn(f"Large image detected ({array_size_gb:.1f} GB). Consider downsampling.")

def check_gpu_memory_available(required_memory_gb: float = 2.0) -> bool:
    """Check if enough GPU memory is available.
    
    Args:
        required_memory_gb: Required memory in GB
        
    Returns:
        bool: True if enough memory is available, False otherwise
    """
    if not HAS_CUDA:
        return False
        
    try:
        mempool = cp.get_default_memory_pool()
        free_memory = mempool.free_bytes() / (1024**3)  # Convert to GB
        return free_memory >= required_memory_gb
    except:
        return False

def pcm(image1: NumArray, image2: NumArray) -> FloatArray:
    """Compute peak correlation matrix for two images using GPU acceleration.

    The PCM is computed using the normalized cross-power spectrum method:
    PCM = IFFT(F1 * conj(F2) / |F1 * conj(F2)|)

    Args:
        image1: First image (2D array)
        image2: Second image (2D array, same size as image1)

    Returns:
        Peak correlation matrix as a 2D float array

    Raises:
        ValueError: If images are invalid or incompatible
        RuntimeError: If GPU computation fails
    """
    validate_image_pair(image1, image2)
    
    # Estimate required memory (rough estimate)
    required_memory_gb = (image1.nbytes + image2.nbytes) * 4 / (1024**3)  # 4x for FFT operations
    
    with managed_gpu_memory():
        try:
            if HAS_CUDA and check_gpu_memory_available(required_memory_gb):
                # GPU computation with memory management
                image1_cp = cp.asarray(image1, dtype=cp.float32)
                image2_cp = cp.asarray(image2, dtype=cp.float32)

                # Free memory after FFT computations
                F1_cp = cp.fft.fft2(image1_cp)
                del image1_cp
                F2_cp = cp.fft.fft2(image2_cp)
                del image2_cp
                
                FC_cp = F1_cp * cp.conjugate(F2_cp)
                del F1_cp, F2_cp
                
                FC_abs_cp = cp.abs(FC_cp)
                FC_normalized_cp = cp.where(FC_abs_cp > 0, FC_cp / FC_abs_cp, 0)
                del FC_cp, FC_abs_cp

                result_cp = cp.fft.ifft2(FC_normalized_cp).real.astype(cp.float32)
                del FC_normalized_cp
                
                result = cp.asnumpy(result_cp)
                del result_cp
                return result
            else:
                # CPU fallback
                F1 = np.fft.fft2(image1)
                F2 = np.fft.fft2(image2)
                FC = F1 * np.conjugate(F2)
                
                FC_abs = np.abs(FC)
                FC_normalized = np.where(FC_abs > 0, FC / FC_abs, 0)

                result = np.fft.ifft2(FC_normalized).real.astype(np.float32)
                return result
            
        except Exception as e:
            if HAS_CUDA:
                # Try to clean up GPU memory on error
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except:
                    pass
                raise RuntimeError(f"GPU computation failed: {e}")
            else:
                raise RuntimeError(f"CPU computation failed: {e}")

def multi_peak_max(PCM: FloatArray) -> Tuple[IntArray, IntArray, FloatArray]:
    """Find the first to n th largest peaks in PCM.

    Parameters
    ---------
    PCM : np.ndarray
        the peak correlation matrix

    Returns
    -------
    rows : np.ndarray
        the row indices for the peaks
    cols : np.ndarray
        the column indices for the peaks
    vals : np.ndarray
        the values of the peaks
    """
    if HAS_CUDA:
        PCM_cp = cp.asarray(PCM)
        flat_sorted_indices = cp.argsort(PCM_cp.ravel())
        row_cp, col_cp = cp.unravel_index(flat_sorted_indices, PCM_cp.shape)
        
        row_rev_cp = row_cp[::-1]
        col_rev_cp = col_cp[::-1]

        vals_cp = PCM_cp[row_rev_cp, col_rev_cp] 

        return cp.asnumpy(row_rev_cp), cp.asnumpy(col_rev_cp), cp.asnumpy(vals_cp)
    else:
        # CPU fallback
        flat_sorted_indices = np.argsort(PCM.ravel())
        row, col = np.unravel_index(flat_sorted_indices, PCM.shape)
        
        row_rev = row[::-1]
        col_rev = col[::-1]

        vals = PCM[row_rev, col_rev]
        
        return row_rev, col_rev, vals

def ncc(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compute normalized cross-correlation between two images.
    
    Args:
        image1: First image array
        image2: Second image array
        
    Returns:
        NCC value between -1 and 1
        
    Raises:
        ValueError: If images are invalid or too small
    """
    if image1.size == 0 or image2.size == 0:
        return float('-inf')
        
    if image1.shape != image2.shape:
        return float('-inf')
        
    # Ensure minimum size for meaningful correlation
    if image1.size < 100:  # Arbitrary minimum size
        return float('-inf')
        
    try:
        if HAS_CUDA:
            # Normalize images
            image1_norm = image1 - cp.mean(image1)
            image2_norm = image2 - cp.mean(image2)
            
            # Compute correlation
            numerator = cp.sum(image1_norm * image2_norm)
            denominator = cp.sqrt(cp.sum(image1_norm**2) * cp.sum(image2_norm**2))
            
            # Handle potential division by zero
            if denominator == 0:
                return float('-inf')
                
            return float(numerator / denominator)
        else:
            # CPU fallback
            # Normalize images
            image1_norm = image1 - np.mean(image1)
            image2_norm = image2 - np.mean(image2)
            
            # Compute correlation
            numerator = np.sum(image1_norm * image2_norm)
            denominator = np.sqrt(np.sum(image1_norm**2) * np.sum(image2_norm**2))
            
            # Handle potential division by zero
            if denominator == 0:
                return float('-inf')
                
            return float(numerator / denominator)
    except Exception as e:
        warnings.warn(f"NCC computation failed: {e}")
        return float('-inf')

def extract_overlap_subregion(image: NumArray, y: Int, x: Int) -> NumArray:
    """Extract the overlapping subregion of the image.
    """
    sizeY = image.shape[0]
    sizeX = image.shape[1]
    assert (abs(y) < sizeY) and (abs(x) < sizeX) # Python abs()
    
    # Original code had key=int in min/max, which is redundant for direct numeric args
    xstart = int(max(0, min(y, sizeY))) 
    xend = int(max(0, min(y + sizeY, sizeY)))
    ystart = int(max(0, min(x, sizeX)))
    yend = int(max(0, min(x + sizeX, sizeX)))
    return image[xstart:xend, ystart:yend]

def interpret_translation(
    image1: NumArray,
    image2: npt.NDArray,
    yins: IntArray,
    xins: IntArray,
    ymin: Int,
    ymax: Int,
    xmin: Int,
    xmax: Int,
    n: Int = 2,
) -> Tuple[float, int, int]:
    """Interpret the translation to find the translation with heighest ncc."""
    # profiling_total_start = time.time() # Keep for profiling if needed
    
    image1_cp = cp.asarray(image1)
    image2_cp = cp.asarray(image2)
    yins_cp = cp.asarray(yins)
    xins_cp = cp.asarray(xins)

    assert image1_cp.ndim == 2
    assert image2_cp.ndim == 2
    assert image1_cp.shape == image2_cp.shape

    sizeY = image1_cp.shape[0]
    sizeX = image1_cp.shape[1]

    if yins_cp.size > 0:
        assert cp.all(0 <= yins_cp) and cp.all(yins_cp < sizeY)
    if xins_cp.size > 0: # Should have same size as yins_cp
        assert cp.all(0 <= xins_cp) and cp.all(xins_cp < sizeX)

    _ncc = -float('inf')
    y_best = 0
    x_best = 0

    if yins_cp.size == 0: # No peaks to process
        # print(f"Total execution time: {(time.time() - profiling_total_start)*1000:.2f}ms")
        return _ncc, y_best, x_best

    # profiling_mag_start = time.time()
    # Calculate magnitude arrays using full yins_cp, xins_cp
    ymags0_full_cp = yins_cp
    ymags1_full_cp = sizeY - yins_cp
    ymags1_full_cp[ymags0_full_cp == 0] = 0 # Handles yins_cp being 0
    ymagss_full_cp: List[cp.ndarray] = [ymags0_full_cp, ymags1_full_cp]

    xmags0_full_cp = xins_cp
    xmags1_full_cp = sizeX - xins_cp
    xmags1_full_cp[xmags0_full_cp == 0] = 0 # Handles xins_cp being 0
    xmagss_full_cp: List[cp.ndarray] = [xmags0_full_cp, xmags1_full_cp]
    # print(f"Magnitude calculation time: {(time.time() - profiling_mag_start)*1000:.2f}ms")
    
    # profiling_candidates_start = time.time()
    # Iteratively compute valid_ind_cp (shape N_peaks_total) to save memory
    valid_ind_cp = cp.zeros(yins_cp.shape, dtype=bool)
    signs = [-1, +1]

    for ymags_item_cp in ymagss_full_cp:
        for xmags_item_cp in xmagss_full_cp:
            for ysign_val in signs: # Renamed to avoid conflict if outer scope has ysign
                yvals_cp = ymags_item_cp * ysign_val
                for xsign_val in signs: # Renamed to avoid conflict
                    xvals_cp = xmags_item_cp * xsign_val
                    current_combination_valid_cp = \
                        (ymin <= yvals_cp) & (yvals_cp <= ymax) & \
                        (xmin <= xvals_cp) & (xvals_cp <= xmax)
                    valid_ind_cp |= current_combination_valid_cp
    # print(f"Candidate generation (for validity_check) time: {(time.time() - profiling_candidates_start)*1000:.2f}ms")
    
    # profiling_filter_start = time.time()
    # Original code had `assert np.any(valid_ind)`. Replicate this.
    # This assertion means the code expects at least one peak to be valid under some combination.
    # If this is not always true, this assert might need to be removed or made conditional.
    if not cp.any(valid_ind_cp): # If no peak is valid under any combination
         # print(f"No valid translation candidates found. Total time: {(time.time() - profiling_total_start)*1000:.2f}ms")
         return _ncc, y_best, x_best # Or handle as per original code's expectation on assert failure

    true_indices_in_valid_ind_cp = cp.where(valid_ind_cp)[0]
    
    num_to_take = min(int(n), true_indices_in_valid_ind_cp.size)

    if num_to_take == 0: # No valid candidates to check further (e.g. n=0 or no valid peaks after all)
        # print(f"No candidates to take. Total time: {(time.time() - profiling_total_start)*1000:.2f}ms")
        return _ncc, y_best, x_best

    indices_of_interest_cp = true_indices_in_valid_ind_cp[:num_to_take]
    # print(f"Position filtering time: {(time.time() - profiling_filter_start)*1000:.2f}ms")

    # Subset yins, xins to only the peaks of interest (typically very few, e.g., n=2)
    yins_subset_cp = yins_cp[indices_of_interest_cp]
    xins_subset_cp = xins_cp[indices_of_interest_cp]

    # Recalculate magnitude arrays for the small subset
    ymags0_subset_cp = yins_subset_cp
    ymags1_subset_cp = sizeY - yins_subset_cp
    if ymags0_subset_cp.size > 0: # Check size before indexing to avoid issues with 0-size arrays
         ymags1_subset_cp[ymags0_subset_cp == 0] = 0
    ymagss_subset_cp: List[cp.ndarray] = [ymags0_subset_cp, ymags1_subset_cp]
    
    xmags0_subset_cp = xins_subset_cp
    xmags1_subset_cp = sizeX - xins_subset_cp
    if xmags0_subset_cp.size > 0:
        xmags1_subset_cp[xmags0_subset_cp == 0] = 0
    xmagss_subset_cp: List[cp.ndarray] = [xmags0_subset_cp, xmags1_subset_cp]

    # Generate candidate positions _only_ for the small subset of peaks
    _poss_list_final = []
    # Determine dtype for poss_final_cp. Fallback logic for empty inputs.
    if yins_subset_cp.size > 0:
        dtype_for_poss = yins_subset_cp.dtype
    elif yins_cp.size > 0:
        dtype_for_poss = yins_cp.dtype
    else: # Should not be reached if yins_cp.size == 0 check at start is effective
        dtype_for_poss = cp.int_

    for ymags_item_subset_cp in ymagss_subset_cp:
        for xmags_item_subset_cp in xmagss_subset_cp:
            for ysign_val in signs:
                yvals_subset_cp = ymags_item_subset_cp * ysign_val
                for xsign_val in signs:
                    xvals_subset_cp = xmags_item_subset_cp * xsign_val
                    _poss_list_final.append([yvals_subset_cp, xvals_subset_cp])
    
    # poss_final_cp will be small: (16, 2, num_to_take)
    if not _poss_list_final or yins_subset_cp.size == 0: # handles num_to_take = 0
         poss_final_cp = cp.empty((16, 2, 0), dtype=dtype_for_poss)
    else:
        poss_final_cp = cp.array(_poss_list_final, dtype=dtype_for_poss)

    # Transfer to CPU for iteration: shape (num_to_take, 16, 2)
    iterable_candidates_host = cp.asnumpy(cp.moveaxis(poss_final_cp, -1, 0))

    # profiling_eval_start = time.time()
    ncc_calls = 0
    for pos_item_host in iterable_candidates_host: # pos_item_host is (16, 2) NumPy array
        for yval_scalar, xval_scalar in pos_item_host:
            yval_int = int(yval_scalar) # Ensure Python native int
            xval_int = int(xval_scalar)

            # Final check of bounds for this specific yval, xval pair
            if (ymin <= yval_int) and (yval_int <= ymax) and (xmin <= xval_int) and (xval_int <= xmax):
                subI1_cp = extract_overlap_subregion(image1_cp, yval_int, xval_int)
                subI2_cp = extract_overlap_subregion(image2_cp, -yval_int, -xval_int)
                
                # ncc is called with CuPy arrays, returns Python float
                ncc_val = ncc(subI1_cp, subI2_cp) 
                ncc_calls += 1
                if ncc_val > _ncc: # _ncc is Python float
                    _ncc = ncc_val
                    y_best = yval_int
                    x_best = xval_int
    
    # print(f"Candidate evaluation time: {(time.time() - profiling_eval_start)*1000:.2f}ms")
    # print(f"Number of NCC calls: {ncc_calls}")
    # print(f"Total execution time: {(time.time() - profiling_total_start)*1000:.2f}ms")
    
    return _ncc, y_best, x_best