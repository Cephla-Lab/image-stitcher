"""Registration module for image stitching.

This module provides functionality for registering microscope image tiles
without performing full stitching.
"""

from .tile_registration import (
    register_tiles,
    register_and_update_coordinates,
    process_multiple_timepoints,
    DEFAULT_FOV_RE,
    extract_tile_indices
)
from .registration_viz import visualize_registration
from ._constrained_refinement import refine_translations
from ._global_optimization import compute_final_position

__all__ = [
    'register_tiles',
    'register_and_update_coordinates',
    'process_multiple_timepoints',
    'visualize_registration',
    'refine_translations',
    'compute_final_position',
    'DEFAULT_FOV_RE',
    'extract_tile_indices'
] 
