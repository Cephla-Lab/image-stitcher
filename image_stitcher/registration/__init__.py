"""Registration module for image stitching.

This module provides functionality for registering microscope image tiles
without performing full stitching.
"""

from .tile_registration import register_tiles
from .registration_viz import visualize_registration
from ._constrained_refinement import refine_translations
from ._global_optimization import compute_final_position

__all__ = [
    'register_tiles',
    'visualize_registration',
    'refine_translations',
    'compute_final_position',
] 