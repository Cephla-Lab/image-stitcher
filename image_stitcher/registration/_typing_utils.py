"""Type aliases and utilities for image registration.

This module provides commonly used type aliases for numpy arrays and numeric types
used throughout the registration package.
"""
from typing import Any, Union

import numpy as np
import numpy.typing as npt

# Array type aliases
NumArray = npt.NDArray[Any]
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# Numeric type aliases
Int = Union[int, np.int64]
Float = Union[float, np.float64]
