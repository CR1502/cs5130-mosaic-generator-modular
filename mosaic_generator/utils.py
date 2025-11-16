"""
utils.py

Utility functions used across the mosaic generator package.
Includes timing decorators, safe image loading, and validation helpers.
"""

import time
import numpy as np
from PIL import Image


# ---------------------------------------------------------
# Timing Utility
# ---------------------------------------------------------

def timer(func):
    """
    Decorator that records execution time of any function.

    Usage:
    @timer
    def my_function(...):
        ...
    """

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()

        elapsed = end - start
        return result, elapsed

    return wrapper


# ---------------------------------------------------------
# Safe Image Loader
# ---------------------------------------------------------

def load_image_safe(path):
    """
    Safely load an image from disk.

    Parameters
    ----------
    path : str or Path
        Path to the image file.

    Returns
    -------
    PIL.Image
        Loaded RGB image.

    Raises
    ------
    RuntimeError
        If file cannot be opened.
    """
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        raise RuntimeError(f"Failed to load image '{path}': {e}")


# ---------------------------------------------------------
# Grid Validation
# ---------------------------------------------------------

def validate_grid_size(grid_size):
    """
    Validate grid size tuple (h, w).

    Conditions:
    - must be tuple of 2 ints
    - each must be > 0
    - both must match (square grid) [your project requirement]

    Parameters
    ----------
    grid_size : tuple(int,int)

    Raises
    ------
    ValueError
        If grid size is invalid.
    """

    if (
        not isinstance(grid_size, tuple)
        or len(grid_size) != 2
        or not all(isinstance(x, int) for x in grid_size)
    ):
        raise ValueError("Grid size must be a tuple of two integers (h, w).")

    if grid_size[0] <= 0 or grid_size[1] <= 0:
        raise ValueError("Grid dimensions must be positive integers.")

    if grid_size[0] != grid_size[1]:
        raise ValueError("Grid must be square (h == w) for this implementation.")

    # valid → return normalized form
    return grid_size


# ---------------------------------------------------------
# Array Conversion Helper
# ---------------------------------------------------------

def ensure_numpy(arr):
    """
    Convert a PIL image to NumPy if needed.

    Parameters
    ----------
    arr : PIL.Image or np.ndarray

    Returns
    -------
    np.ndarray
    """
    if isinstance(arr, np.ndarray):
        return arr
    return np.array(arr, dtype=np.uint8)