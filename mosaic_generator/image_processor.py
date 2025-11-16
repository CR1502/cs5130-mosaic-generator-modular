"""
image_processor.py

Module responsible for image loading, RGB conversion, resizing,
and vectorized grid extraction for mosaic processing.
"""

import numpy as np
from PIL import Image
from pathlib import Path

from .config import (
    DEFAULT_TARGET_SIZE,
    RESIZE_INTERPOLATION,
    DEFAULT_GRID_SIZE,
    VALID_IMAGE_EXTENSIONS,
)


def load_image(path: str | Path) -> np.ndarray:
    """
    Load an image from disk and convert it to an RGB numpy array.

    Parameters
    ----------
    path : str or Path
        Path to the image file.

    Returns
    -------
    np.ndarray
        RGB image as uint8 array of shape (H, W, 3).
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def preprocess_image(image, target_size: int = DEFAULT_TARGET_SIZE) -> np.ndarray:
    """
    Convert input image to RGB and resize it to a square target size.

    Parameters
    ----------
    image : PIL.Image.Image or np.ndarray
        Input image.

    target_size : int
        Final width/height of the preprocessed image.

    Returns
    -------
    np.ndarray
        Resized RGB image of shape (target_size, target_size, 3).
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((target_size, target_size), RESIZE_INTERPOLATION)
    return np.array(image, dtype=np.uint8)


def compute_grid_shapes(image_array: np.ndarray, grid_size=DEFAULT_GRID_SIZE):
    """
    Compute height/width of each grid cell.

    Parameters
    ----------
    image_array : np.ndarray
        Preprocessed image array of shape (H, W, 3).

    grid_size : tuple(int, int)
        Number of grid rows and columns.

    Returns
    -------
    tuple(int, int, int, int)
        (grid_h, grid_w, cell_h, cell_w)
    """
    height, width = image_array.shape[:2]
    grid_h, grid_w = grid_size

    cell_h = height // grid_h
    cell_w = width // grid_w

    return grid_h, grid_w, cell_h, cell_w


def extract_cells_and_colors(image_array: np.ndarray, grid_size=DEFAULT_GRID_SIZE):
    """
    Vectorized extraction of grid cells and their averaged colors.

    Parameters
    ----------
    image_array : np.ndarray
        Preprocessed RGB image of shape (H, W, 3).

    grid_size : tuple(int, int)
        (grid_h, grid_w)

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        cells : array of shape (grid_h, grid_w, cell_h, cell_w, 3)
        cell_colors : array of shape (grid_h, grid_w, 3)
    """
    grid_h, grid_w, cell_h, cell_w = compute_grid_shapes(image_array, grid_size)

    # reshape + transpose → extract grid without loops
    cells = (
        image_array
        .reshape(grid_h, cell_h, grid_w, cell_w, 3)
        .transpose(0, 2, 1, 3, 4)
    )

    # vectorized average color per cell
    cell_colors = cells.mean(axis=(2, 3))

    return cells, cell_colors