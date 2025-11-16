"""
config.py

Central configuration constants for the mosaic_generator package.
These values control default image sizes, grid sizes, and tile behavior.
"""

from PIL import Image

# Default preprocessing target size for input images (square: target_size × target_size)
DEFAULT_TARGET_SIZE: int = 512

# Default grid size: (rows, columns)
DEFAULT_GRID_SIZE = (32, 32)

# Base logical size of tiles before per-grid resizing
TILE_BASE_SIZE = (32, 32)

# Added because tile_manager imports TILE_SIZE
TILE_SIZE = 32

# Interpolation method for resizing images and tiles
RESIZE_INTERPOLATION = Image.NEAREST

# Valid image extensions for tiles and sample images
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}