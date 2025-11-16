"""
tile_manager.py

Loads tiles from disk, resizes them once, and computes their average colors.
Fully vectorized and safe for modularized architecture.
"""

import numpy as np
from pathlib import Path
from PIL import Image

from .config import (
    TILE_SIZE,
    VALID_IMAGE_EXTENSIONS,
)


class TileManager:
    """
    Loads and manages tiles for mosaic generation.
    """

    def __init__(self, tile_directory: str):
        self.tile_directory = Path(tile_directory)

        if not self.tile_directory.exists():
            raise RuntimeError(f"Tile directory does not exist: {tile_directory}")

        self.tiles = None
        self.tile_colors = None

        self._load_tiles()

    def _load_tiles(self):
        """
        Load tiles, resize to TILE_SIZE, convert to arrays, compute average color.
        """
        valid_files = [
            f for f in self.tile_directory.iterdir()
            if f.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]

        if not valid_files:
            raise RuntimeError(f"No valid tiles found in {self.tile_directory}")

        tile_list = []

        for path in valid_files:
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize((TILE_SIZE, TILE_SIZE), Image.NEAREST)

                arr = np.array(img, dtype=np.uint8)
                tile_list.append(arr)

            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")

        if not tile_list:
            raise RuntimeError("All tiles failed to load — check your images.")

        # Stack tiles to shape: (#tiles, H, W, 3)
        self.tiles = np.stack(tile_list, axis=0)

        # Precompute average tile color
        self.tile_colors = self.tiles.mean(axis=(1, 2))


    def get_resized_tiles(self, cell_h, cell_w):
        """
        Resize cached tiles to target cell size.
        """

        resized = np.empty((self.tiles.shape[0], cell_h, cell_w, 3), dtype=np.uint8)

        for i, tile in enumerate(self.tiles):
            tile_img = Image.fromarray(tile)
            tile_resized = tile_img.resize((cell_w, cell_h), Image.NEAREST)
            resized[i] = np.array(tile_resized, dtype=np.uint8)

        return resized