"""
tile_manager.py

Handles loading tiles, caching them in memory, computing features,
and efficiently providing resized versions for mosaic assembly.
"""

import numpy as np
from PIL import Image
from pathlib import Path

from .config import (
    VALID_IMAGE_EXTENSIONS,
    TILE_BASE_SIZE,
    RESIZE_INTERPOLATION,
)


class TileManager:
    """
    Loads tile images, computes their representative features,
    and caches resized versions for performance.
    """

    def __init__(self, tile_directory: str):
        self.tile_directory = Path(tile_directory)

        if not self.tile_directory.exists():
            raise FileNotFoundError(
                f"Tile directory not found: {self.tile_directory}"
            )

        self.tiles = None                # shape: (N, 32, 32, 3)
        self.tile_colors = None          # shape: (N, 3)
        self.resized_tiles_cache = {}    # keyed by (cell_h, cell_w)

        self._load_tiles()

    # ---------------------------------------------------------
    # Tile Loading
    # ---------------------------------------------------------

    def _load_tiles(self):
        """Load tiles from disk and precompute their average colors."""

        tile_files = [
            f for f in self.tile_directory.iterdir()
            if f.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]

        if not tile_files:
            raise RuntimeError(
                f"No tiles found in {self.tile_directory}"
            )

        tile_list = []
        for tile_path in tile_files:
            try:
                img = Image.open(tile_path).convert("RGB")
                img = img.resize((TILE_BASE_SIZE, TILE_BASE_SIZE), RESIZE_INTERPOLATION)
                tile_list.append(np.array(img, dtype=np.uint8))
            except Exception as e:
                print(f"Warning: Failed to load {tile_path}: {e}")

        self.tiles = np.stack(tile_list, axis=0)
        self.tile_colors = self.tiles.mean(axis=(1, 2))   # vectorized (N, 3)

        # Clear old cache
        self.resized_tiles_cache.clear()

    # ---------------------------------------------------------
    # Resizing (cached)
    # ---------------------------------------------------------

    def get_resized_tiles(self, cell_h: int, cell_w: int) -> np.ndarray:
        """
        Returns cached resized tiles for the given cell size.

        Parameters
        ----------
        cell_h : int
            Height of mosaic cell.

        cell_w : int
            Width of mosaic cell.

        Returns
        -------
        np.ndarray
            Resized tile set of shape (N, cell_h, cell_w, 3).
        """

        key = (cell_h, cell_w)
        if key in self.resized_tiles_cache:
            return self.resized_tiles_cache[key]

        num_tiles = self.tiles.shape[0]
        resized = np.empty((num_tiles, cell_h, cell_w, 3), dtype=np.uint8)

        for idx in range(num_tiles):
            img = Image.fromarray(self.tiles[idx])
            img = img.resize((cell_w, cell_h), RESIZE_INTERPOLATION)
            resized[idx] = np.array(img, dtype=np.uint8)

        self.resized_tiles_cache[key] = resized
        return resized

    # ---------------------------------------------------------
    # Tile Feature Access
    # ---------------------------------------------------------

    def get_tile_colors(self) -> np.ndarray:
        """Return tile average RGB colors (N, 3)."""
        return self.tile_colors

    def get_tile_count(self) -> int:
        """Return number of loaded tiles."""
        return self.tiles.shape[0]