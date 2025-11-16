"""
mosaic_builder.py

Main logic for constructing the mosaic using vectorized grid extraction
and vectorized tile matching. Uses TileManager and image_processor utilities.
"""

import numpy as np
from . import image_processor
from .config import DEFAULT_GRID_SIZE


class MosaicBuilder:
    """
    High-performance mosaic builder using vectorized NumPy operations.
    """

    def __init__(self, tile_manager):
        """
        Parameters
        ----------
        tile_manager : TileManager
            Manages tile loading, caching, and feature extraction.
        """
        self.tile_manager = tile_manager

    # ------------------------------------------------------------------
    # Internal vectorized helpers
    # ------------------------------------------------------------------

    def _match_cells_to_tiles(self, cell_colors_flat: np.ndarray) -> np.ndarray:
        """
        Vectorized nearest-tile matching using Euclidean distance.

        Parameters
        ----------
        cell_colors_flat : np.ndarray
            Shape (num_cells, 3) array with average RGB for each grid cell.

        Returns
        -------
        np.ndarray
            Tile indices mapped 1-to-1 to grid cells: shape (num_cells,)
        """

        # ✅ FIXED: Use tile_manager.tile_colors (correct attribute)
        tile_colors = self.tile_manager.tile_colors

        diffs = cell_colors_flat[:, None, :] - tile_colors[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)

        return np.argmin(dists, axis=1)

    # ------------------------------------------------------------------
    # Public mosaic creation method
    # ------------------------------------------------------------------

    def create_mosaic(self, image_array: np.ndarray, grid_size=DEFAULT_GRID_SIZE) -> np.ndarray:
        """
        Create a mosaic from the input image using vectorized operations.

        Parameters
        ----------
        image_array : np.ndarray
            Preprocessed RGB image of shape (H, W, 3).

        grid_size : tuple(int, int)
            Number of (rows, columns) in the mosaic grid.

        Returns
        -------
        np.ndarray
            Final mosaic image, same shape as input (H, W, 3).
        """

        # Extract grid cells + average colors
        cells, cell_colors = image_processor.extract_cells_and_colors(image_array, grid_size)

        # Flatten for vector matching
        num_cells = grid_size[0] * grid_size[1]
        cell_colors_flat = cell_colors.reshape(num_cells, 3)

        # Vectorized nearest-tile assignment
        best_tile_indices = self._match_cells_to_tiles(cell_colors_flat)

        # Resize tiles for this grid cell size
        grid_h, grid_w, cell_h, cell_w = image_processor.compute_grid_shapes(image_array, grid_size)
        resized_tiles = self.tile_manager.get_resized_tiles(cell_h, cell_w)

        # Prepare mosaic array
        mosaic = np.zeros_like(image_array, dtype=np.uint8)

        # Reshape mosaic to match tile block layout
        mosaic_cells = (
            mosaic
            .reshape(grid_h, cell_h, grid_w, cell_w, 3)
            .transpose(0, 2, 1, 3, 4)
        )

        # Map best tile index → resized tile
        tiles_per_cell = resized_tiles[best_tile_indices]
        tiles_per_cell = tiles_per_cell.reshape(grid_h, grid_w, cell_h, cell_w, 3)

        mosaic_cells[...] = tiles_per_cell

        return mosaic