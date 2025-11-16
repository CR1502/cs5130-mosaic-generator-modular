"""
mosaic_builder.py

Main mosaic construction logic. This module takes preprocessed images,
grid structures from image_processor, and tile data from tile_manager
to assemble the final optimized mosaic using NumPy vectorization.
"""

import numpy as np
from .tile_manager import TileManager
from .image_processor import (
    preprocess_image,
    compute_grid_shapes,
    extract_cells_and_colors,
)
from .config import DEFAULT_GRID_SIZE


class MosaicBuilder:
    """
    Builds a high-performance mosaic using vectorized operations.
    """

    def __init__(self, tile_manager: TileManager, grid_size=DEFAULT_GRID_SIZE):
        self.tile_manager = tile_manager
        self.grid_size = grid_size

    # ---------------------------------------------------------
    # Tile Matching Logic
    # ---------------------------------------------------------

    def _match_cells_to_tiles(self, cell_colors_flat: np.ndarray) -> np.ndarray:
        """
        Vectorized nearest-tile matching using Euclidean distance.

        Parameters
        ----------
        cell_colors_flat : np.ndarray
            Array of shape (num_cells, 3) with average RGB per cell.

        Returns
        -------
        np.ndarray
            Tile indices of shape (num_cells,)
        """

        tile_colors = self.tile_manager.get_tile_colors()

        # Broadcasting: (C,1,3) - (1,T,3) → (C,T,3)
        diffs = cell_colors_flat[:, None, :] - tile_colors[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)  # (C,T)
        return np.argmin(dists, axis=1)

    # ---------------------------------------------------------
    # Mosaic Construction
    # ---------------------------------------------------------

    def create_mosaic(self, image):
        """
        Full mosaic creation pipeline.

        Parameters
        ----------
        image : PIL.Image or np.ndarray
            Original input image.

        Returns
        -------
        np.ndarray
            Final mosaic as an array of shape (H,W,3)
        """

        # Step 1: preprocess image (resize + RGB)
        processed = preprocess_image(image)

        # Step 2: identify grid shapes
        grid_h, grid_w, cell_h, cell_w = compute_grid_shapes(
            processed, self.grid_size
        )

        # Step 3: extract image cells + their average colors
        _, cell_colors = extract_cells_and_colors(processed, self.grid_size)

        num_cells = grid_h * grid_w
        cell_colors_flat = cell_colors.reshape(num_cells, 3)

        # Step 4: vectorized best-tile matching
        best_tile_indices = self._match_cells_to_tiles(cell_colors_flat)

        # Step 5: retrieve cached resized tiles
        resized_tiles = self.tile_manager.get_resized_tiles(cell_h, cell_w)

        # Step 6: allocate output and reshape for easy placement
        mosaic = np.zeros_like(processed, dtype=np.uint8)

        mosaic_cells = mosaic.reshape(
            grid_h, cell_h, grid_w, cell_w, 3
        ).transpose(0, 2, 1, 3, 4)

        # Step 7: place tiles using vectorization
        tiles_per_cell = resized_tiles[best_tile_indices]
        tiles_per_cell = tiles_per_cell.reshape(
            grid_h, grid_w, cell_h, cell_w, 3
        )

        mosaic_cells[...] = tiles_per_cell

        # Step 8: restore original image layout
        return mosaic_cells.transpose(0, 2, 1, 3, 4).reshape(processed.shape)