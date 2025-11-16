import unittest
import numpy as np
from pathlib import Path
from PIL import Image

from mosaic_generator.image_processor import ImageProcessor
from mosaic_generator.tile_manager import TileManager
from mosaic_generator.mosaic_builder import MosaicBuilder
from mosaic_generator.metrics import compute_mse
from mosaic_generator.config import DEFAULT_TARGET_SIZE, DEFAULT_GRID_SIZE


class TestMosaicGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure tiles exist
        cls.tile_dir = Path("tiles")
        if not cls.tile_dir.exists():
            raise RuntimeError("Tiles folder missing. Add tiles before running tests.")

        cls.tile_manager = TileManager(tile_directory=str(cls.tile_dir))
        cls.processor = ImageProcessor()
        cls.builder = MosaicBuilder(cls.tile_manager)

        # Create a small synthetic test image (RGB gradient)
        cls.test_image = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            cls.test_image[i, :, 0] = i        # R gradient
            cls.test_image[:, i, 1] = i        # G gradient
        # Blue stays 0

    def test_tile_loading(self):
        """Confirm tiles load correctly and average colors computed."""
        tm = self.tile_manager

        self.assertGreater(len(tm.tiles), 0, "No tiles loaded.")
        self.assertEqual(tm.tiles.shape[1:], (32, 32, 3))
        self.assertEqual(len(tm.tile_colors), tm.tiles.shape[0])

    def test_preprocessing(self):
        """Ensure preprocessing resizes correctly."""
        processed = self.processor.preprocess_image(self.test_image)
        self.assertEqual(
            processed.shape,
            (DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE, 3)
        )

    def test_grid_extraction(self):
        """Check extraction of grid dimensions."""
        processed = self.processor.preprocess_image(self.test_image)
        grid_h, grid_w, cell_h, cell_w = self.processor.compute_grid_shapes(
            processed,
            DEFAULT_GRID_SIZE
        )

        self.assertEqual(grid_h, DEFAULT_GRID_SIZE[0])
        self.assertEqual(grid_w, DEFAULT_GRID_SIZE[1])
        self.assertEqual(cell_h, DEFAULT_TARGET_SIZE // grid_h)
        self.assertEqual(cell_w, DEFAULT_TARGET_SIZE // grid_w)

    def test_mosaic_generation(self):
        """Check mosaic output shape matches input shape."""
        processed = self.processor.preprocess_image(self.test_image)
        mosaic = self.builder.create_mosaic(processed, DEFAULT_GRID_SIZE)

        self.assertEqual(mosaic.shape, processed.shape)
        self.assertTrue(np.any(mosaic != 0))

    def test_mse(self):
        """Ensure MSE computation works."""
        processed = self.processor.preprocess_image(self.test_image)
        mosaic = self.builder.create_mosaic(processed, DEFAULT_GRID_SIZE)

        mse = compute_mse(processed, mosaic)
        self.assertIsInstance(mse, float)
        self.assertGreaterEqual(mse, 0.0)


if __name__ == "__main__":
    unittest.main()