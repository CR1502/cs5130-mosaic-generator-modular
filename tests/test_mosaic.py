import unittest
import numpy as np
from pathlib import Path

from mosaic_generator import image_processor
from mosaic_generator.tile_manager import TileManager
from mosaic_generator.mosaic_builder import MosaicBuilder
from mosaic_generator.metrics import compute_mse
from mosaic_generator.config import DEFAULT_TARGET_SIZE, DEFAULT_GRID_SIZE


class TestMosaicGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Resolve root of project (folder containing mosaic_generator/)
        project_root = Path(__file__).resolve().parents[1]

        # Correct location of tiles folder inside the package
        cls.tile_dir = project_root / "mosaic_generator" / "tiles"
        if not cls.tile_dir.exists():
            raise RuntimeError(f"Tiles folder missing. Expected: {cls.tile_dir}")

        # Initialize managers
        cls.tile_manager = TileManager(tile_directory=str(cls.tile_dir))
        cls.processor = image_processor  # module, not class
        cls.builder = MosaicBuilder(cls.tile_manager)

        # Create synthetic 256×256 gradient test image
        cls.test_image = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            cls.test_image[i, :, 0] = i
            cls.test_image[:, i, 1] = i

    def test_tile_loading(self):
        """Ensure tiles load and average colors compute correctly."""
        tm = self.tile_manager

        self.assertGreater(len(tm.tiles), 0, "No tiles loaded.")
        self.assertEqual(tm.tiles.shape[1:], (32, 32, 3))
        self.assertEqual(len(tm.tile_colors), tm.tiles.shape[0])

    def test_preprocessing(self):
        """Ensure preprocessing resizes to DEFAULT_TARGET_SIZE."""
        processed = self.processor.preprocess_image(self.test_image)
        self.assertEqual(
            processed.shape,
            (DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE, 3)
        )

    def test_grid_extraction(self):
        """Ensure grid extraction computes correct shapes."""
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
        """Ensure mosaic has same shape and contains non-zero pixels."""
        processed = self.processor.preprocess_image(self.test_image)
        mosaic = self.builder.create_mosaic(processed)

        self.assertEqual(mosaic.shape, processed.shape)
        self.assertTrue(np.any(mosaic != 0), "Mosaic should not be all zeros.")

    def test_mse(self):
        """Ensure MSE computes correctly and returns a float."""
        processed = self.processor.preprocess_image(self.test_image)
        mosaic = self.builder.create_mosaic(processed)

        mse = compute_mse(processed, mosaic)

        self.assertIsInstance(mse, float)
        self.assertGreaterEqual(mse, 0.0)


if __name__ == "__main__":
    unittest.main()