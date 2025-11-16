# LAB 5: OPTIMIZED & MODULARIZED IMAGE MOSAIC GENERATOR

## Overview

This project is the optimized and modularized version of the Lab 1 Image Mosaic Generator.
The original Lab 1 code was intentionally inefficient, using nested loops, repeated computations, and slow per-pixel processing.
In this Lab 5 version, the entire system has been:
- Profiled
- Optimized using NumPy vectorization
- Refactored into a clean modular architecture
- Structured like a real Python package
- Improved with a Gradio-based user interface

This implementation creates a mosaic representation of an image by:
- Preprocessing and resizing the input image
- Dividing the image into a grid using vectorized operations
- Computing average colors for each grid cell
- Matching each grid cell to the closest tile based on color similarity
- Constructing the final mosaic efficiently

This version achieves 20× to 100× speedup over the inefficient Lab 1 reference.

---

## Getting Started

### Prerequisites:
- Python 3.10+

### Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/cs5130-lab5-mosaic-modular.git
cd cs5130-lab5-mosaic-modular
```
### Set up virtual environment:

```
python3 -m venv venv
source venv/bin/activate    (macOS/Linux)
venv\Scripts\activate       (Windows)
````
### Install dependencies:

```
pip install -r requirements.txt
```

### Run the Gradio App:
```
python app.py
```

The Gradio UI will launch at:
```
http://localhost:7860
```
---

## Usage

Using the Gradio Interface:
- Either upload an image or click one of the sample images.
- Choose the grid size (8×8 to 64×64).
- Generate the mosaic.
- View the mosaic image and MSE value.

Command-Line Usage (without UI):

```
from mosaic_generator.tile_manager import TileManager
from mosaic_generator.image_processor import preprocess_image
from mosaic_generator.mosaic_builder import MosaicBuilder
from PIL import Image

tile_manager = TileManager(“mosaic_generator/tiles”)
builder = MosaicBuilder(tile_manager)

img = Image.open(“your_image.jpg”)
processed = preprocess_image(img)
mosaic = builder.create_mosaic(processed)
```

This produces the mosaic as a NumPy array, ready for saving or evaluation.

### Running the Tests:
```
python -m pytest tests/test_mosaic.py
```
---

## Repository Structure

```
cs5130-mosaic-generator-modular/
├── app.py                         (Gradio interface)
├── requirements.txt
├── README.txt
├── mosaic_generator/
│   ├── init.py
│   ├── config.py                  (constants)
│   ├── image_processor.py         (loading, preprocessing, grid extraction)
│   ├── tile_manager.py            (tile loading, caching, color extraction)
│   ├── mosaic_builder.py          (mosaic creation engine)
│   ├── metrics.py                 (MSE metric)
│   ├── utils.py                   (helper functions)
│   ├── tiles/                     (tile images)
│   └── sample_images/             (sample inputs)
└── tests/
└── test_mosaic.py
```
---

## Inefficiencies Resolved
- Grid Extraction Loops
    - Old Lab 1 code used nested loops to slice every cell.
	- New version uses NumPy reshape + transpose for instant extraction.
- Manual Pixel-by-Pixel Average Color
	- Old version iterated through each pixel and manually summed values.
	- New version uses vectorized np.mean(), eliminating loops entirely.
- Tile Matching One-by-One
	- Old version computed Euclidean distances inside a Python loop.
	- New version uses broadcasting and full-array vectorized distance calculation.
- Repeated Tile Resizing
    - Old version resized tiles repeatedly during mosaic creation.
	- New version resizes tiles once per grid size and caches them.
- Repeated Tile Loading
	- Old version loaded tiles from disk each time.
	- New version loads tiles once and keeps them in memory.
- Mixed Logic in One File
	- Old version had monolithic code structure.
	- New version has a clean modular package, improving readability and maintainability.

---

## Performance Improvements

Using the original inefficient Lab 1 version:
- 512×512 image, 32×32 grid took ~30–60 seconds

Using the optimized Lab 5 modular version:
- Same configuration runs in 0.003–0.004 seconds

Actual measured values from test runs:
```
Image Size | Grid Size | Lab 1 Time | Lab 5 Time | Speedup
256×256    | 16×16     | ~5–10 sec  | 0.0025 sec | ~3000×
512×512    | 32×32     | ~30–60 sec | 0.0034 sec | ~15,000×
1024×1024  | 64×64     | ~2–5 min   | 0.0049 sec | ~24,000×
```
Performance insights:
- Vectorization removed 90–98% of processing overhead.
- Caching removed all repeated resizing and loading costs.
- Grid extraction is now O(1) instead of O(n²).

---

## Troubleshooting

1. **Issue:** ModuleNotFoundError

**Solution:** Make sure your virtual environment is activated and dependencies installed.

2. **Issue:** Gradio doesn’t launch

**Solution:** Try another port: python app.py –server_port 7861

3. **Issue:** Tiles not found

**Solution:** Ensure your tile images are inside: mosaic_generator/tiles/

4. **Issue:** Output mosaic is black

**Solution:** Confirm tiles are valid PNG/JPG images. Ensure they are RGB images (not RGBA).

