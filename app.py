"""
Gradio App for Modular Mosaic Generator
"""

import gradio as gr
from pathlib import Path
import numpy as np
from PIL import Image

from mosaic_generator.image_processor import (
    load_image,
    preprocess_image,
    extract_cells_and_colors,
)
from mosaic_generator.tile_manager import TileManager
from mosaic_generator.mosaic_builder import MosaicBuilder
from mosaic_generator.metrics import compute_mse
from mosaic_generator.config import DEFAULT_GRID_SIZE


def create_gradio_interface():

    # Load tiles
    tile_manager = TileManager(tile_directory="mosaic_generator/tiles")
    mosaic_builder = MosaicBuilder(tile_manager)

    # Load example images
    example_dir = Path("mosaic_generator/sample_images")
    example_paths = []
    if example_dir.exists():
        for f in sorted(example_dir.iterdir()):
            if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                example_paths.append(str(f))

    def process_image(input_img, grid_size):

        if input_img is None:
            return None, "Please upload or select an image."

        # Preprocess
        processed = preprocess_image(input_img)

        # Build mosaic
        mosaic = mosaic_builder.create_mosaic(processed)

        # Compute error metric
        mse = compute_mse(processed, mosaic)

        msg = f"Mosaic generated.\nGrid: {grid_size}×{grid_size}\nMSE: {mse:.2f}"
        return mosaic, msg

    with gr.Blocks(title="Modular Optimized Mosaic Generator") as demo:
        gr.Markdown("# 🎨 Modular Optimized Mosaic Generator")
        gr.Markdown("Using fully vectorized backend + modular architecture.")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")

                if example_paths:
                    gr.Markdown("### Example Images")
                    gr.Examples(
                        examples=example_paths,
                        inputs=input_image,
                        label="Click a sample image"
                    )

                grid_size = gr.Slider(8, 64, value=32, step=8, label="Grid Size")
                generate_btn = gr.Button("Generate Mosaic")

            with gr.Column():
                output_image = gr.Image(label="Mosaic Output")
                output_text = gr.Textbox(label="Status", lines=3)

        generate_btn.click(
            fn=process_image,
            inputs=[input_image, grid_size],
            outputs=[output_image, output_text],
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()