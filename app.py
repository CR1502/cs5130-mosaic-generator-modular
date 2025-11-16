"""
app.py

Gradio interface for the modularized, optimized mosaic generator.
Connects the ImageProcessor, TileManager, MosaicBuilder, and metrics
modules into a unified UI.
"""

import gradio as gr
from pathlib import Path

from mosaic_generator.config import DEFAULT_GRID_SIZE
from mosaic_generator.image_processor import ImageProcessor
from mosaic_generator.tile_manager import TileManager
from mosaic_generator.mosaic_builder import MosaicBuilder
from mosaic_generator.metrics import compute_mse


# ---------------------------------------------------------
# Initialize Core Components
# ---------------------------------------------------------

tile_manager = TileManager(tile_directory="tiles")
processor = ImageProcessor()
builder = MosaicBuilder(tile_manager)


# ---------------------------------------------------------
# Gradio Processing Function
# ---------------------------------------------------------

def process_image(input_image, grid_size):
    """
    Core function called by Gradio.
    """
    try:
        grid = (grid_size, grid_size)
        proc_img = processor.preprocess_image(input_image)
        mosaic = builder.create_mosaic(proc_img, grid)
        mse = compute_mse(proc_img, mosaic)

        message = f"Mosaic created.\nGrid: {grid_size}×{grid_size}\nMSE: {mse:.2f}"
        return mosaic, message

    except Exception as e:
        return None, f"Error: {str(e)}"


# ---------------------------------------------------------
# Load Example Images
# ---------------------------------------------------------

def get_examples():
    example_dir = Path("sample_images")
    if not example_dir.exists():
        return []
    paths = []
    for f in sorted(example_dir.iterdir()):
        if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            paths.append(str(f))
    return paths


example_paths = get_examples()


# ---------------------------------------------------------
# Build Gradio Interface
# ---------------------------------------------------------

def create_interface():
    with gr.Blocks(title="Optimized Modular Image Mosaic Generator") as demo:

        gr.Markdown("# 🎨 Optimized Modular Mosaic Generator")
        gr.Markdown("Built with NumPy vectorization and modular architecture.")

        with gr.Row():
            with gr.Column():

                input_image = gr.Image(type="pil", label="Input Image")

                # Example Section
                if example_paths:
                    gr.Markdown("### Example Images")
                    gr.Examples(
                        examples=example_paths,
                        inputs=input_image,
                        label="Click to load",
                    )

                grid_slider = gr.Slider(
                    minimum=8, maximum=64, step=8,
                    value=DEFAULT_GRID_SIZE[0],
                    label="Grid Size",
                )

                generate_btn = gr.Button("Generate Mosaic")

            with gr.Column():
                output_image = gr.Image(label="Mosaic Output")
                output_text = gr.Textbox(label="Status", lines=3)

        generate_btn.click(
            fn=process_image,
            inputs=[input_image, grid_slider],
            outputs=[output_image, output_text]
        )

    return demo


if __name__ == "__main__":
    app = create_interface()
    app.launch()