# === engine/tileset_loader.py ===
from pathlib import Path
from PIL import Image
import io
import numpy as np
from cairosvg import svg2png  # For SVG rasterization


def clean_tile_background(img: Image.Image) -> Image.Image:
    """Clean PNG background color (21,21,21) to transparent."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    data = np.array(img)

    # Only wipe pixels matching exact (21,21,21) background color
    mask = np.all(data[:, :, :3] == (21, 21, 21), axis=2)
    data[mask, 3] = 0  # Set alpha to 0 (transparent)

    return Image.fromarray(data, "RGBA")


def rasterize_svg(svg_path: Path, width: int, height: int) -> Image.Image:
    """Convert an SVG file to a PIL Image at the specified size."""
    png_bytes = svg2png(url=str(svg_path), output_width=width, output_height=height)
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    return img  # SVGs already have correct transparency


def load_tiles(
    folder: str, tile_width: int, tile_height: int
) -> tuple[dict[int, Image.Image], bool]:
    """
    Loads tiles from a folder of PNG or SVG files.
    PNGs are cleaned and resized.
    SVGs are rasterized to correct size.

    Returns:
        tiles: A dictionary {tile_index: PIL Image}
        is_svg: Whether any SVGs were found
    """
    path = Path(folder)
    if not path.is_dir():
        raise ValueError(f"Invalid tileset folder: {folder}")

    tiles = {}
    is_svg = False
    svg_paths = {}

    # First pass: collect all files
    for file in path.iterdir():
        if file.suffix.lower() == ".png":
            tile_id = int(file.stem.split("_")[-1])
            img = Image.open(file).convert("RGBA")
            cleaned_img = clean_tile_background(img)
            cleaned_img = cleaned_img.resize(
                (tile_width, tile_height), Image.Resampling.NEAREST
            )
            tiles[tile_id] = cleaned_img
        elif file.suffix.lower() == ".svg":
            tile_id = int(file.stem.split("_")[-1])
            svg_paths[tile_id] = file
            is_svg = True

    # Second pass: rasterize SVGs if present
    if is_svg:
        for tile_id, svg_path in svg_paths.items():
            rasterized = rasterize_svg(svg_path, tile_width, tile_height)
            tiles[tile_id] = rasterized

    return tiles, is_svg
