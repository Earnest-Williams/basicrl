# === engine/tileset_loader.py ===
import io
from pathlib import Path

import numpy as np
import structlog  # Added
from cairosvg import svg2png  # For SVG rasterization
from PIL import Image

log = structlog.get_logger()  # Added


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
    log.debug("Rasterizing SVG", path=str(svg_path), width=width, height=height)
    try:
        png_bytes = svg2png(url=str(svg_path), output_width=width, output_height=height)
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        return img  # SVGs already have correct transparency
    except Exception as e:
        log.error(
            "Failed to rasterize SVG", path=str(svg_path), error=str(e), exc_info=True
        )
        # Return a blank placeholder image on error? Or re-raise?
        # Returning placeholder to avoid crashing load_tiles entirely.
        return Image.new(
            "RGBA", (width, height), (255, 0, 255, 255)
        )  # Magenta placeholder


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
    log.info("Loading tileset", path=folder, width=tile_width, height=tile_height)
    path = Path(folder)
    if not path.is_dir():
        # Log error before raising
        log.error("Invalid tileset folder path", path=str(path))
        raise ValueError(f"Invalid tileset folder: {folder}")

    tiles = {}
    is_svg = False
    png_count = 0
    svg_count = 0
    svg_paths = {}  # Store paths for second pass

    # First pass: collect all files, process PNGs
    try:
        for file in path.iterdir():
            if not file.is_file():
                continue

            try:
                tile_id = int(
                    file.stem.split("_")[-1]
                )  # Assumes format like name_ID.ext
            except (ValueError, IndexError):
                log.warning(
                    "Skipping file with unexpected name format", filename=file.name
                )
                continue

            if file.suffix.lower() == ".png":
                try:
                    img = Image.open(file).convert("RGBA")
                    cleaned_img = clean_tile_background(img)
                    resized_img = cleaned_img.resize(
                        (tile_width, tile_height), Image.Resampling.NEAREST
                    )
                    tiles[tile_id] = resized_img
                    png_count += 1
                except Exception as e:
                    log.warning(
                        "Failed to process PNG tile", filename=file.name, error=str(e)
                    )

            elif file.suffix.lower() == ".svg":
                svg_paths[tile_id] = file
                is_svg = True
                svg_count += 1
            # Else: ignore other file types silently or log debug message

    except Exception as e:
        log.error(
            "Error iterating tileset directory",
            path=str(path),
            error=str(e),
            exc_info=True,
        )
        # Re-raise or return empty dict? Returning empty seems safer.
        return {}, False

    log.debug("Initial file scan complete", png_found=png_count, svg_found=svg_count)

    # Second pass: rasterize SVGs if present
    if is_svg:
        log.info("Rasterizing SVG tiles...")
        rasterized_count = 0
        for tile_id, svg_path in svg_paths.items():
            rasterized = rasterize_svg(svg_path, tile_width, tile_height)
            if rasterized:  # Check if rasterization returned an image
                tiles[tile_id] = rasterized
                rasterized_count += 1
        log.info("SVG rasterization complete", count=rasterized_count)

    log.info(
        "Tileset loading finished",
        path=folder,
        total_tiles=len(tiles),
        png_count=png_count,
        svg_count=svg_count,
    )
    return tiles, is_svg
