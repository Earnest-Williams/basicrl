from PIL import Image
import sys
import os

def create_color_map(image_path):
    """
    Reads a PNG image and creates a pixel-by-pixel color map.

    Args:
        image_path (str): The path to the input PNG image file.

    Returns:
        list: A nested list representing the color map. Each inner list is a row,
              and each element in the inner list is an RGBA tuple (R, G, B, A).
        None: Returns None if the file cannot be opened or is not a valid image.
    """
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}", file=sys.stderr)
        return None

    try:
        img = Image.open(image_path)
        img = img.convert("RGBA")
        width, height = img.size
        pixel_data = img.load()

        color_map = []
        print(f"Reading image: {image_path} ({width}x{height} pixels)")

        for y in range(height):
            row = []
            for x in range(width):
                color_tuple = pixel_data[x, y]
                row.append(color_tuple)
            color_map.append(row)

        print("Color map created successfully.")
        return color_map

    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.", file=sys.stderr)
        return None
    except IOError:
        print(f"Error: Could not open or read the file '{image_path}'. It might not be a valid image file.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return None

def save_color_map_to_file(color_map, output_file_path):
    """
    Saves the generated color map to a text file.

    Args:
        color_map (list): The nested list representing the color map.
        output_file_path (str): The path to save the text file.
    """
    if not color_map:
        print("Error: Color map is empty, cannot save.", file=sys.stderr)
        return

    try:
        with open(output_file_path, 'w') as f:
            print(f"Saving color map to: {output_file_path}")
            for y, row in enumerate(color_map):
                row_str = "\t".join(map(str, row))
                f.write(f"{row_str}\n")
            print("Color map saved successfully.")
    except IOError:
        print(f"Error: Could not write to file '{output_file_path}'.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        input_image_path = sys.argv[1]
    else:
        print("Usage: python color_test.py <path_to_image.png>", file=sys.stderr)
        sys.exit(1)

    output_text_file = 'color_map.txt'

    pixel_map = create_color_map(input_image_path)

    if pixel_map:
        print("\n--- Sample Pixels (Top-Left 5x5) ---")
        height_limit = min(5, len(pixel_map))
        for i in range(height_limit):
            if i < len(pixel_map):
                width_limit = min(5, len(pixel_map[i]))
                row_sample = [str(p) for p in pixel_map[i][:width_limit]]
                print(f"Row {i}: {' '.join(row_sample)}")

        if output_text_file:
            save_color_map_to_file(pixel_map, output_text_file)

        try:
            target_y, target_x = 20, 10
            if target_y < len(pixel_map) and target_x < len(pixel_map[target_y]):
                example_pixel = pixel_map[target_y][target_x]
                print(f"\nPixel color at (x={target_x}, y={target_y}): {example_pixel}")
            else:
                print(f"\nCould not access example pixel ({target_x}, {target_y}) - image might be too small or indices out of bounds.")
        except IndexError:
            print(f"\nCould not access example pixel ({target_x}, {target_y}) - Index Error.")
    else:
        print("\nCould not create or process the pixel map.")
