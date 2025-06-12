import os
from PIL import Image
from pathlib import Path
from rembg import remove
import sys


def setup_folders(base_path):
    """Create output folder structure and return the base path"""
    base_path = Path(base_path)
    for split in ['train', 'test']:
        for gender in ['man', 'women']:  # Changed to match your folder names
            new_folder = base_path.parent / f"{base_path.name}_nobg" / split / gender
            new_folder.mkdir(parents=True, exist_ok=True)
            print(f"Created folder: {new_folder}")
    return base_path.parent / f"{base_path.name}_nobg"


def remove_bg_and_save(input_path, output_path):
    """Process single image and handle errors"""
    try:
        with open(input_path, "rb") as inp:
            img = Image.open(inp)
            print(f"Processing: {input_path} (Size: {img.size})")
            output = remove(img)

            # Convert to RGB if RGBA (for better compatibility)
            if output.mode == 'RGBA':
                output = output.convert('RGB')

            output.save(output_path, "PNG")
            print(f"Successfully saved: {output_path}")
            return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}", file=sys.stderr)
        return False


def process_dataset(original_path="dataset"):
    """Main processing function with enhanced logging"""
    original_path = Path(original_path)
    if not original_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {original_path}")

    new_base = setup_folders(original_path)
    print(f"Output base: {new_base}")

    processed_count = 0
    error_count = 0

    for split in ['train', 'test']:
        for gender in ['man', 'women']:  # Changed to match your folder names
            input_dir = original_path / split / gender
            output_dir = new_base / split / gender

            if not input_dir.exists():
                print(f"Warning: Input directory not found: {input_dir}")
                continue

            print(f"\nProcessing: {input_dir}")

            for img_file in input_dir.glob('*.*'):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue

                output_path = output_dir / f"{img_file.stem}.png"

                if remove_bg_and_save(img_file, output_path):
                    processed_count += 1
                else:
                    error_count += 1

    print(f"\nProcessing complete!")
    print(f"Total processed: {processed_count}")
    print(f"Total errors: {error_count}")


if __name__ == "__main__":
    process_dataset()