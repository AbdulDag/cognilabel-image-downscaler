"""
Cognilabel Image Resolution Reducer
====================================
Batch-reduces image resolution by ~10× (in megapixels) while preserving
aspect ratio and visual quality.

Algorithm
---------
To reduce total megapixels by a factor of 10 while keeping aspect ratio:
  - Area scales as width × height, so reducing area by 10× means each
    linear dimension must be divided by √10 ≈ 3.162.
  - new_width  = round(original_width  / √10)
  - new_height = round(original_height / √10)
This guarantees the output has ~1/10 the pixels of the original.
"""

import argparse
import math
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageFile

# ---------------------------------------------------------------------------
# Safety: allow Pillow to open very large images without a DecompressionBomb
# warning, and tolerate truncated files gracefully.
# ---------------------------------------------------------------------------
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Supported extensions (case-insensitive check performed at runtime)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

# Scale factor: 1 / √10  (reduces area by 10×)
LINEAR_SCALE_FACTOR = 1.0 / math.sqrt(10)


def setup_logging(log_dir: str) -> logging.Logger:
    """Configure a logger that writes to both stdout and a timestamped file."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"reduction_{timestamp}.log")

    logger = logging.getLogger("cognilabel_reducer")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_fmt)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def compute_new_dimensions(width: int, height: int) -> tuple[int, int]:
    """Return (new_width, new_height) that yield ~1/10 the original megapixels."""
    new_width = max(1, round(width * LINEAR_SCALE_FACTOR))
    new_height = max(1, round(height * LINEAR_SCALE_FACTOR))
    return new_width, new_height


def megapixels(width: int, height: int) -> float:
    """Return the megapixel count for the given dimensions."""
    return (width * height) / 1_000_000


def reduce_image(input_path: str, output_path: str, logger: logging.Logger) -> None:
    """Open *input_path*, downscale by ~10× MP, and save to *output_path*."""
    with Image.open(input_path) as img:
        orig_w, orig_h = img.size
        new_w, new_h = compute_new_dimensions(orig_w, orig_h)

        orig_mp = megapixels(orig_w, orig_h)
        new_mp = megapixels(new_w, new_h)

        filename = os.path.basename(input_path)

        logger.info(f"filename: {filename}")
        logger.info(f"  original_resolution: {orig_w} x {orig_h} ({orig_mp:.0f} MP)")
        logger.info(f"  new_resolution:      {new_w} x {new_h} ({new_mp:.1f} MP)")
        logger.info("")

        # Downscale with high-quality LANCZOS resampling
        resized = img.resize((new_w, new_h), Image.LANCZOS)

        # Preserve original format; fall back to PNG
        save_format = img.format or "PNG"
        save_kwargs: dict = {}
        if save_format.upper() in ("JPEG", "JPG"):
            save_kwargs["quality"] = 90
            save_kwargs["optimize"] = True
            # JPEG cannot save RGBA — convert if necessary
            if resized.mode in ("RGBA", "P"):
                resized = resized.convert("RGB")

        resized.save(output_path, format=save_format, **save_kwargs)


def process_directory(input_dir: str, output_dir: str, logger: logging.Logger) -> int:
    """Walk *input_dir*, reduce every supported image, and save to *output_dir*.

    Returns the number of images successfully processed.
    """
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for entry in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(entry)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        input_path = os.path.join(input_dir, entry)
        if not os.path.isfile(input_path):
            continue

        output_path = os.path.join(output_dir, entry)

        try:
            reduce_image(input_path, output_path, logger)
            count += 1
        except Exception as exc:
            logger.error(f"Failed to process {entry}: {exc}")

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cognilabel Image Resolution Reducer — "
        "batch-reduce image resolution by ~10× (megapixels).",
    )
    parser.add_argument(
        "--input",
        default="data/input_images",
        help="Path to the folder containing source images (default: data/input_images)",
    )
    parser.add_argument(
        "--output",
        default="data/output_images",
        help="Path to the folder for reduced images (default: data/output_images)",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for log files (default: logs)",
    )

    args = parser.parse_args()
    logger = setup_logging(args.log_dir)

    logger.info("=" * 60)
    logger.info("Cognilabel Image Resolution Reducer")
    logger.info("=" * 60)
    logger.info(f"Input directory:  {os.path.abspath(args.input)}")
    logger.info(f"Output directory: {os.path.abspath(args.output)}")
    logger.info(f"Reduction factor: ~10× (linear scale ≈ {LINEAR_SCALE_FACTOR:.4f})")
    logger.info("")

    if not os.path.isdir(args.input):
        logger.error(f"Input directory does not exist: {args.input}")
        sys.exit(1)

    count = process_directory(args.input, args.output, logger)

    logger.info("-" * 60)
    logger.info(f"Done. {count} image(s) processed.")


if __name__ == "__main__":
    main()
