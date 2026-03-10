"""
Tests for the Cognilabel Image Resolution Reducer.

Creates synthetic images, runs the reducer, and validates output dimensions.
"""

import math
import os
import tempfile

import pytest
from PIL import Image

# Add src to the import path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reduce_resolution import (
    LINEAR_SCALE_FACTOR,
    compute_new_dimensions,
    megapixels,
    process_directory,
    setup_logging,
)


# ── Unit tests for helper functions ──────────────────────────────────────────


class TestComputeNewDimensions:
    """Verify that compute_new_dimensions correctly scales by 1/√10."""

    def test_basic_scale(self):
        new_w, new_h = compute_new_dimensions(4000, 3000)
        # Expected: 4000/√10 ≈ 1265, 3000/√10 ≈ 949
        assert new_w == round(4000 * LINEAR_SCALE_FACTOR)
        assert new_h == round(3000 * LINEAR_SCALE_FACTOR)

    def test_megapixel_ratio(self):
        """Output area should be ~1/10 of the input area."""
        orig_w, orig_h = 4000, 3000
        new_w, new_h = compute_new_dimensions(orig_w, orig_h)
        ratio = (new_w * new_h) / (orig_w * orig_h)
        assert abs(ratio - 0.1) < 0.01  # within 1%

    def test_preserves_aspect_ratio(self):
        orig_w, orig_h = 6000, 4000
        new_w, new_h = compute_new_dimensions(orig_w, orig_h)
        orig_ar = orig_w / orig_h
        new_ar = new_w / new_h
        assert abs(orig_ar - new_ar) < 0.01

    def test_small_image(self):
        """Even a tiny image should produce at least 1×1."""
        new_w, new_h = compute_new_dimensions(2, 2)
        assert new_w >= 1
        assert new_h >= 1

    def test_large_image(self):
        """Simulate a 200 MP image (16384 × 12288)."""
        new_w, new_h = compute_new_dimensions(16384, 12288)
        new_mp = megapixels(new_w, new_h)
        # Should be ~20 MP (±1 MP tolerance)
        assert 19.0 < new_mp < 21.0


class TestMegapixels:
    def test_12mp(self):
        assert abs(megapixels(4000, 3000) - 12.0) < 0.01

    def test_201mp(self):
        assert abs(megapixels(16384, 12288) - 201.3) < 0.1


# ── Integration test: full pipeline ─────────────────────────────────────────


class TestProcessDirectory:
    """Create synthetic images in a temp dir, process them, and validate."""

    @pytest.fixture()
    def dirs(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        log_dir = tmp_path / "logs"
        input_dir.mkdir()
        output_dir.mkdir()
        log_dir.mkdir()
        return input_dir, output_dir, log_dir

    def _make_image(self, path, width, height, fmt="JPEG"):
        """Create a solid-color image of the given size."""
        img = Image.new("RGB", (width, height), color=(100, 150, 200))
        img.save(str(path), format=fmt)

    def test_process_jpg(self, dirs):
        input_dir, output_dir, log_dir = dirs
        self._make_image(input_dir / "photo.jpg", 4000, 3000)

        logger = setup_logging(str(log_dir))
        count = process_directory(str(input_dir), str(output_dir), logger)

        assert count == 1
        out_path = output_dir / "photo.jpg"
        assert out_path.exists()

        with Image.open(str(out_path)) as out_img:
            w, h = out_img.size
            ratio = (w * h) / (4000 * 3000)
            assert abs(ratio - 0.1) < 0.01

    def test_process_png(self, dirs):
        input_dir, output_dir, log_dir = dirs
        self._make_image(input_dir / "diagram.png", 2000, 1500, fmt="PNG")

        logger = setup_logging(str(log_dir))
        count = process_directory(str(input_dir), str(output_dir), logger)

        assert count == 1
        assert (output_dir / "diagram.png").exists()

    def test_skips_unsupported(self, dirs):
        """Non-image files should be silently skipped."""
        input_dir, output_dir, log_dir = dirs
        (input_dir / "notes.txt").write_text("not an image")
        self._make_image(input_dir / "real.jpg", 800, 600)

        logger = setup_logging(str(log_dir))
        count = process_directory(str(input_dir), str(output_dir), logger)

        assert count == 1
        assert not (output_dir / "notes.txt").exists()

    def test_empty_directory(self, dirs):
        input_dir, output_dir, log_dir = dirs
        logger = setup_logging(str(log_dir))
        count = process_directory(str(input_dir), str(output_dir), logger)
        assert count == 0
