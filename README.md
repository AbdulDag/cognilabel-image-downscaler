# Cognilabel Image Resolution Reducer

A Python CLI tool that **batch-reduces image resolution by ~10×** (in megapixels) while preserving aspect ratio and visual quality. Designed for machine learning and labeling workflows where full-resolution images are unnecessarily large.

## Algorithm

To shrink total megapixels by a factor of 10 while keeping the aspect ratio:

```
scale = 1 / √10 ≈ 0.3162

new_width  = round(original_width  × scale)
new_height = round(original_height × scale)
```

Because both dimensions are multiplied by the same factor, the aspect ratio is preserved exactly. The resulting pixel area is `width × height × (1/10)` — a 10× reduction.

| Original | New (≈) |
|----------|---------|
| 150 MP   | 15 MP   |
| 200 MP   | 20 MP   |
| 80 MP    | 8 MP    |
| 12 MP    | 1.2 MP  |

Downscaling uses **LANCZOS** resampling for high visual quality.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place images in the input folder
#    Supported formats: JPG, JPEG, PNG, TIFF
cp your_images/* data/input_images/

# 3. Run the reducer
python src/reduce_resolution.py
```

## CLI Options

```
python src/reduce_resolution.py [OPTIONS]

Options:
  --input   PATH   Input image directory  (default: data/input_images)
  --output  PATH   Output image directory (default: data/output_images)
  --log-dir PATH   Log file directory     (default: logs)
```

### Example

```bash
python src/reduce_resolution.py --input my_photos --output my_photos_reduced
```

### Example Log Output

```
filename: image1.jpg
  original_resolution: 16384 x 12288 (201 MP)
  new_resolution:      5182 x 3886 (20.1 MP)
```

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Project Structure

```
cogni-label_img_comp/
├── data/
│   ├── input_images/     ← place source images here
│   └── output_images/    ← reduced images saved here
├── logs/                 ← timestamped log files
├── src/
│   └── reduce_resolution.py
├── tests/
│   └── test_reduce.py
├── requirements.txt
└── README.md
```
