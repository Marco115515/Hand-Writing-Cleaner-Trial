# Handwriting Remover – Printed vs Handwritten Text Separation

U-Net based model that detects and removes **handwritten content** from scanned/mixed documents while attempting to preserve printed text.

The model takes a grayscale image (512×512 patches) and outputs a binary mask where handwritten regions are marked (1 = handwriting, 0 = background/printed).

## Quick Results Preview

### Example 1 – IAM-style handwriting (seen during training)

**Before**  
![Image 1 Before](https://github.com/Marco115515/Hand-Writing-Cleaner-Trial/blob/main/sample_1.png)

**After** (handwriting mostly removed)  
![Image 1 After](https://github.com/Marco115515/Hand-Writing-Cleaner-Trial/blob/main/sample_1_cleaned.png)

### Example 2 – Unseen handwriting style

**Before**  
![Image 2 Before](https://github.com/Marco115515/Hand-Writing-Cleaner-Trial/blob/main/sample_2.jpg)

**After** (poor generalization – many misses / false positives)  
![Image 2 Before](https://github.com/Marco115515/Hand-Writing-Cleaner-Trial/blob/main/sample_2_cleaned.jpg)

> The model performs reasonably on synthetic IAM-like data but struggles significantly with real-world handwriting never seen during training.

## Features

- U-Net architecture for pixel-level handwriting segmentation
- Grayscale 512×512 input → binary mask output
- **Sliding window** inference for images larger than 512×512
- Pure inference script (no training code included in this repo yet)

## Requirements

```txt
numpy==2.1.1          # (you wrote 2.4.3 – probably typo; 2.1.x is common with TF 2.19+)
opencv-python==4.10.0.84     # or 4.13.0.90 / 4.13.0.92
opencv-contrib-python==4.10.0.84
tensorflow==2.17.0    # 2.19 / 2.21 might be very new / unstable in 2025–2026
```
## How to Run (Inference)
1. Download the [model](https://drive.google.com/file/d/1oEMoZySilJ6JUUh4XPICuFiuIJ5i886y/view?usp=sharing), put it into the same directory as application.py
1. Place your input image anywhere (supports jpg, png, tiff, etc.)
2. Edit application.py and update the paths:

```python
# application.py

INPUT_PATH  = "path/to/your/scan_or_document.jpg"   # ← change this
OUTPUT_PATH = "results/cleaned_version.png"         # ← change this or leave as-is
```
3. Run the script:

```bash
python application.py
```

What happens:

- Image is loaded and converted to grayscale
- Processed in overlapping 512×512 windows
- Handwriting mask is predicted
- Cleaned image is generated (simple removal/inpainting logic)
- Result saved to OUTPUT_PATH

## Planned Improvements

- Collect real annotated mixed documents (printed + handwritten + masks)
→ Most promising way to close the domain gap (but expensive in labeling time)
Create more realistic synthetic data
- Overlay handwriting on real printed pages/forms/books
Add blur, rotation, noise, shadows, compression artifacts
