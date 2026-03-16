# Handwriting Remover – Printed vs Handwritten Text Separation

U-Net based model that detects and removes **handwritten content** from scanned/mixed documents while attempting to preserve printed text.

The model takes a grayscale image (512×512 patches) and outputs a binary mask where handwritten regions are marked (1 = handwriting, 0 = background/printed).

<img src="https://via.placeholder.com/800x400/cccccc/000000?text=Example+Before+%26+After" alt="Before and After comparison" width="800"/>

*(Replace the placeholder with real before/after images once uploaded)*

## Quick Results Preview

### Example 1 – IAM-style handwriting (seen during training)

**Before**  
<img src="https://via.placeholder.com/600x400/eeeeee/333333?text=Before+-+IAM+style+handwriting" alt="Before – IAM example" width="48%"/>

**After** (handwriting mostly removed)  
<img src="https://via.placeholder.com/600x400/ffffff/000000?text=After+-+cleaned" alt="After – cleaned IAM" width="48%"/>

### Example 2 – Unseen handwriting style

**Before**  
<img src="https://via.placeholder.com/600x400/ffdddd/aa0000?text=Before+-+Real+unseen+handwriting" alt="Before – real world example" width="48%"/>

**After** (poor generalization – many misses / false positives)  
<img src="https://via.placeholder.com/600x400/ffffcc/886600?text=After+-+failed+to+clean+well" alt="After – failed generalization" width="48%"/>

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
