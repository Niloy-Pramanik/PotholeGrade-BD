# 🏗️ PotholeGrade-BD: Master Implementation Guide

**Project Objective:** To build a localized, smartphone-based computer vision system for Bangladesh roads. The system uses **Transfer Learning (YOLOv8-Seg)** to detect pothole boundaries (polygons), and then applies custom **Digital Image Processing (DIP)** to estimate physical depth from shadow gradients, calculate asphalt repair volume (kg), and assign a Repair Priority Score (RPS).

---

## 📂 Directory Structure

The project is organized as follows:

```text
PotholeGrade-BD/
│
├── data/                       # (Ignored by Git) Local dataset storage
│   ├── raw_videos/             # Smartphone mp4 dashcam footage
│   ├── images/                 # Extracted 1-fps .jpg frames
│   └── dataset_yolo/           # Exported Roboflow dataset (Train/Val/Test)
│
├── src/                        # Core Engineering Scripts
│   ├── __init__.py             # Package initialization
│   ├── 01_extract_frames.py    # Extracts 1-fps images from raw videos
│   ├── 02_train_yolo.py        # Fine-tunes YOLOv8-Seg (Transfer Learning)
│   ├── 03_dip_engine.py        # OpenCV math: Area, Shadow Depth, Volume
│   ├── 04_rps_logic.py         # Priority scoring algorithms
│   └── 05_inference_main.py    # The final pipeline combining YOLO + DIP
│
├── notebooks/                  # Jupyter Notebooks for testing & visualization
│   ├── depth_math_test.ipynb   # Visualizing Sobel filters for shadow gradients
│   └── dataset_eda.ipynb       # Explaining data diversity to faculty
│
├── runs/                       # (Ignored by Git) YOLO training logs & weights
│   └── segment/train/weights/best.pt 
│
├── .gitignore                  # Prevents uploading heavy files to GitHub
├── requirements.txt            # Python dependencies
└── README.md                   # This document
```

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Installed packages:**
- `opencv-python==4.8.0.76` - Image processing
- `numpy==1.26.0` - Numerical computing
- `ultralytics==8.1.0` - YOLOv8 models
- `matplotlib==3.8.0` - Visualization
- `Pillow==10.0.0` - Image I/O
- `tqdm==4.66.1` - Progress bars

---

## 📋 Phase-by-Phase Implementation

### Phase 1: Data Collection & Frame Extraction

**Goal:** Capture diverse Dhaka road conditions and extract frames for annotation.

```bash
# 1. Add your smartphone video files to data/raw_videos/
cd /Users/niloypramanik/PotholeGrade-BD
mkdir -p data/raw_videos

# 2. Copy your dashcam videos (*.mp4) to data/raw_videos/

# 3. Extract frames at 1 fps
python src/01_extract_frames.py
```

**What happens:**
- Reads all `.mp4` files from `data/raw_videos/`
- Extracts exactly 1 frame per second
- Saves JPEG images to `data/images/` with progress bar
- Example: A 10-minute video (600 seconds) → 600 frames

---

### Phase 2: Annotation & Dataset Preparation

**Goal:** Trace pothole polygons and prepare data for YOLO training.

1. **Go to [Roboflow.com](https://roboflow.com)** and create a free account.
2. **Create a new project** with type "Instance Segmentation"
3. **Upload your extracted frames** from `data/images/`
4. **Annotate potholes** using the Polygon tool:
   - Trace the jagged edges carefully
   - Use classes: `Dry_Pothole`, `Wet_Pothole`
   - Aim for 300-500 annotated images
5. **Apply Data Augmentation** in Roboflow:
   - Random brightness/contrast
   - Slight rotation (±15°)
   - Blur augmentation
   - This expands your 300 images → 1,500+ training images
6. **Export in YOLOv8 PyTorch format**
7. **Download and extract to** `data/dataset_yolo/`

Your `data/dataset_yolo/` should contain:
```
dataset_yolo/
├── data.yaml           # Dataset metadata
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

### Phase 3: Transfer Learning with YOLOv8

**Goal:** Fine-tune the pre-trained model on your pothole dataset.

```bash
# Train the model (50 epochs on a GPU takes ~30 minutes)
python src/02_train_yolo.py
```

**What happens:**
- Loads the pre-trained `yolov8n-seg.pt` (nano model, 3.2 MB)
- Trains on your custom dataset for 50 epochs
- Saves the best weights to `runs/segment/train/weights/best.pt`
- Generates training logs and metrics

**Expected output:**
- Training completed in 30-60 minutes (GPU) or 2-3 hours (CPU)
- `best.pt` file (~10-15 MB)

---

### Phase 4: DIP Engine – Shadow-Based Depth Estimation

**Goal:** Calculate pothole depth and volume using pure OpenCV.

The `PotholeDIPEngine` class in `src/03_dip_engine.py` implements:

1. **Virtual Masking:** Isolate pothole pixels using polygon mask
2. **Shadow Gradient Analysis:**
   - Convert frame to grayscale
   - Apply Sobel filters (X and Y directions)
   - Compute gradient magnitude: $G = \sqrt{Sobel_x^2 + Sobel_y^2}$
3. **Depth Estimation:**
   - Calculate standard deviation of gradients
   - $Depth_{cm} = StdDev \times 0.15$ (calibration constant)
4. **Volume Calculation:**
   - 2D pixel area: $Area_{pixels} = cv2.contourArea(polygon)$
   - Convert to cm²: $Area_{cm^2} = Area_{pixels} \times 0.5$
   - Volume: $Volume_{kg} = Area_{cm^2} \times Depth_{cm} \times 2.4 \text{ (asphalt density)}$

**Return values:**
```python
PotholeMetrics(
    depth_cm=8.5,           # Estimated depth in cm
    area_cm2=245.0,         # 2D pothole area
    volume_kg=5.0,          # Asphalt needed for repair
    gradient_std=56.7       # Raw shadow gradient metric
)
```

---

### Phase 5: Repair Priority Score (RPS) Logic

**Goal:** Automate maintenance urgency classification.

The `calculate_rps()` function assigns scores (1-5):

| RPS | Class | Volume | Description |
|-----|-------|--------|-------------|
| **5** | Any | Any | **Wet Pothole** – Hidden depth danger |
| **4** | Dry | > 20 kg | Major structural damage |
| **3** | Dry | 5–20 kg | Standard repair needed |
| **1** | Dry | < 5 kg | Monitor / routine maintenance |

**Example:**
```python
from src.rps_logic import calculate_rps

rps = calculate_rps(volume_kg=12.5, pothole_class="Dry_Pothole")
# Returns: 3 (Standard repair)

rps = calculate_rps(volume_kg=2.0, pothole_class="Wet_Pothole")
# Returns: 5 (Critical – hidden depth)
```

---

### Phase 6: Full Pipeline Inference

**Goal:** Combine YOLO + DIP + RPS for real-time pothole detection.

```bash
# Copy a test video to data/test_video.mp4
cp /path/to/your/test_video.mp4 data/test_video.mp4

# Run inference
python src/05_inference_main.py
```

**What happens:**
1. Loads trained YOLO model
2. For each frame:
   - YOLO detects pothole polygons
   - DIP Engine calculates volume
   - RPS Logic scores urgency
   - Draws annotations:
     - **Green polygon**: RPS 1-2 (low priority)
     - **Orange polygon**: RPS 3 (medium priority)
     - **Red polygon**: RPS 4-5 (high priority)
   - Text overlay: `Dry_Pothole | Vol: 12.5 kg | RPS: 3`
3. Saves annotated video to `output_annotated.mp4`

---

## 🔧 Advanced Usage

### Adjust Calibration Constants

Edit `src/03_dip_engine.py`:

```python
class PotholeDIPEngine:
    CALIBRATION_CONSTANT: float = 0.15    # Adjust for your camera
    PIXEL_TO_CM2: float = 0.5              # Depends on camera focal length
    ASPHALT_DENSITY: float = 2.4           # kg/cm³ (standard for Bangladesh)
```

### Run Only Frame Extraction

```bash
python src/01_extract_frames.py
# Output: 1-fps JPEG images in data/images/
```

### Run Only Training

```bash
python src/02_train_yolo.py
# Output: Trained model in runs/segment/train/weights/best.pt
```

---

## 📊 Engineering Pipeline Summary

```
Raw Video (30fps)
    ↓
[01_extract_frames.py]
    ↓
JPEG Frames (1fps)
    ↓
[Roboflow Annotation]
    ↓
Labeled Dataset
    ↓
[02_train_yolo.py]
    ↓
YOLOv8 Model (best.pt)
    ↓
Test Video
    ↓
[05_inference_main.py]
    ├─ YOLO: Detect polygon
    ├─ DIP Engine: Calculate depth & volume
    ├─ RPS Logic: Assign urgency score
    └─ Visualization: Draw annotations
    ↓
Output Video + Metrics
```

---

## ⚠️ Important Notes

### Critical First Commits

Before running training, ensure `.gitignore` is set up:

```bash
git add .gitignore
git commit -m "Add gitignore to prevent uploading heavy files"
```

This prevents:
- `data/` → Raw videos & images (~500 MB+)
- `runs/` → Training checkpoints (~1 GB+)
- `__pycache__/` → Python bytecode

### GPU Acceleration

For faster training, install CUDA-enabled PyTorch:

```bash
# For NVIDIA GPUs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then reinstall ultralytics
pip install ultralytics --upgrade
```

Training speed: GPU (30 min) vs CPU (2+ hours)

---

## 🎓 Faculty Presentation Structure

Use these files for your academic presentation:

1. **`notebooks/dataset_eda.ipynb`** – Data diversity visualization
2. **`notebooks/depth_math_test.ipynb`** – Sobel filter walkthrough
3. **Output video** – Real-time inference results
4. **Metrics summary** – RPS distribution, accuracy stats

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `No .mp4 files found` | Ensure videos are in `data/raw_videos/` |
| `data.yaml not found` | Export dataset from Roboflow to `data/dataset_yolo/` |
| `best.pt not found` | Train model first with `02_train_yolo.py` |
| `CUDA out of memory` | Reduce batch size in `02_train_yolo.py` (e.g., 8 instead of 16) |
| No detections in inference | Check confidence threshold in `05_inference_main.py` (try 0.3 instead of 0.5) |

---

## 📞 Support & References

- **YOLOv8 Docs:** https://docs.ultralytics.com/
- **OpenCV Docs:** https://docs.opencv.org/
- **Roboflow Guide:** https://roboflow.com/course

---

**Version:** 1.0.0  
**Last Updated:** May 2026  
**Status:** Production Ready ✅
