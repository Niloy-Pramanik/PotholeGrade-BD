# PotholeGrade-BD: Integrated Pipeline Documentation

## Overview

The complete pipeline integrates **YOLOv8 detection** with **DIP analysis** to provide end-to-end pothole assessment:

```
Image Input
    ↓
YOLOv8 Detection (640×640 model)
    ↓
For each detected pothole:
  - Extract bounding box [x1, y1, x2, y2]
  - Crop ROI (Region of Interest)
  - Calculate pothole area from dimensions
  - Estimate depth from pixel intensity variation
  - Compute volume and asphalt mass
  - Calculate Repair Priority Score (RPS)
    ↓
Output: RPS Level 1-3 for each pothole
```

## Pipeline Components

### Phase 1: YOLOv8 Detection
**File:** `src/05_inference_demo.py`

- Loads trained YOLOv8 Nano model (6.3 MB)
- Runs inference at 10% confidence threshold
- Detects all pothole regions in resized image (640×640)
- Extracts bounding box coordinates: `[x1, y1, x2, y2]`

**Example Output:**
```
[DETECTION #1] Pothole found at [37, 438, 116, 640] with 14.7% confidence
[DETECTION #2] Pothole found at [71, 365, 446, 460] with 11.7% confidence
[DETECTION #3] Pothole found at [74, 365, 563, 467] with 11.5% confidence
```

### Phase 2: DIP Engine (Digital Image Processing)
**File:** `src/05_inference_demo.py` (embedded) / `src/03_dip_engine.py` (standalone)

**Step 1: ROI Extraction**
```
Original image: 720×720 pixels
Bounding box: [37, 438, 116, 640]
Extracted ROI: 79×202 pixels
```

**Step 2: Area Calculation**
```
Pixel calibration: 0.1 cm per pixel
ROI dimensions: 79 × 202 pixels
Area = 79 × 202 × (0.1)² = 159.6 cm² = 0.0160 m²
```

**Step 3: Depth Estimation**
```
Convert ROI to grayscale
Calculate intensity std: σ = standard deviation of pixel values
Depth = (σ / 255.0) × 5.0 cm
Example: σ = 14.3 → depth = 0.28 cm
```

**Step 4: Volume Calculation**
```
Volume = Area × Depth × Asphalt Density
Volume = 159.6 cm² × 0.28 cm = 44.7 cm³
Mass = (44.7 / 1000) × 2.4 g/cm³ = 0.107 kg
```

**Step 5: RPS Determination**
```
if volume_kg > 40:
    RPS Level 3 (CRITICAL) - Immediate repair
elif volume_kg >= 20:
    RPS Level 2 (MODERATE) - Schedule repair
else:
    RPS Level 1 (LOW) - Monitor
```

## Running the Pipeline

### Option 1: Full Demo (with GUI)
```bash
./run_integrated_demo.sh
```
Shows:
- YOLOv8 detections with bounding boxes
- DIP analysis for each pothole
- Visual window displaying annotated image

### Option 2: Automated Test (no GUI)
```bash
python test_integrated_pipeline.py
```
Useful for:
- Headless servers
- CI/CD pipelines
- Batch processing
- No X11 display required

### Option 3: Manual Python Execution
```bash
.venv/bin/python src/05_inference_demo.py
```

## Test Results

**Image:** `data/test_pothole_2.jpg` (720×720 pixels)

| Detection | Confidence | Dimensions | Area | Depth | Volume | RPS |
|-----------|-----------|-----------|------|-------|--------|-----|
| #1 | 14.7% | 79×202 px | 0.0160 m² | 0.28 cm | 0.11 kg | Level 1 (LOW) |
| #2 | 11.7% | 375×95 px | 0.0356 m² | 0.25 cm | 0.21 kg | Level 1 (LOW) |
| #3 | 11.5% | 489×102 px | 0.0499 m² | 0.26 cm | 0.31 kg | Level 1 (LOW) |

**Summary:**
- ✅ 3/3 detections processed successfully
- ✅ All RPS calculations completed
- ✅ Pipeline integration validated

## Implementation Details

### Type Handling
- YOLOv8 outputs numpy int64 coordinates
- Automatically converted to Python int via: `[int(c) for c in coords]`
- Handles both int and np.integer types

### Coordinate System
```
Image (720×720)
┌─────────────────────────────┐
│                             │
│   [x1, y1, x2, y2]         │
│   ↓                        │
│   ┌──────────────┐         │
│   │ ROI (79×202) │         │
│   └──────────────┘         │
│                             │
└─────────────────────────────┘

x1=37, y1=438 (top-left)
x2=116, y2=640 (bottom-right)
width = x2 - x1 = 79 pixels
height = y2 - y1 = 202 pixels
```

### Calibration Constants
- **Pixel Size:** 0.1 cm/pixel (adjustable based on camera/resolution)
- **Max Depth:** 5.0 cm (maximum pothole depth to simulate)
- **Asphalt Density:** 2.4 g/cm³ (standard road asphalt)

## Customization

### Adjust Confidence Threshold
```python
# In 05_inference_demo.py, line ~155
results = model.predict(image_resized, conf=0.20, verbose=False)  # Change 0.10 to 0.20
```

### Modify Pixel Calibration
```python
# In 05_inference_demo.py or 03_dip_engine.py, line ~74
pixel_size_cm: float = 0.15  # Change from 0.1 to 0.15
```

### Change RPS Thresholds
```python
# In 05_inference_demo.py, lines ~92-99
if volume_kg_int > 50:  # Increased from 40
    rps_level = 3
elif volume_kg_int >= 25:  # Increased from 20
    rps_level = 2
else:
    rps_level = 1
```

## Future Enhancements

1. **Real Depth Sensing:** Replace intensity-based depth with LiDAR/stereo depth
2. **Multi-image Processing:** Batch process entire video sequences
3. **Database Integration:** Store RPS results in PostgreSQL/SQLite
4. **Web Dashboard:** Display pothole locations on interactive map
5. **Edge Deployment:** Deploy on Nvidia Jetson for real-time road analysis

## Files Summary

| File | Purpose | Status |
|------|---------|--------|
| `src/05_inference_demo.py` | Integrated pipeline (YOLO + DIP) | ✅ Ready |
| `src/03_dip_engine.py` | Standalone DIP engine | ✅ Ready |
| `test_integrated_pipeline.py` | Automated test without GUI | ✅ Ready |
| `run_integrated_demo.sh` | Simple execution script | ✅ Ready |
| `run_demo.sh` | Original YOLO demo script | ✅ Ready |
| `run_dip.sh` | Standalone DIP demo script | ✅ Ready |

---

**Last Updated:** May 16, 2026
**Status:** ✅ Integrated Pipeline Operational
