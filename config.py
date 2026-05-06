"""
Configuration Module for PotholeGrade-BD.

This module centralizes all calibration constants and configuration parameters
used throughout the pipeline. Modify these values based on your camera setup
and specific Bangladesh road conditions.
"""

# ============================================================================
# DIP ENGINE CALIBRATION CONSTANTS
# ============================================================================

# Gradient Standard Deviation to Physical Depth Conversion
# Default: 1 unit of gradient std ≈ 0.15 cm depth
# Adjust based on:
# - Camera focal length
# - Distance from road surface
# - Lighting conditions
DIP_CALIBRATION_CONSTANT = 0.15

# Pixel Area to Square Centimeter Conversion
# Default: 1 pixel² ≈ 0.5 cm²
# Adjust based on:
# - Camera resolution
# - Height of camera above road
# - Smartphone camera specs
DIP_PIXEL_TO_CM2 = 0.5

# Asphalt Material Density
# Standard value: 2.4 g/cm³
# This is consistent for most hot-mix asphalt used in Bangladesh
DIP_ASPHALT_DENSITY = 2.4

# ============================================================================
# YOLO TRAINING PARAMETERS
# ============================================================================

YOLO_MODEL_NAME = "yolov8n-seg.pt"  # nano model (fastest)
YOLO_EPOCHS = 50                    # Training epochs
YOLO_IMGSZ = 640                    # Input image size
YOLO_BATCH_SIZE = 16                # Batch size per iteration
YOLO_DEVICE = 0                     # GPU device ID (0 for first GPU, -1 for CPU)
YOLO_PATIENCE = 10                  # Early stopping patience (epochs)

# ============================================================================
# INFERENCE SETTINGS
# ============================================================================

# Minimum confidence threshold for YOLO detections
# Higher = fewer false positives, but might miss small potholes
# Lower = more detections, but more false positives
INFERENCE_CONFIDENCE_THRESHOLD = 0.5

# ============================================================================
# RPS (REPAIR PRIORITY SCORE) THRESHOLDS
# ============================================================================

# Volume thresholds for priority scoring (in kilograms of asphalt)
RPS_WET_POTHOLE_SCORE = 5              # Any wet pothole = RPS 5 (critical)
RPS_MAJOR_DAMAGE_THRESHOLD = 20.0      # volume_kg > 20 → RPS 4
RPS_STANDARD_REPAIR_THRESHOLD = 5.0    # volume_kg > 5 → RPS 3
# volume_kg <= 5 → RPS 1 (monitor)

# ============================================================================
# POTHOLE CLASS NAMES
# ============================================================================

POTHOLE_CLASSES = {
    "Dry_Pothole": 0,
    "Wet_Pothole": 1,
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Color codes for annotations (BGR format for OpenCV)
VISUALIZATION_COLORS = {
    1: (0, 255, 0),      # Green - RPS 1-2 (low priority)
    3: (0, 165, 255),    # Orange - RPS 3 (medium priority)
    4: (0, 0, 255),      # Red - RPS 4-5 (high priority)
}

# Font settings for text overlay
VISUALIZATION_FONT = "FONT_HERSHEY_SIMPLEX"
VISUALIZATION_FONT_SCALE = 0.6
VISUALIZATION_FONT_THICKNESS = 2

# ============================================================================
# FILE PATHS
# ============================================================================

DATA_RAW_VIDEOS_DIR = "data/raw_videos"
DATA_IMAGES_DIR = "data/images"
DATA_DATASET_YOLO_DIR = "data/dataset_yolo"
YOLO_WEIGHTS_PATH = "runs/segment/train/weights/best.pt"
INFERENCE_OUTPUT_VIDEO = "output_annotated.mp4"

# ============================================================================
# FRAME EXTRACTION SETTINGS
# ============================================================================

# Target frames per second for extraction
EXTRACT_FPS_TARGET = 1

# ============================================================================
# LOGGING & DEBUGGING
# ============================================================================

# Enable verbose output for debugging
VERBOSE = True

# Progress bar settings
TQDM_LEAVE = True  # Keep progress bars after completion
TQDM_UNIT = "frame"

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

"""
from config import *

# In your DIP engine:
def calculate_depth(gradient_std):
    return gradient_std * DIP_CALIBRATION_CONSTANT

# In your RPS logic:
def calculate_rps(volume_kg, pothole_class):
    if pothole_class.upper() == "WET_POTHOLE":
        return RPS_WET_POTHOLE_SCORE
    elif volume_kg > RPS_MAJOR_DAMAGE_THRESHOLD:
        return 4
    elif volume_kg > RPS_STANDARD_REPAIR_THRESHOLD:
        return 3
    else:
        return 1

# In your inference:
pipeline = PotholeInferencePipeline(
    model_path=YOLO_WEIGHTS_PATH,
    confidence_threshold=INFERENCE_CONFIDENCE_THRESHOLD
)
"""

# ============================================================================
# TUNING GUIDE FOR BANGLADESH CONDITIONS
# ============================================================================

"""
SCENARIO 1: High Sunlight / Strong Shadows
- Increase: DIP_CALIBRATION_CONSTANT (to 0.18-0.20)
- Reason: Strong shadows increase gradient magnitude

SCENARIO 2: Monsoon / Wet Conditions
- Increase: DIP_CALIBRATION_CONSTANT (to 0.15-0.17)
- Reason: Water reflections reduce visible shadow gradients

SCENARIO 3: Different Camera Height
- If lowering camera (closer to road):
  - Decrease: DIP_PIXEL_TO_CM2 (more pixels per cm²)
- If raising camera (further from road):
  - Increase: DIP_PIXEL_TO_CM2 (fewer pixels per cm²)

SCENARIO 4: Better Accuracy Needed
- Increase: YOLO_EPOCHS (50 → 75)
- Increase: YOLO_BATCH_SIZE (16 → 32, if GPU memory allows)
- Decrease: INFERENCE_CONFIDENCE_THRESHOLD (0.5 → 0.3)

SCENARIO 5: Faster Inference (real-time critical)
- Decrease: YOLO_IMGSZ (640 → 512)
- Decrease: YOLO_EPOCHS (50 → 30)
- Increase: INFERENCE_CONFIDENCE_THRESHOLD (0.5 → 0.7)
"""
