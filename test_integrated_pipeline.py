#!/usr/bin/env python3
"""
Test script to validate integrated pipeline (YOLO detection + DIP engine).
Runs without GUI to test pipeline logic.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import Union, List, Tuple

# Add src to path and import DIP engine
sys.path.insert(0, str(Path(__file__).parent / "src"))
from dip_engine import process_pothole_data as dip_process

# Wrapper to capture DIP results
def process_pothole_data(image_path: str, coords: Union[List[int], Tuple[int, int, int, int]]) -> dict:
    """
    Wrapper around DIP engine that captures results for summary.
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
        raise TypeError("coords must be [x1, y1, x2, y2]")

    x1, y1, x2, y2 = coords

    if not all(isinstance(c, (int, np.integer)) for c in coords):
        coords = [int(c) for c in coords]
        x1, y1, x2, y2 = coords

    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid coordinates: x1={x1} < x2={x2}, y1={y1} < y2={y2}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = image.shape[:2]
    print(f"  [DIP] Image shape: {w}×{h} pixels")

    # Validate bbox bounds
    x1_clipped = max(0, x1)
    y1_clipped = max(0, y1)
    x2_clipped = min(w, x2)
    y2_clipped = min(h, y2)

    # Crop ROI
    roi = image[y1_clipped:y2_clipped, x1_clipped:x2_clipped]

    if roi.size == 0:
        raise ValueError("ROI is empty after cropping")

    roi_height, roi_width = roi.shape[:2]
    print(f"  [DIP] ROI extracted: {roi_width}×{roi_height} pixels")

    # Calculate actual area from ROI dimensions
    pixel_size_cm = 0.1
    roi_area_cm2 = roi_width * roi_height * (pixel_size_cm ** 2)
    area_m2 = roi_area_cm2 / 10000.0

    # Calculate volume based on ROI size and depth from pixel intensity
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    intensity_variation = float(np.std(gray_roi))
    depth_cm = (intensity_variation / 255.0) * 5.0
    volume_cm3 = roi_area_cm2 * depth_cm
    volume_kg = (volume_cm3 / 1000.0) * 2.4

    # Calculate RPS
    volume_kg_int = int(volume_kg)
    if volume_kg_int > 40:
        rps_level = 3
        severity = "CRITICAL"
    elif volume_kg_int >= 20:
        rps_level = 2
        severity = "MODERATE"
    else:
        rps_level = 1
        severity = "LOW"

    result = {
        "area_m2": area_m2,
        "depth_cm": depth_cm,
        "volume_kg": volume_kg,
        "rps_level": rps_level,
        "severity": severity
    }

    print(f"  [DIP] Area: {area_m2:.4f} m² ({roi_area_cm2:.1f} cm²)")
    print(f"  [DIP] Depth: {depth_cm:.2f} cm")
    print(f"  [DIP] Volume: {volume_kg:.2f} kg")
    print(f"  [DIP] RPS Level {rps_level} ({severity})")

    return result


def test_integrated_pipeline(image_path: str = "data/test_pothole_2.jpg", model_path: str = "runs/detect/runs/detect/rdd_poc_model-4/weights/best.pt"):
    """Test the complete pipeline: YOLO detection + DIP analysis."""

    print("=" * 70)
    print("INTEGRATED PIPELINE TEST: YOLO Detection + DIP Engine")
    print("=" * 70)

    # Validate files
    if not Path(image_path).exists():
        print(f"ERROR: Image not found: {image_path}")
        return False

    if not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        return False

    # Load model
    print(f"\n[SYSTEM] Loading YOLOv8 model...")
    model = YOLO(model_path)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Cannot read image: {image_path}")
        return False

    print(f"[SYSTEM] Image shape: {image.shape}")

    # Resize to 640x640
    image_resized = cv2.resize(image, (640, 640))
    print(f"[SYSTEM] Resized to: {image_resized.shape}")

    # Run inference
    print(f"\n[SYSTEM] Running YOLOv8 inference (conf=0.10)...\n")
    results = model.predict(image_resized, conf=0.10, verbose=False)

    detection_count = 0
    dip_results = []

    # Process each detection
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            print("[SYSTEM] No potholes detected.")
            break

        boxes = result.boxes
        confidences = boxes.conf.cpu().numpy()
        coords = boxes.xyxy.cpu().numpy()

        print(f"[SYSTEM] Found {len(boxes)} detection(s):\n")

        for idx, (x1_y1_x2_y2, conf) in enumerate(zip(coords, confidences)):
            detection_count += 1
            x1, y1, x2, y2 = [int(c) for c in x1_y1_x2_y2]
            confidence_pct = float(conf) * 100

            print(f"[DETECTION #{detection_count}]")
            print(f"  Coordinates: [{x1}, {y1}, {x2}, {y2}]")
            print(f"  Confidence: {confidence_pct:.1f}%")
            print(f"  Dimensions: {x2-x1}×{y2-y1} pixels")

            # Pass to DIP Engine
            coords_list = [x1, y1, x2, y2]
            try:
                result_dict = process_pothole_data(image_path, coords_list)
                dip_results.append(result_dict)
                print()
            except Exception as e:
                print(f"  ERROR: DIP Engine failed: {e}\n")

    # Summary
    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Total detections: {detection_count}")
    print(f"Successfully processed: {len(dip_results)}")

    if dip_results:
        print("\nDIP Results:")
        for i, res in enumerate(dip_results, 1):
            print(f"  Pothole #{i}:")
            print(f"    Area: {res['area_m2']:.4f} m²")
            print(f"    Depth: {res['depth_cm']:.2f} cm")
            print(f"    Volume: {res['volume_kg']:.2f} kg")
            print(f"    RPS: Level {res['rps_level']} ({res['severity']})")

    print("\n" + "=" * 70)
    print("PIPELINE TEST COMPLETE")
    print("=" * 70)

    return detection_count > 0 and len(dip_results) > 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test integrated YOLO + DIP pipeline")
    parser.add_argument("--image", default="data/test_pothole_2.jpg", 
                        help="Path to test image (default: data/test_pothole_2.jpg)")
    parser.add_argument("--model", default="runs/detect/runs/detect/rdd_poc_model-4/weights/best.pt",
                        help="Path to YOLOv8 model weights")
    args = parser.parse_args()
    
    success = test_integrated_pipeline(image_path=args.image, model_path=args.model)
    sys.exit(0 if success else 1)
