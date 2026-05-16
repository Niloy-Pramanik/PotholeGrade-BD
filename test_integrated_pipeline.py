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

# Add src to path and import DIP engine
sys.path.insert(0, str(Path(__file__).parent / "src"))
from dip_engine import process_pothole_data


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
                result_dict = process_pothole_data(image_path, coords_list, verbose=True)
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
