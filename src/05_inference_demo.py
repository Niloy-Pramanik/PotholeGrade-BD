"""
Integration Testing Script - YOLOv8 Inference Demo.

Runs inference on test images and extracts bounding box coordinates for
downstream OpenCV DIP pipeline. Demonstrates complete detection workflow.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import Optional


def run_pipeline_demo(image_path: str) -> None:
    """
    Run inference pipeline demo on a test image.

    Loads the trained YOLOv8 model, runs inference at 20% confidence,
    extracts bounding box coordinates, and displays annotated results.

    Args:
        image_path: Path to input image file (JPEG/PNG).

    Raises:
        FileNotFoundError: If image or model weights not found.
        ValueError: If image cannot be read.

    Returns:
        None
    """
    # Validate inputs
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model_path: str = "runs/detect/runs/detect/rdd_poc_model-4/weights/best.pt"
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    # Load model
    print("[SYSTEM] Loading custom YOLOv8 model...")
    model: YOLO = YOLO(model_path)
    print(f"[SYSTEM] Model loaded from: {model_path}\n")

    # Read image
    image: Optional[np.ndarray] = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    original_shape: tuple = image.shape
    print(f"[SYSTEM] Processing image: {image_path}")
    print(f"[SYSTEM] Original image shape: {image.shape}")

    # Resize to 640x640 to match training resolution
    image_resized: np.ndarray = cv2.resize(image, (640, 640))
    print(f"[SYSTEM] Resized to: {image_resized.shape}")
    print(f"[SYSTEM] Confidence threshold: 0.20\n")

    # Run inference
    results = model.predict(image_resized, conf=0.20, verbose=False)

    detection_count: int = 0

    # Extract and process detections
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            print("[SYSTEM] No potholes detected.")
            break

        boxes = result.boxes
        confidences: np.ndarray = boxes.conf.cpu().numpy()
        coords: np.ndarray = boxes.xyxy.cpu().numpy()

        print(f"[SYSTEM] Detections: {len(boxes)}\n")

        for idx, (x1_y1_x2_y2, conf) in enumerate(zip(coords, confidences)):
            detection_count += 1
            x1, y1, x2, y2 = x1_y1_x2_y2.astype(int)
            confidence_pct: float = float(conf) * 100

            print(
                f"[DETECTION #{detection_count}] "
                f"Pothole found at [{x1}, {y1}, {x2}, {y2}] "
                f"with {confidence_pct:.1f}% confidence. "
                f"Passing array to Phase 2 (Shadow Depth Math)..."
            )

            # Additional metadata for pipeline handoff
            width: int = x2 - x1
            height: int = y2 - y1
            area: int = width * height
            print(f"[METADATA] Bbox dims: {width}x{height}px, Area: {area}px²\n")

    # Visualize results
    if detection_count > 0:
        print(f"[SYSTEM] Annotating image with {detection_count} detection(s)...")
        annotated_image: np.ndarray = results[0].plot()

        # Display
        window_name: str = "PotholeGrade-BD: YOLOv8 Inference Demo"
        cv2.imshow(window_name, annotated_image)
        print(f"[SYSTEM] Displaying: {window_name}")
        print("[SYSTEM] Press any key to close window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("[SYSTEM] Window closed.\n")
    else:
        print("[SYSTEM] No detections to visualize.\n")

    print("[SYSTEM] Pipeline demo complete.")


if __name__ == "__main__":
    try:
        run_pipeline_demo("data/test_pothole.jpg")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except ValueError as e:
        print(f"[ERROR] {e}")
