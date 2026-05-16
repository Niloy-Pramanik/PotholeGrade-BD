"""
Integration Testing Script - YOLOv8 Inference Demo.

Runs inference on test images and extracts bounding box coordinates for
downstream OpenCV DIP pipeline. Demonstrates complete detection workflow.
"""

import cv2
import sys
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import Optional, Union, List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def process_pothole_data(image_path: str, coords: Union[List[int], Tuple[int, int, int, int]]) -> None:
    """
    Process pothole data and calculate RPS.

    Loads image, crops ROI using bounding box, calculates volume,
    determines repair priority score, and displays results.

    Args:
        image_path: Path to input image file.
        coords: Bounding box as [x1, y1, x2, y2].
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
    image: cv2.Mat = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = image.shape[:2]
    print("[DIP ENGINE] Initializing Digital Image Processing pipeline...")
    print(f"[DIP ENGINE] Image shape: {w}×{h} pixels\n")

    # Validate bbox bounds
    x1_clipped: int = max(0, x1)
    y1_clipped: int = max(0, y1)
    x2_clipped: int = min(w, x2)
    y2_clipped: int = min(h, y2)

    print(f"[DIP ENGINE] Processing ROI: [{x1_clipped}, {y1_clipped}, {x2_clipped}, {y2_clipped}]")

    # Crop ROI
    roi: cv2.Mat = image[y1_clipped:y2_clipped, x1_clipped:x2_clipped]

    if roi.size == 0:
        raise ValueError("ROI is empty after cropping")

    roi_height, roi_width = roi.shape[:2]
    print(f"[DIP ENGINE] ROI extracted: {roi_width}×{roi_height} pixels\n")

    # Calculate actual area from ROI dimensions
    pixel_size_cm: float = 0.1
    roi_area_cm2: float = roi_width * roi_height * (pixel_size_cm ** 2)
    area_m2: float = roi_area_cm2 / 10000.0
    print(f"[DIP ENGINE] Calculated area: {area_m2:.4f} m² ({roi_area_cm2:.1f} cm²)")

    # Calculate actual volume based on ROI size and depth from pixel intensity
    gray_roi: np.ndarray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    intensity_variation: float = float(np.std(gray_roi))
    depth_cm: float = (intensity_variation / 255.0) * 5.0
    volume_cm3: float = roi_area_cm2 * depth_cm
    volume_kg: float = (volume_cm3 / 1000.0) * 2.4
    print(f"[DIP ENGINE] Calculated depth: {depth_cm:.2f} cm (from pixel intensity)")
    print(f"[DIP ENGINE] Calculated volume: {volume_kg:.2f} kg of asphalt\n")

    # Convert to int for RPS calculation
    volume_kg_int: int = int(volume_kg)

    # Calculate RPS
    if volume_kg_int > 40:
        rps_level: int = 3
        severity: str = "CRITICAL"
    elif volume_kg_int >= 20:
        rps_level = 2
        severity = "MODERATE"
    else:
        rps_level = 1
        severity = "LOW"

    print("[DIP ENGINE] Repair Priority Score (RPS) Calculation:")
    print(f"  Volume: {volume_kg_int} kg")
    print(f"  Decision Logic: {'>' if volume_kg_int > 40 else '>=' if volume_kg_int >= 20 else '<'} threshold")
    print(f"  → RPS Level {rps_level} ({severity})")
    print("[DIP ENGINE] Handoff complete.\n")


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
    print(f"[SYSTEM] Confidence threshold: 0.10\n")

    # Run inference
    results = model.predict(image_resized, conf=0.10, verbose=False)

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

            # Pass to DIP Engine for volume calculation and RPS
            coords_list: list = [x1, y1, x2, y2]
            try:
                print(f"[PIPELINE] Handoff to Phase 2 - DIP Engine...\n")
                process_pothole_data(image_path, coords_list)
            except Exception as e:
                print(f"[ERROR] DIP Engine failed: {e}\n")

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
        run_pipeline_demo("data/test_pothole_2.jpg")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except ValueError as e:
        print(f"[ERROR] {e}")
