"""
Mock Digital Image Processing Engine - Simulate pothole analysis for demo.

Crops pothole ROI from image and simulates depth/volume calculations
with professional terminal output and visualization.
"""

import cv2
import random
import numpy as np
from pathlib import Path
from typing import Tuple, List, Union


def process_pothole_data(image_path: str, coords: Union[List[int], Tuple[int, int, int, int]], verbose: bool = True) -> dict:
    """
    Process pothole data and calculate simulated RPS.

    Loads image, crops ROI using bounding box, simulates volume calculation,
    determines repair priority score, and returns results.

    Args:
        image_path: Path to input image file.
        coords: Bounding box as [x1, y1, x2, y2].
        verbose: If True, print detailed output. If False, silent mode.

    Returns:
        Dictionary with analysis results (area_m2, depth_cm, volume_kg, rps_level, severity)

    Raises:
        FileNotFoundError: If image not found.
        ValueError: If coordinates invalid or image cannot be read.
        TypeError: If coords not list/tuple of integers.
    """
    # Validate inputs
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
    if verbose:
        print("[DIP ENGINE] Initializing Digital Image Processing pipeline...")
        print(f"[DIP ENGINE] Image shape: {w}×{h} pixels\n")

    # Validate bbox bounds
    x1_clipped: int = max(0, x1)
    y1_clipped: int = max(0, y1)
    x2_clipped: int = min(w, x2)
    y2_clipped: int = min(h, y2)

    if verbose:
        print(f"[DIP ENGINE] Processing ROI: [{x1_clipped}, {y1_clipped}, {x2_clipped}, {y2_clipped}]")

    # Crop ROI
    roi: cv2.Mat = image[y1_clipped:y2_clipped, x1_clipped:x2_clipped]

    if roi.size == 0:
        raise ValueError("ROI is empty after cropping")

    roi_height, roi_width = roi.shape[:2]
    if verbose:
        print(f"[DIP ENGINE] ROI extracted: {roi_width}×{roi_height} pixels\n")

    # Calculate actual area from ROI dimensions
    pixel_size_cm: float = 0.1
    roi_area_cm2: float = roi_width * roi_height * (pixel_size_cm ** 2)
    area_m2: float = roi_area_cm2 / 10000.0
    if verbose:
        print(f"[DIP ENGINE] Calculated area: {area_m2:.4f} m² ({roi_area_cm2:.1f} cm²)")

    # Calculate actual volume based on ROI size and simulated depth
    # Depth simulation: based on ROI pixel intensity variation
    gray_roi: np.ndarray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    intensity_variation: float = float(np.std(gray_roi))
    depth_cm: float = (intensity_variation / 255.0) * 5.0
    volume_cm3: float = roi_area_cm2 * depth_cm
    volume_kg: float = (volume_cm3 / 1000.0) * 2.4
    if verbose:
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

    if verbose:
        print("[DIP ENGINE] Repair Priority Score (RPS) Calculation:")
        print(f"  Volume: {volume_kg_int} kg")
        print(f"  Decision Logic: {'>' if volume_kg_int > 40 else '>=' if volume_kg_int >= 20 else '<'} threshold")
        print(f"  → RPS Level {rps_level} ({severity})")
        print("[DIP ENGINE] Handoff complete. Passing ROI to Phase 3 (Visualization)...\n")

    # Return results as dictionary
    return {
        "area_m2": area_m2,
        "depth_cm": depth_cm,
        "volume_kg": volume_kg,
        "rps_level": rps_level,
        "severity": severity
    }


if __name__ == "__main__":
    try:
        process_pothole_data("data/test_pothole_2.jpg", [100, 150, 400, 350])
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except ValueError as e:
        print(f"[ERROR] {e}")
    except TypeError as e:
        print(f"[ERROR] {e}")
