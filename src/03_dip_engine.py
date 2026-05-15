"""
Digital Image Processing Engine - Calculate pothole depth and volume.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class PotholeMetrics:
    """Pothole measurement results."""
    depth_cm: float
    area_cm2: float
    volume_kg: float
    gradient_std: float


class PotholeDIPEngine:
    """
    OpenCV-based engine for pothole depth and volume estimation.
    
    Uses shadow gradient analysis to estimate depth without neural networks.
    """

    CALIBRATION_CONSTANT: float = 0.15
    PIXEL_TO_CM2: float = 0.5
    ASPHALT_DENSITY: float = 2.4

    def __init__(self) -> None:
        """Initialize DIP engine."""
        pass

    def calculate_metrics(
        self,
        image: np.ndarray,
        polygon_coords: np.ndarray
    ) -> PotholeMetrics:
        """
        Calculate pothole depth and volume.

        Args:
            image: RGB image array (H, W, 3)
            polygon_coords: Polygon vertices (N, 2)

        Returns:
            PotholeMetrics with depth, area, volume

        Raises:
            ValueError: If polygon < 3 points or image invalid
            TypeError: If inputs not numpy arrays
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected ndarray for image, got {type(image)}")
        if not isinstance(polygon_coords, np.ndarray):
            raise TypeError(f"Expected ndarray for polygon, got {type(polygon_coords)}")

        if len(polygon_coords) < 3:
            raise ValueError(f"Polygon needs >= 3 points, got {len(polygon_coords)}")

        if image.size == 0:
            raise ValueError("Image is empty")

        # Create mask for pothole region
        mask: np.ndarray = self._create_mask(image, polygon_coords)

        # Extract pothole region
        gray_image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pothole_region: np.ndarray = gray_image[mask > 0]

        if pothole_region.size == 0:
            raise ValueError("Pothole mask is empty")

        # Calculate gradient-based depth
        gradient_std: float = self._calculate_shadow_gradient(gray_image, mask)
        depth_cm: float = gradient_std * self.CALIBRATION_CONSTANT

        # Calculate area
        area_pixels: float = float(cv2.contourArea(polygon_coords.astype(np.float32)))
        area_cm2: float = area_pixels * self.PIXEL_TO_CM2

        # Calculate volume
        volume_cm3: float = area_cm2 * depth_cm
        volume_g: float = volume_cm3 * self.ASPHALT_DENSITY
        volume_kg: float = volume_g / 1000.0

        return PotholeMetrics(
            depth_cm=depth_cm,
            area_cm2=area_cm2,
            volume_kg=volume_kg,
            gradient_std=gradient_std
        )

    def _create_mask(self, image: np.ndarray, polygon_coords: np.ndarray) -> np.ndarray:
        """Create binary mask for pothole region."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        polygon_int = polygon_coords.astype(np.int32)
        cv2.fillPoly(mask, [polygon_int], 255)
        return mask

    def _calculate_shadow_gradient(
        self,
        gray_image: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """Calculate gradient magnitude std dev in masked region."""
        sobel_x: np.ndarray = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y: np.ndarray = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude: np.ndarray = np.sqrt(sobel_x**2 + sobel_y**2)
        masked_gradients: np.ndarray = gradient_magnitude[mask > 0]
        
        if masked_gradients.size == 0:
            return 0.0
        
        return float(np.std(masked_gradients))
