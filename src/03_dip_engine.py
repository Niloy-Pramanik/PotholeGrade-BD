"""
Digital Image Processing Engine for PotholeGrade-BD.

Core module for calculating pothole physical metrics (depth and volume)
from image analysis using shadow gradients, without any neural networks.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PotholeMetrics:
    """
    Data class to hold pothole metric results.

    Attributes:
        depth_cm (float): Estimated depth of the pothole in centimeters.
        area_cm2 (float): 2D area of the pothole in square centimeters.
        volume_kg (float): Estimated volume of asphalt needed for repair in kilograms.
        gradient_std (float): Standard deviation of gradient magnitude (raw metric).
    """
    depth_cm: float
    area_cm2: float
    volume_kg: float
    gradient_std: float


class PotholeDIPEngine:
    """
    Digital Image Processing Engine for pothole depth and volume estimation.

    This class implements a pure OpenCV-based approach to estimate pothole
    physical dimensions from smartphone camera images using shadow gradient
    analysis. No neural networks are used for depth calculation.

    Attributes:
        CALIBRATION_CONSTANT (float): Multiplier to convert gradient std to depth (cm).
            Empirically determined: 1 unit of gradient std ≈ 0.15 cm depth.
        PIXEL_TO_CM2 (float): Conversion factor from pixel area to cm².
            Assumes 1 pixel ≈ 0.5 cm² (adjustable based on camera calibration).
        ASPHALT_DENSITY (float): Density of asphalt in g/cm³. Standard value is 2.4.

    Example:
        >>> engine = PotholeDIPEngine()
        >>> image = cv2.imread("frame.jpg")
        >>> polygon = np.array([[100, 100], [150, 100], [150, 150], [100, 150]])
        >>> metrics = engine.calculate_metrics(image, polygon)
        >>> print(f"Depth: {metrics.depth_cm:.2f} cm, Volume: {metrics.volume_kg:.2f} kg")
    """

    CALIBRATION_CONSTANT: float = 0.15
    PIXEL_TO_CM2: float = 0.5
    ASPHALT_DENSITY: float = 2.4

    def __init__(self) -> None:
        """Initialize the Pothole DIP Engine with default calibration constants."""
        pass

    def calculate_metrics(
        self,
        image: np.ndarray,
        polygon_coords: np.ndarray
    ) -> PotholeMetrics:
        """
        Calculate pothole depth and volume from image and polygon boundaries.

        This method:
        1. Isolates the pothole region using the polygon mask.
        2. Analyzes shadow gradients inside the pothole using Sobel operators.
        3. Estimates depth from gradient standard deviation.
        4. Calculates 2D pixel area and converts to cm².
        5. Computes required asphalt volume for repair.

        Args:
            image (np.ndarray): RGB image array of shape (H, W, 3), dtype uint8.
            polygon_coords (np.ndarray): Polygon coordinates of shape (N, 2) where
                each row is [x, y] pixel coordinate. Must have at least 3 points.

        Returns:
            PotholeMetrics: Data class containing depth_cm, area_cm2, volume_kg,
                and gradient_std.

        Raises:
            ValueError: If polygon has fewer than 3 points or if image is invalid.
            TypeError: If inputs are not numpy arrays.

        Example:
            >>> engine = PotholeDIPEngine()
            >>> image = cv2.imread("pothole_frame.jpg")
            >>> polygon = np.array([[50, 50], [100, 50], [75, 100]], dtype=np.int32)
            >>> metrics = engine.calculate_metrics(image, polygon)
        """
        # Validate inputs
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected np.ndarray for image, got {type(image)}")
        if not isinstance(polygon_coords, np.ndarray):
            raise TypeError(f"Expected np.ndarray for polygon_coords, got {type(polygon_coords)}")

        if len(polygon_coords) < 3:
            raise ValueError(f"Polygon must have at least 3 points, got {len(polygon_coords)}")

        if image.size == 0:
            raise ValueError("Image array is empty.")

        # Step 1: Create a mask to isolate the pothole
        mask: np.ndarray = self._create_mask(image, polygon_coords)

        # Step 2: Extract pothole region and analyze shadows
        gray_image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pothole_region: np.ndarray = gray_image[mask > 0]

        if pothole_region.size == 0:
            raise ValueError("Pothole mask is empty. Check polygon coordinates.")

        # Step 3: Calculate gradient magnitude using Sobel
        gradient_std: float = self._calculate_shadow_gradient(gray_image, mask)

        # Step 4: Calculate depth from gradient
        depth_cm: float = gradient_std * self.CALIBRATION_CONSTANT

        # Step 5: Calculate 2D area
        area_pixels: float = float(cv2.contourArea(polygon_coords.astype(np.float32)))
        area_cm2: float = area_pixels * self.PIXEL_TO_CM2

        # Step 6: Calculate volume
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
        """
        Create a binary mask for the pothole region.

        Uses cv2.fillPoly to create a binary mask where the pothole region
        is set to 255 and background is 0.

        Args:
            image (np.ndarray): Original RGB image.
            polygon_coords (np.ndarray): Polygon vertices of shape (N, 2).

        Returns:
            np.ndarray: Binary mask (H, W) where pothole = 255, background = 0.
        """
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
        """
        Calculate the standard deviation of gradient magnitude within the mask.

        Uses Sobel operators (X and Y) to compute gradient magnitude, then
        calculates the standard deviation of gradients inside the masked region.

        Args:
            gray_image (np.ndarray): Grayscale image (H, W).
            mask (np.ndarray): Binary mask (H, W) where 255 = region of interest.

        Returns:
            float: Standard deviation of gradient magnitude in the masked region.
        """
        # Apply Sobel in X direction
        sobel_x: np.ndarray = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)

        # Apply Sobel in Y direction
        sobel_y: np.ndarray = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude
        gradient_magnitude: np.ndarray = np.sqrt(sobel_x**2 + sobel_y**2)

        # Apply mask and extract gradients
        masked_gradients: np.ndarray = gradient_magnitude[mask > 0]

        # Return standard deviation
        if masked_gradients.size == 0:
            return 0.0

        return float(np.std(masked_gradients))
