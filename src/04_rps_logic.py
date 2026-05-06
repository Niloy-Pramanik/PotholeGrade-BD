"""
Repair Priority Score (RPS) Logic for PotholeGrade-BD.

Implements decision logic to assign urgency scores to detected potholes
based on volume and classification (wet vs. dry).
"""


def calculate_rps(volume_kg: float, pothole_class: str) -> int:
    """
    Calculate Repair Priority Score (RPS) for a pothole.

    Assigns a priority score (1-5) based on the pothole's volume and
    classification. Wet potholes are marked as highest priority due to
    hidden depth danger. Volume determines secondary priority.

    Priority Levels:
        - RPS 5: Wet pothole (hidden depth danger, water reflection hides true depth)
        - RPS 4: Dry pothole with volume > 20 kg (major structural damage)
        - RPS 3: Dry pothole with volume 5-20 kg (standard repair)
        - RPS 1: Dry pothole with volume < 5 kg (monitor/routine maintenance)

    Args:
        volume_kg (float): Estimated volume of asphalt needed for repair, in kilograms.
            Must be non-negative.
        pothole_class (str): Classification of the pothole. Expected values are
            'Wet_Pothole' or 'Dry_Pothole'. Case-insensitive.

    Returns:
        int: Repair Priority Score in range [1, 5].
            - 1: Low priority (monitor)
            - 3: Medium priority (standard repair)
            - 4: High priority (major damage)
            - 5: Critical priority (wet pothole)

    Raises:
        ValueError: If volume_kg < 0 or pothole_class is invalid.
        TypeError: If volume_kg is not numeric or pothole_class is not str.

    Example:
        >>> calculate_rps(15.5, "Dry_Pothole")
        3
        >>> calculate_rps(25.0, "Dry_Pothole")
        4
        >>> calculate_rps(10.0, "Wet_Pothole")
        5
    """
    # Input validation
    if not isinstance(volume_kg, (int, float)):
        raise TypeError(f"volume_kg must be numeric, got {type(volume_kg)}")

    if not isinstance(pothole_class, str):
        raise TypeError(f"pothole_class must be str, got {type(pothole_class)}")

    if volume_kg < 0:
        raise ValueError(f"volume_kg must be non-negative, got {volume_kg}")

    pothole_class_upper: str = pothole_class.upper().strip()

    # Decision logic
    if pothole_class_upper == "WET_POTHOLE":
        return 5

    elif pothole_class_upper == "DRY_POTHOLE":
        if volume_kg > 20.0:
            return 4
        elif volume_kg > 5.0:
            return 3
        else:
            return 1

    else:
        raise ValueError(
            f"❌ Invalid pothole_class '{pothole_class}'. "
            f"Expected 'Wet_Pothole' or 'Dry_Pothole'."
        )
