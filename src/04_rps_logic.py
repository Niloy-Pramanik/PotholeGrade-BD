"""
Repair Priority Score (RPS) Logic - Assign urgency scores to potholes.
"""


def calculate_rps(volume_kg: float, pothole_class: str) -> int:
    """
    Calculate Repair Priority Score (RPS) for pothole.

    Priority levels:
        RPS 5: Wet pothole (hidden depth danger)
        RPS 4: Dry pothole > 20 kg (major damage)
        RPS 3: Dry pothole 5-20 kg (standard repair)
        RPS 1: Dry pothole < 5 kg (monitor)

    Args:
        volume_kg: Estimated repair volume in kg (must be >= 0)
        pothole_class: 'Wet_Pothole' or 'Dry_Pothole' (case-insensitive)

    Returns:
        int: RPS score (1-5)

    Raises:
        ValueError: If volume_kg < 0 or invalid pothole_class
        TypeError: If volume_kg not numeric or pothole_class not str
    """
    if not isinstance(volume_kg, (int, float)):
        raise TypeError(f"volume_kg must be numeric, got {type(volume_kg)}")

    if not isinstance(pothole_class, str):
        raise TypeError(f"pothole_class must be str, got {type(pothole_class)}")

    if volume_kg < 0:
        raise ValueError(f"volume_kg must be >= 0, got {volume_kg}")

    pothole_class_upper: str = pothole_class.upper().strip()

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
            f"Invalid pothole_class '{pothole_class}'. "
            f"Expected 'Wet_Pothole' or 'Dry_Pothole'"
        )
