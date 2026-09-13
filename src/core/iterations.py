"""Validation shared by image rendering and saved settings."""
def validate_iterations(value) -> int:
    message = "Iterations must be a whole number of 1 or more."
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if isinstance(value, bool) or count != value or count < 1:
        raise ValueError(message)
    return count
