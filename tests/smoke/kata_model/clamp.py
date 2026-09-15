"""Intentionally buggy fixture for a real-model repair smoke."""


def clamp(value, lower, upper):
    """Return value restricted to the inclusive interval [lower, upper]."""
    return max(lower, value)
