"""Simple Linear Iterative Clustering (SLIC) superpixels."""

from .core import SLICResult, find_boundaries, overlay_boundaries, slic
from .io import load_rgb_image, save_rgb_image

__all__ = [
    "SLICResult",
    "find_boundaries",
    "load_rgb_image",
    "overlay_boundaries",
    "save_rgb_image",
    "slic",
]
