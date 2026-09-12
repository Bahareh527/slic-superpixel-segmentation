from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

FloatImage = NDArray[np.float32]


def load_rgb_image(path: str | Path) -> FloatImage:
    """Load an image as an RGB float32 array scaled to [0, 1]."""
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0


def save_rgb_image(path: str | Path, image: NDArray[np.floating]) -> None:
    """Save an RGB image whose values are scaled to [0, 1]."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(image * 255.0, 0, 255).round().astype(np.uint8)
    Image.fromarray(clipped, mode="RGB").save(destination)
