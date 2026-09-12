from collections import deque

import numpy as np
import pytest

from slic_superpixels import find_boundaries, overlay_boundaries, slic


def four_color_image(size: int = 32) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.float32)
    half = size // 2
    image[:half, :half] = (1.0, 0.0, 0.0)
    image[:half, half:] = (0.0, 1.0, 0.0)
    image[half:, :half] = (0.0, 0.0, 1.0)
    image[half:, half:] = (1.0, 1.0, 0.0)
    return image


def assert_each_label_is_connected(labels: np.ndarray) -> None:
    height, width = labels.shape
    for label in np.unique(labels):
        pixels = set(map(tuple, np.argwhere(labels == label)))
        queue = deque([next(iter(pixels))])
        visited = {queue[0]}
        while queue:
            y, x = queue.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                candidate = (y + dy, x + dx)
                if 0 <= candidate[0] < height and 0 <= candidate[1] < width:
                    if candidate in pixels and candidate not in visited:
                        visited.add(candidate)
                        queue.append(candidate)
        assert visited == pixels


def test_segmentation_is_deterministic_and_connected() -> None:
    image = four_color_image()
    first = slic(image, n_segments=16, max_iter=5)
    second = slic(image, n_segments=16, max_iter=5)
    assert np.array_equal(first.labels, second.labels)
    assert first.labels.dtype == np.int32
    assert first.labels.min() == 0
    assert first.n_superpixels >= 4
    assert_each_label_is_connected(first.labels)


@pytest.mark.parametrize(
    ("image", "segments"),
    [
        (np.zeros((4, 4), dtype=np.float32), 4),
        (np.zeros((2, 3, 3), dtype=np.float32), 4),
        (np.full((4, 4, 3), 2.0, dtype=np.float32), 4),
        (np.zeros((4, 4, 3), dtype=np.float32), 0),
    ],
)
def test_invalid_inputs_are_rejected(image: np.ndarray, segments: int) -> None:
    with pytest.raises(ValueError):
        slic(image, n_segments=segments)


def test_boundaries_and_overlay() -> None:
    image = four_color_image(8)
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[:, 4:] = 1
    boundaries = find_boundaries(labels)
    overlay = overlay_boundaries(image, labels, color=(1.0, 1.0, 1.0))
    assert boundaries.dtype == bool
    assert boundaries.sum() == 8
    assert np.all(overlay[boundaries] == 1.0)
    assert np.array_equal(overlay[~boundaries], image[~boundaries])
