from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ArrayLike = np.ndarray


@dataclass(frozen=True, slots=True)
class SLICResult:
    labels: ArrayLike
    centers: ArrayLike
    lab_image: ArrayLike
    iterations: int
    residual: float

    @property
    def n_superpixels(self) -> int:
        """Number of connected superpixels in the final label map."""
        return int(np.unique(self.labels).size)


def load_rgb_image(path: str | Path) -> ArrayLike:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def save_rgb_image(path: str | Path, image: ArrayLike) -> None:
    clipped = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(clipped, mode="RGB").save(path)


def rgb_to_lab(image: ArrayLike) -> ArrayLike:
    return cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2LAB)


def _compute_gradient_magnitude(lab_image: ArrayLike) -> ArrayLike:
    gradient = np.full(lab_image.shape[:2], np.inf, dtype=np.float32)
    center = lab_image[1:-1, 1:-1]
    right = lab_image[1:-1, 2:]
    down = lab_image[2:, 1:-1]
    diff = (right - center) ** 2 + (down - center) ** 2
    gradient[1:-1, 1:-1] = diff.sum(axis=2)
    return gradient


def _initialize_centers(lab_image: ArrayLike, step: float) -> ArrayLike:
    height, width = lab_image.shape[:2]
    gradient = _compute_gradient_magnitude(lab_image)
    centers: list[list[float]] = []

    offset = step / 2.0
    y_positions = np.arange(offset, height, step)
    x_positions = np.arange(offset, width, step)

    for y in y_positions:
        for x in x_positions:
            cy = int(round(y))
            cx = int(round(x))
            cy = min(max(cy, 1), height - 2)
            cx = min(max(cx, 1), width - 2)

            best_y, best_x = cy, cx
            best_gradient = gradient[cy, cx]
            for ny in range(cy - 1, cy + 2):
                for nx in range(cx - 1, cx + 2):
                    if gradient[ny, nx] < best_gradient:
                        best_gradient = gradient[ny, nx]
                        best_y, best_x = ny, nx

            l_value, a_value, b_value = lab_image[best_y, best_x]
            centers.append([l_value, a_value, b_value, float(best_x), float(best_y)])

    if not centers:
        cy = max(height // 2, 0)
        cx = max(width // 2, 0)
        l_value, a_value, b_value = lab_image[cy, cx]
        centers.append([l_value, a_value, b_value, float(cx), float(cy)])

    return np.asarray(centers, dtype=np.float32)


def _update_centers(lab_image: ArrayLike, labels: ArrayLike, n_centers: int) -> ArrayLike:
    height, width = labels.shape
    flat_labels = labels.reshape(-1)
    valid_mask = flat_labels >= 0
    valid_labels = flat_labels[valid_mask]

    if valid_labels.size == 0:
        raise RuntimeError("No pixels were assigned to SLIC clusters.")

    flat_lab = lab_image.reshape(-1, 3)[valid_mask]
    yy, xx = np.indices((height, width), dtype=np.float32)
    flat_x = xx.reshape(-1)[valid_mask]
    flat_y = yy.reshape(-1)[valid_mask]

    counts = np.bincount(valid_labels, minlength=n_centers).astype(np.float32)
    safe_counts = np.where(counts == 0.0, 1.0, counts)

    sums_l = np.bincount(valid_labels, weights=flat_lab[:, 0], minlength=n_centers)
    sums_a = np.bincount(valid_labels, weights=flat_lab[:, 1], minlength=n_centers)
    sums_b = np.bincount(valid_labels, weights=flat_lab[:, 2], minlength=n_centers)
    sums_x = np.bincount(valid_labels, weights=flat_x, minlength=n_centers)
    sums_y = np.bincount(valid_labels, weights=flat_y, minlength=n_centers)

    centers = np.stack(
        [
            sums_l / safe_counts,
            sums_a / safe_counts,
            sums_b / safe_counts,
            sums_x / safe_counts,
            sums_y / safe_counts,
        ],
        axis=1,
    ).astype(np.float32)

    return centers


def _enforce_connectivity(labels: ArrayLike, step: float) -> ArrayLike:
    height, width = labels.shape
    connected = np.full((height, width), -1, dtype=np.int32)
    min_component_size = max(1, int(step * step / 4.0))
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    current_label = 0

    for y in range(height):
        for x in range(width):
            if connected[y, x] != -1:
                continue

            original_label = labels[y, x]
            queue: deque[tuple[int, int]] = deque([(y, x)])
            component: list[tuple[int, int]] = [(y, x)]
            connected[y, x] = current_label
            adjacent_label = -1

            while queue:
                cy, cx = queue.popleft()
                for dy, dx in neighbors:
                    ny = cy + dy
                    nx = cx + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if (
                        connected[ny, nx] >= 0
                        and connected[ny, nx] != current_label
                        and adjacent_label < 0
                    ):
                        adjacent_label = connected[ny, nx]
                    if connected[ny, nx] != -1:
                        continue
                    if labels[ny, nx] != original_label:
                        continue
                    connected[ny, nx] = current_label
                    queue.append((ny, nx))
                    component.append((ny, nx))

            if len(component) < min_component_size and adjacent_label >= 0:
                for cy, cx in component:
                    connected[cy, cx] = adjacent_label
            else:
                current_label += 1

    return connected


def slic(
    image: ArrayLike,
    n_segments: int,
    compactness: float = 10.0,
    max_iter: int = 10,
    tol: float = 1e-4,
    enforce_connectivity: bool = True,
) -> SLICResult:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")
    if image.shape[0] < 3 or image.shape[1] < 3:
        raise ValueError("The image must be at least 3 x 3 pixels.")
    if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
        raise ValueError("RGB values must be finite and scaled to [0, 1].")
    if n_segments <= 0:
        raise ValueError("n_segments must be positive.")
    if compactness <= 0:
        raise ValueError("compactness must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    lab_image = rgb_to_lab(image)
    height, width = lab_image.shape[:2]
    n_pixels = height * width
    step = max(1.0, math.sqrt(n_pixels / float(n_segments)))

    centers = _initialize_centers(lab_image, step)
    n_centers = centers.shape[0]
    labels = np.full((height, width), -1, dtype=np.int32)
    distances = np.full((height, width), np.inf, dtype=np.float32)
    compactness_scale = (compactness / step) ** 2

    residual = math.inf
    iterations = 0
    for _ in range(max_iter):
        iterations += 1
        distances.fill(np.inf)
        labels.fill(-1)

        for center_index, center in enumerate(centers):
            l_value, a_value, b_value, x_value, y_value = center
            x_start = max(int(x_value - step), 0)
            x_stop = min(int(x_value + step) + 1, width)
            y_start = max(int(y_value - step), 0)
            y_stop = min(int(y_value + step) + 1, height)

            local_lab = lab_image[y_start:y_stop, x_start:x_stop]
            local_distances = distances[y_start:y_stop, x_start:x_stop]
            local_labels = labels[y_start:y_stop, x_start:x_stop]

            yy, xx = np.indices(local_lab.shape[:2], dtype=np.float32)
            xx += x_start
            yy += y_start

            dc2 = (
                (local_lab[:, :, 0] - l_value) ** 2
                + (local_lab[:, :, 1] - a_value) ** 2
                + (local_lab[:, :, 2] - b_value) ** 2
            )
            ds2 = (xx - x_value) ** 2 + (yy - y_value) ** 2
            distance = dc2 + compactness_scale * ds2

            update_mask = distance < local_distances
            local_distances[update_mask] = distance[update_mask]
            local_labels[update_mask] = center_index

        new_centers = _update_centers(lab_image, labels, n_centers)

        # Keep empty clusters at their previous positions rather than letting them collapse to zero.
        valid_labels = labels.reshape(-1)
        valid_labels = valid_labels[valid_labels >= 0]
        counts = np.bincount(valid_labels, minlength=n_centers)
        empty = counts == 0
        new_centers[empty] = centers[empty]

        residual = np.linalg.norm(new_centers - centers, axis=1).sum()
        centers = new_centers
        if residual <= tol:
            break

    if enforce_connectivity:
        labels = _enforce_connectivity(labels, step)

    return SLICResult(
        labels=labels,
        centers=centers,
        lab_image=lab_image,
        iterations=iterations,
        residual=float(residual),
    )


def find_boundaries(labels: ArrayLike) -> ArrayLike:
    if labels.ndim != 2:
        raise ValueError("Expected a two-dimensional label map.")
    boundaries = np.zeros_like(labels, dtype=bool)
    boundaries[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundaries[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return boundaries


def overlay_boundaries(
    image: ArrayLike,
    labels: ArrayLike,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> ArrayLike:
    if labels.shape != image.shape[:2]:
        raise ValueError("The label map must match the image height and width.")
    overlay = image.copy()
    boundaries = find_boundaries(labels)
    overlay[boundaries] = np.asarray(color, dtype=np.float32)
    return overlay
