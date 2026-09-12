from pathlib import Path

import numpy as np

from slic_superpixels import load_rgb_image, save_rgb_image
from slic_superpixels.cli import main


def test_image_round_trip(tmp_path: Path) -> None:
    image = np.linspace(0, 1, 8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
    path = tmp_path / "image.png"
    save_rgb_image(path, image)
    loaded = load_rgb_image(path)
    assert loaded.shape == image.shape
    assert np.max(np.abs(loaded - image)) <= 1 / 255


def test_cli_writes_an_overlay(tmp_path: Path) -> None:
    image = np.zeros((16, 16, 3), dtype=np.float32)
    image[:, 8:] = 1.0
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    labels = tmp_path / "labels.npy"
    save_rgb_image(source, image)
    main(
        [
            str(source),
            "--segments",
            "8",
            "--max-iter",
            "3",
            "--output",
            str(output),
            "--labels-output",
            str(labels),
        ]
    )
    assert output.exists()
    assert labels.exists()
    assert np.load(labels).shape == image.shape[:2]
