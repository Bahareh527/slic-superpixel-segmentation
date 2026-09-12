from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .core import overlay_boundaries, slic
from .io import load_rgb_image, save_rgb_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segment an RGB image with SLIC superpixels.")
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument("--segments", type=int, default=200, help="Approximate superpixel count.")
    parser.add_argument("--compactness", type=float, default=10.0, help="Shape regularity weight.")
    parser.add_argument("--max-iter", type=int, default=10, help="Maximum clustering iterations.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Convergence threshold.")
    parser.add_argument("--output", type=Path, default=Path("slic_boundaries.png"))
    parser.add_argument("--labels-output", type=Path, help="Optional NumPy label-map output.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    image = load_rgb_image(args.image)
    result = slic(image, args.segments, args.compactness, args.max_iter, args.tol)
    save_rgb_image(args.output, overlay_boundaries(image, result.labels))
    if args.labels_output:
        args.labels_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.labels_output, result.labels)
    print(
        f"Generated {result.n_superpixels} connected superpixels "
        f"in {result.iterations} iteration(s); residual={result.residual:.6g}."
    )
    print(f"Boundary overlay saved to: {args.output.resolve()}")
