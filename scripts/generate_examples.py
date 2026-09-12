from pathlib import Path

from slic_superpixels import load_rgb_image, overlay_boundaries, save_rgb_image, slic

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples" / "input" / "demo_still_life.png"
OUTPUT = ROOT / "examples" / "output"


def main() -> None:
    image = load_rgb_image(INPUT)
    for count in (100, 200, 400):
        result = slic(image, n_segments=count, compactness=10.0)
        destination = OUTPUT / f"demo_segments_{count}.png"
        save_rgb_image(destination, overlay_boundaries(image, result.labels))
        print(f"requested={count}, produced={result.n_superpixels}")


if __name__ == "__main__":
    main()
