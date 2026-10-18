"""Synthetic training data for HW2: MNIST digits composited onto backgrounds.

Each sample is an MNIST digit (train + test, 70k images) that is rotated, colored and pasted over a background:
plain white/black or a random 28x28 crop of one of the photos in classroom/ or horseshoe/. Samples are rendered on
the fly on the GPU, so training never sees the exact same image twice.

This file is only used for training. The submitted model_def.py does not depend on it.

Run directly to build the background crop cache and save preview grids:
    python synth_data.py
"""
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from argparse import ArgumentParser
import gzip
import math
import time

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image, ImageOps

HERE: Path = Path(__file__).resolve().parent

DIGIT_COLORS: List[str] = ["white", "black", "red", "blue", "green"]
BACKGROUNDS: List[str] = ["white", "black", "classroom", "horseshoe"]
PHOTO_BACKGROUNDS: List[str] = ["classroom", "horseshoe"]
COLOR_RGB: Tensor = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
MAX_ROTATION: float = 90.0
PHOTO_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")


def load_mnist(mnist_dir: Path) -> Tuple[NDArray, NDArray]:
    """MNIST train + test as (N, 28, 28) uint8 images and (N,) int64 labels."""
    imgs: List[NDArray] = []
    labels: List[NDArray] = []
    for split in ["train", "t10k"]:
        with gzip.open(mnist_dir / f"{split}-images-idx3-ubyte.gz", "rb") as f:
            imgs.append(np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28))
        with gzip.open(mnist_dir / f"{split}-labels-idx1-ubyte.gz", "rb") as f:
            labels.append(np.frombuffer(f.read(), np.uint8, offset=8))
    return np.concatenate(imgs), np.concatenate(labels).astype(np.int64)


def photo_paths(photo_dir: Path) -> List[Path]:
    return sorted(p for p in photo_dir.iterdir() if p.suffix.lower() in PHOTO_EXTS)


def load_photo(path: Path, short_side: int) -> Image.Image:
    """Load a photo upright and downscaled so its short side is short_side pixels."""
    im: Image.Image = Image.open(path)
    im.draft("RGB", (short_side * 2, short_side * 2))  # fast reduced-size JPEG decode
    im = ImageOps.exif_transpose(im).convert("RGB")
    scale: float = short_side / min(im.size)
    return im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)


def build_background_bank(paths: List[Path], n_crops: int, rng: np.random.Generator, short_side: int = 384,
                          min_frac: float = 0.4) -> NDArray:
    """Random crops of the photos, resized to (n_crops, 28, 28, 3) uint8.

    Each crop covers 40-100% of the photo's short side (so some show the whole scene, some zoom in), with a
    slightly random aspect ratio and a random horizontal flip.
    """
    photos: List[Image.Image] = [load_photo(p, short_side) for p in paths]
    crops: NDArray = np.empty((n_crops, 28, 28, 3), np.uint8)
    for i in range(n_crops):
        im: Image.Image = photos[rng.integers(len(photos))]
        w, h = im.size
        side: float = rng.uniform(min_frac, 1.0) * min(w, h)
        aspect: float = math.exp(rng.uniform(math.log(0.8), math.log(1.25)))
        cw: float = min(w, side * math.sqrt(aspect))
        ch: float = min(h, side / math.sqrt(aspect))
        x0: float = rng.uniform(0, w - cw)
        y0: float = rng.uniform(0, h - ch)
        crop: Image.Image = im.resize((28, 28), Image.LANCZOS, box=(x0, y0, x0 + cw, y0 + ch))
        if rng.random() < 0.5:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        crops[i] = np.asarray(crop)
    return crops


def load_background_banks(cache_path: Path, n_crops: int, seed: int, rebuild: bool = False) -> Dict[str, NDArray]:
    """Crop banks for each photo background, cached in cache_path.

    The cache is rebuilt when asked to, or when the photo folders or crop count no longer match it.
    """
    paths: Dict[str, List[Path]] = {name: photo_paths(HERE / name) for name in PHOTO_BACKGROUNDS}
    for name, ps in paths.items():
        assert len(ps) > 0, f"no photos found in {HERE / name}"
    file_list: NDArray = np.array([f"{name}/{p.name}" for name in PHOTO_BACKGROUNDS for p in paths[name]])

    if cache_path.exists() and not rebuild:
        cache = np.load(cache_path)
        if (np.array_equal(cache["files"], file_list)
                and all(cache[name].shape[0] == n_crops for name in PHOTO_BACKGROUNDS)):
            return {name: cache[name] for name in PHOTO_BACKGROUNDS}
        print("Background photos changed, rebuilding crop cache")

    rng: np.random.Generator = np.random.default_rng(seed)
    banks: Dict[str, NDArray] = {}
    for name in PHOTO_BACKGROUNDS:
        start_time = time.time()
        banks[name] = build_background_bank(paths[name], n_crops, rng)
        print(f"Built {n_crops} {name} crops from {len(paths[name])} photos in {time.time() - start_time:.1f}s")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, files=file_list, **banks)
    return banks


def rotate(imgs: Tensor, degrees: Tensor) -> Tensor:
    """Rotate each image in (B, C, H, W) counter-clockwise by its own angle (same convention as torchvision)."""
    rad: Tensor = degrees * (math.pi / 180.0)
    cos, sin = rad.cos(), rad.sin()
    zero: Tensor = torch.zeros_like(cos)
    theta: Tensor = torch.stack([torch.stack([cos, -sin, zero], 1), torch.stack([sin, cos, zero], 1)], 1)
    grid: Tensor = F.affine_grid(theta, list(imgs.shape), align_corners=False)
    return F.grid_sample(imgs, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def stroke_visibility(bg: Tensor, alpha: Tensor, fg: Tensor, min_dist: float = 0.25) -> Tensor:
    """Fraction of the digit's stroke (weighted by alpha) lying on background pixels whose RGB distance from the
    digit color, scaled to [0, 1], is at least min_dist. Leading dims are batch dims; the last three are (C, H, W),
    with alpha having C = 1 and fg having H = W = 1."""
    dist: Tensor = (bg - fg).pow(2).sum(dim=-3, keepdim=True).sqrt() / math.sqrt(3.0)
    return (alpha * (dist >= min_dist).float()).sum(dim=(-3, -2, -1)) / alpha.sum(dim=(-3, -2, -1)).clamp(min=1e-6)


class DigitSynthesizer:
    """Renders batches of colored, rotated MNIST digits on backgrounds, with their condition labels."""

    def __init__(self, device: torch.device, mnist_dir: Path = HERE.parent / "Homework01" / "data",
                 cache_path: Path = HERE / "data" / "background_crops.npz", crops_per_background: int = 30000,
                 seed: int = 0, rebuild_cache: bool = False, min_visibility: float = 0.0, visibility_tries: int = 8):
        self.device: torch.device = device
        self.min_visibility: float = min_visibility
        self.visibility_tries: int = visibility_tries

        imgs, labels = load_mnist(mnist_dir)
        order: NDArray = np.argsort(labels, kind="stable")  # group by class so a digit can be sampled uniformly
        self.digits: Tensor = torch.from_numpy(imgs[order]).to(device)
        counts: NDArray = np.bincount(labels, minlength=10)
        self.class_start: Tensor = torch.from_numpy(np.concatenate([[0], np.cumsum(counts)[:-1]])).to(device)
        self.class_count: Tensor = torch.from_numpy(counts).to(device)

        banks: Dict[str, NDArray] = load_background_banks(cache_path, crops_per_background, seed, rebuild_cache)
        self.photo_banks: List[Tensor] = [torch.from_numpy(banks[name]).permute(0, 3, 1, 2).contiguous().to(device)
                                          for name in PHOTO_BACKGROUNDS]
        self.colors: Tensor = COLOR_RGB.to(device)

    def sample(self, batch_size: int, digit: Optional[int] = None, digit_color: Optional[int] = None,
               background: Optional[int] = None, rotation: Optional[float] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        """A random batch of images (B, 3, 28, 28) in [0, 1] and their labels.

        Every condition is sampled uniformly unless it is given, in which case it is fixed for the whole batch.
        Colors and backgrounds are indices into DIGIT_COLORS and BACKGROUNDS. Labels are digit, digit_color,
        background (int64) and rotation (float, degrees).
        """
        labels: Dict[str, Tensor] = {
            "digit": self._fill_or_sample(digit, 10, batch_size),
            "digit_color": self._fill_or_sample(digit_color, len(DIGIT_COLORS), batch_size),
            "background": self._fill_or_sample(background, len(BACKGROUNDS), batch_size),
        }
        if rotation is None:
            labels["rotation"] = (torch.rand(batch_size, device=self.device) * 2 - 1) * MAX_ROTATION
        else:
            labels["rotation"] = torch.full((batch_size,), float(rotation), device=self.device)

        offset: Tensor = (torch.rand(batch_size, device=self.device) * self.class_count[labels["digit"]]).long()
        digit_idx: Tensor = self.class_start[labels["digit"]] + offset
        return self.render(digit_idx, labels["digit_color"], labels["background"], labels["rotation"]), labels

    def render(self, digit_idx: Tensor, digit_color: Tensor, background: Tensor, rotation: Tensor) -> Tensor:
        """Composite: alpha * color + (1 - alpha) * background, with alpha the rotated MNIST intensity."""
        alpha: Tensor = rotate(self.digits[digit_idx].float().div(255.0).unsqueeze(1), rotation)
        fg: Tensor = self.colors[digit_color].view(-1, 3, 1, 1)
        return alpha * fg + (1.0 - alpha) * self._backgrounds(background, alpha, fg)

    def _backgrounds(self, background: Tensor, alpha: Tensor, fg: Tensor) -> Tensor:
        """Backgrounds for each sample. With min_visibility > 0, each photo background is the first of
        visibility_tries random crops where at least min_visibility of the digit's stroke stands out from it (or the
        most visible one if none does), so the digit never disappears into a same-colored part of the photo."""
        bg: Tensor = torch.ones(background.shape[0], 3, 28, 28, device=self.device)
        bg[background == BACKGROUNDS.index("black")] = 0.0
        tries: int = self.visibility_tries if self.min_visibility > 0 else 1
        for name, bank in zip(PHOTO_BACKGROUNDS, self.photo_banks):
            mask: Tensor = background == BACKGROUNDS.index(name)
            n: int = int(mask.sum())
            if n == 0:
                continue
            crop_idx: Tensor = torch.randint(bank.shape[0], (n, tries), device=self.device)
            brightness: Tensor = torch.empty(n, tries, 1, 1, 1, device=self.device).uniform_(0.85, 1.15)
            cands: Tensor = (bank[crop_idx].float().div(255.0) * brightness).clamp(0.0, 1.0)  # (n, tries, 3, 28, 28)
            if tries == 1:
                bg[mask] = cands[:, 0]
                continue
            visible: Tensor = stroke_visibility(cands, alpha[mask].unsqueeze(1), fg[mask].unsqueeze(1))  # (n, tries)
            ok: Tensor = visible >= self.min_visibility
            choice: Tensor = torch.where(ok.any(dim=1), ok.float().argmax(dim=1), visible.argmax(dim=1))
            bg[mask] = cands[torch.arange(n, device=self.device), choice]
        return bg

    def _fill_or_sample(self, value: Optional[int], n_values: int, batch_size: int) -> Tensor:
        if value is None:
            return torch.randint(n_values, (batch_size,), device=self.device)
        return torch.full((batch_size,), value, dtype=torch.long, device=self.device)


def save_grid(path: Path, imgs: Tensor, titles: List[str], n_cols: int, cell_size: float = 1.4) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows: int = math.ceil(len(titles) / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * cell_size, n_rows * (cell_size + 0.3)))
    for i, ax in enumerate(np.array(axs).flatten()):
        ax.axis("off")
        if i < len(titles):
            ax.imshow(imgs[i].permute(1, 2, 0).cpu().numpy(), interpolation="nearest")
            ax.set_title(titles[i], fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--crops", type=int, default=30000, help="cached crops per photo background")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the background crop cache")
    parser.add_argument("--out_dir", type=str, default=str(HERE / "previews"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synth: DigitSynthesizer = DigitSynthesizer(device, crops_per_background=args.crops, seed=args.seed,
                                               rebuild_cache=args.rebuild)
    out_dir: Path = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # throughput check: rendering should be negligible next to a training step
    synth.sample(512)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        synth.sample(512)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"Rendering speed: {100 * 512 / (time.time() - start_time):,.0f} images/s on {device}")

    # random samples with all conditions free
    imgs, labels = synth.sample(48)
    titles: List[str] = [f"{labels['digit'][i]} {DIGIT_COLORS[labels['digit_color'][i]]}\n"
                         f"{BACKGROUNDS[labels['background'][i]]} {labels['rotation'][i]:.0f}°" for i in range(48)]
    save_grid(out_dir / "random_samples.png", imgs, titles, n_cols=8)

    # every background x color combination, no rotation
    cells: List[Tensor] = []
    titles = []
    for bg_idx, bg_name in enumerate(BACKGROUNDS):
        for color_idx, color_name in enumerate(DIGIT_COLORS):
            img, labels = synth.sample(1, digit_color=color_idx, background=bg_idx, rotation=0.0)
            cells.append(img)
            titles.append(f"{labels['digit'][0]} {color_name}\n{bg_name}")
    save_grid(out_dir / "colors_x_backgrounds.png", torch.cat(cells), titles, n_cols=len(DIGIT_COLORS))

    # rotation sweep on one fixed digit, to check the angle convention (positive = counter-clockwise)
    angles: List[float] = [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0]
    digit_idx: Tensor = synth.class_start[7].repeat(len(angles))
    imgs = synth.render(digit_idx, torch.full((len(angles),), DIGIT_COLORS.index("red"), device=device),
                        torch.full((len(angles),), BACKGROUNDS.index("white"), device=device),
                        torch.tensor(angles, device=device))
    save_grid(out_dir / "rotation_sweep.png", imgs, [f"{a:.0f}°" for a in angles], n_cols=len(angles))

    # background variety: 24 crops of each photo background
    for bg_idx in [BACKGROUNDS.index(name) for name in PHOTO_BACKGROUNDS]:
        imgs, labels = synth.sample(24, background=bg_idx)
        save_grid(out_dir / f"{BACKGROUNDS[bg_idx]}_samples.png", imgs, [""] * 24, n_cols=8)

    print(f"Saved previews to {out_dir}")


if __name__ == "__main__":
    main()
