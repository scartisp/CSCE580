"""Find MNIST digits that look like a different digit (e.g. 5s with a closed loop that read as 6s) and save the
indices of the unambiguous ones, for training with `train.py --mnist_subset data/mnist_clean_idx.npy`.

A digit is kept when the HW1 MNIST classifier gives its true label at least --min_prob probability. Indices refer to
the order load_mnist returns (train then test). Without --mnist_subset, training uses all of MNIST as before.

    python filter_mnist.py
"""
from pathlib import Path
from argparse import ArgumentParser
import importlib.util

import numpy as np
import torch

from deepxube.pytorch.nnet_utils import load_nnet
from synth_data import load_mnist, save_grid, HERE


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--classifier", type=str, default=str(HERE.parent / "Homework01" / "son_of_anton.pt"))
    parser.add_argument("--min_prob", type=float, default=0.5)
    parser.add_argument("--out", type=str, default=str(HERE / "data" / "mnist_clean_idx.npy"))
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("hw1_model_def", HERE.parent / "Homework01" / "model_def.py")
    hw1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hw1)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = load_nnet(args.classifier, hw1.get_model()).to(device).eval()

    imgs, labels = load_mnist(HERE.parent / "Homework01" / "data")
    x: torch.Tensor = torch.from_numpy(imgs).float().div(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
    probs: list = []
    with torch.no_grad():
        for i in range(0, len(x), 4096):
            probs.append(clf(x[i:i + 4096].to(device)).cpu())
    p_true: torch.Tensor = torch.cat(probs)[torch.arange(len(labels)), torch.from_numpy(labels)]
    keep: np.ndarray = (p_true >= args.min_prob).numpy()

    for d in range(10):
        n: int = int((labels == d).sum())
        print(f"digit {d}: dropped {n - int(keep[labels == d].sum()):4d} of {n}")
    print(f"total: kept {int(keep.sum())} of {len(keep)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, np.nonzero(keep)[0])
    print(f"Saved {args.out}")

    dropped: np.ndarray = np.nonzero(~keep)[0]
    show: np.ndarray = np.random.default_rng(0).choice(dropped, size=min(45, len(dropped)), replace=False)
    save_grid(HERE / "previews" / "mnist_dropped.png", x[show], [f"label {labels[i]}" for i in show], n_cols=9)


if __name__ == "__main__":
    main()
