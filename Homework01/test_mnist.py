import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

from model_def import get_model
from argparse import ArgumentParser

import torch
from torch import nn
from deepxube.pytorch.nnet_utils import load_nnet
import time

from emnist import extract_test_samples


def download(url: str, path: Path) -> None:
    """Download a file only if it does not already exist."""
    if not path.exists():
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, path)


def load_images(path: Path) -> np.ndarray:
    """Load MNIST images as an (N, 28, 28) uint8 NumPy array."""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))

        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file: magic={magic}")

        images = np.frombuffer(f.read(), dtype=np.uint8)

    images = images.astype(np.float32) / 255.0
    return images.reshape(n, rows, cols)


def load_labels(path: Path) -> np.ndarray:
    """Load MNIST labels as an (N,) uint8 NumPy array."""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))

        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file: magic={magic}")

        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels


def load_mnist_validation(root="./data"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    image_file = "t10k-images-idx3-ubyte.gz"
    label_file = "t10k-labels-idx1-ubyte.gz"
    image_path = root / image_file
    label_path = root / label_file

    mnist_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    download(mnist_url + image_file, image_path)
    download(mnist_url + label_file, label_path)

    images = load_images(image_path)
    labels = load_labels(label_path)

    images = np.expand_dims(images, 1)
    images = np.repeat(images, 3, axis=1)

    return images, labels

def load_emnist():
    emnist_images, emnist_labels = extract_test_samples('digits')
    emnist_images = emnist_images.astype(np.float32) / 255.0
    emnist_images = emnist_images[:, None, :, :]
    emnist_images = emnist_images.repeat(3, axis=1)
    return emnist_images, emnist_labels


def add_colored_mnist(images, labels):
    n = images.shape[0]
    fg_color = torch.rand(n, 3, 1, 1).numpy()*0.8+0.2
    bg_color = torch.rand(n, 3, 1, 1).numpy()*0.8+0.2
    mask = images[ :, 0:1, :, :]
    colored_images = mask*fg_color + (1-mask)*bg_color
    return colored_images, labels, fg_color, bg_color

def do_test(nnet, images, labels, colored_images, colored_labels, fg_color, bg_color):
    # evaluate nnet for gray scale
    start_time = time.time()
    nnet_out = nnet(torch.tensor(images, device="cpu")).data.cpu().numpy()
    print(f"NNet time: {time.time() - start_time} seconds")

    for label in np.unique(labels):
        label_mask = labels == label
        accuracy_label: float = 100.0 * np.mean(nnet_out[label_mask].argmax(axis=1) == labels[label_mask])
        print(f"Accuracy (for label {label} with {sum(label_mask)} examples): {accuracy_label:.2f}%")

    accuracy: float = 100.0 * np.mean(nnet_out.argmax(axis=1) == labels)
    print(f"Accuracy (total with {labels.shape[0]} examples): {accuracy:.2f}%")

    # evaluate nnet for tinted mnist
    start_time = time.time()
    nnet_out = nnet(torch.tensor(colored_images, dtype=torch.float32, device="cpu")).data.cpu().numpy()
    print(f"NNet time: {time.time() - start_time} seconds")

    preds = nnet_out.argmax(axis=1)

    for label in np.unique(colored_labels):
        label_mask = colored_labels == label
        accuracy_label_colored: float = 100.0 * np.mean(preds[label_mask] == colored_labels[label_mask])
        print(f"Accuracy (for label {label} with {sum(label_mask)} examples): {accuracy_label_colored:.2f}%")

    accuracy_colored: float = 100.0 * np.mean(preds == colored_labels)
    print(f"Accuracy (total with {colored_labels.shape[0]} examples): {accuracy_colored:.2f}%")

    # contrast-based breakdown
    contrast = np.abs(fg_color - bg_color).mean(axis=1).squeeze()
    quartiles = np.percentile(contrast, [25, 50, 75])
    q1_mask = contrast <= quartiles[0]
    q4_mask = contrast > quartiles[2]

    acc_q1 = 100.0 * np.mean(preds[q1_mask] == colored_labels[q1_mask])
    acc_q4 = 100.0 * np.mean(preds[q4_mask] == colored_labels[q4_mask])
    print(f"bottom quartile contrast acc: {acc_q1:.2f}%, top quartile contrast acc: {acc_q4:.2f}%")

def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)

    parser.parse_args()
    args = parser.parse_args()

    torch.manual_seed(42)

    # load nnet
    nnet: nn.Module = get_model()
    nnet = load_nnet(args.model, nnet)
    nnet.eval()

    #### TESTING CODE FOR MNIST ####
    # load data
    images, labels = load_mnist_validation()
    #print(images.min(), images.max(), images.mean())
    colored_images, colored_labels, fg_color, bg_color = add_colored_mnist(images, labels)

    do_test(nnet, images, labels, colored_images, colored_labels, fg_color, bg_color)

    #### TESTING CODE FOR EMNIST ####
    emnist_images, emnist_labels = load_emnist()
    colored_emnist_images, colored_emnist_labels, emnist_fg_color, emnist_bg_color = add_colored_mnist(emnist_images, emnist_labels)
    do_test(nnet, emnist_images, emnist_labels, colored_emnist_images, colored_emnist_labels, emnist_fg_color, emnist_bg_color)



if __name__ == "__main__":
    main()