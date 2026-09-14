import gzip
import math
import struct
import urllib.request
from pathlib import Path

import numpy as np

from model_def import get_model
from argparse import ArgumentParser

import torch
from torch import nn

from scipy.ndimage import rotate

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
    fg_color = torch.rand(n, 3, 1, 1).numpy() * 0.8 + 0.2
    bg_color = torch.rand(n, 3, 1, 1).numpy() * 0.8 + 0.2
    mask = images[:, 0:1, :, :]
    colored_images = mask * fg_color + (1 - mask) * bg_color
    return colored_images, labels, fg_color, bg_color


def random_rotate(images, max_angle=45):
    """
    Rotates each image by a random angle in [-max_angle, max_angle] degrees.
    images: numpy ndarray, shape (N, 3, H, W)
    returns: numpy ndarray, same shape, rotated
    """
    n = images.shape[0]
    angles = np.random.uniform(-max_angle, max_angle, size=n)

    rotated = np.empty_like(images)
    for i in range(n):
        rotated[i] = rotate(images[i], angle=angles[i], axes=(1, 2),
                             reshape=False, order=1, mode='constant', cval=0.0)

    return rotated


def random_occlude(images, max_patches=2, max_patch_size=12):
    """
    Randomly occludes each image with 0-max_patches rectangular patches,
    filled with a random solid color.
    images: numpy ndarray, shape (N, 3, H, W)
    returns: numpy ndarray, same shape
    """
    n, c, h, w = images.shape
    occluded = images.copy()

    for i in range(n):
        num_patches = np.random.randint(0, max_patches + 1)
        for _ in range(num_patches):
            patch_h = np.random.randint(4, max_patch_size + 1)
            patch_w = np.random.randint(4, max_patch_size + 1)

            top = np.random.randint(0, max(1, h - patch_h))
            left = np.random.randint(0, max(1, w - patch_w))

            patch_color = np.random.rand(c, 1, 1)
            occluded[i, :, top:top+patch_h, left:left+patch_w] = patch_color

    return occluded


def test_gray(nnet, images, labels):
    start_time = time.time()
    nnet_out = nnet(torch.tensor(images, dtype=torch.float32, device="cpu")).data.cpu().numpy()
    print(f"NNet time: {time.time() - start_time} seconds")

    for label in np.unique(labels):
        label_mask = labels == label
        accuracy_label = 100.0 * np.mean(nnet_out[label_mask].argmax(axis=1) == labels[label_mask])
        print(f"Accuracy (for label {label} with {sum(label_mask)} examples): {accuracy_label:.2f}%")

    accuracy = 100.0 * np.mean(nnet_out.argmax(axis=1) == labels)
    print(f"Accuracy (total with {labels.shape[0]} examples): {accuracy:.2f}%")


def test_colored(nnet, colored_images, colored_labels, fg_color, bg_color):
    start_time = time.time()
    nnet_out = nnet(torch.tensor(colored_images, dtype=torch.float32, device="cpu")).data.cpu().numpy()
    print(f"NNet time: {time.time() - start_time} seconds")

    preds = nnet_out.argmax(axis=1)

    for label in np.unique(colored_labels):
        label_mask = colored_labels == label
        accuracy_label = 100.0 * np.mean(preds[label_mask] == colored_labels[label_mask])
        print(f"Accuracy (for label {label} with {sum(label_mask)} examples): {accuracy_label:.2f}%")

    accuracy = 100.0 * np.mean(preds == colored_labels)
    print(f"Accuracy (total with {colored_labels.shape[0]} examples): {accuracy:.2f}%")

    contrast = np.abs(fg_color - bg_color).mean(axis=1).squeeze()
    quartiles = np.percentile(contrast, [25, 50, 75])
    q1_mask = contrast <= quartiles[0]
    q4_mask = contrast > quartiles[2]

    acc_q1 = 100.0 * np.mean(preds[q1_mask] == colored_labels[q1_mask])
    acc_q4 = 100.0 * np.mean(preds[q4_mask] == colored_labels[q4_mask])
    print(f"bottom quartile contrast acc: {acc_q1:.2f}%, top quartile contrast acc: {acc_q4:.2f}%")


def run_full_eval(nnet, images, labels, dataset_name):
    colored_images, colored_labels, fg_color, bg_color = add_colored_mnist(images, labels)
    rotated_images = random_rotate(images)
    colored_rotated_images = random_rotate(colored_images)
    occluded_images = random_occlude(images)
    occluded_rotated_images = random_occlude(rotated_images)

    print(f'\n############### TESTING {dataset_name} GRAY-SCALE ###############\n')
    test_gray(nnet, images, labels)

    print(f'\n############### TESTING {dataset_name} ROTATED ###############\n')
    test_gray(nnet, rotated_images, labels)

    print(f'\n############### TESTING {dataset_name} OCCLUDED ###############\n')
    test_gray(nnet, occluded_images, labels)

    print(f'\n############### TESTING {dataset_name} OCCLUDED + ROTATED ###############\n')
    test_gray(nnet, occluded_rotated_images, labels)

    print(f'\n############### TESTING {dataset_name} COLORED ###############\n')
    test_colored(nnet, colored_images, colored_labels, fg_color, bg_color)

    print(f'\n############### TESTING {dataset_name} COLORED + ROTATED ###############\n')
    test_colored(nnet, colored_rotated_images, colored_labels, fg_color, bg_color)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    nnet = get_model()
    nnet = load_nnet(args.model, nnet)
    nnet.eval()

    images, labels = load_mnist_validation()
    run_full_eval(nnet, images, labels, "MNIST")

    emnist_images, emnist_labels = load_emnist()
    run_full_eval(nnet, emnist_images, emnist_labels, "EMNIST")

if __name__ == "__main__":
    main()