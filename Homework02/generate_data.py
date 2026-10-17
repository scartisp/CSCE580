from typing import List

from model_def import get_model, generate_image
from argparse import ArgumentParser

from torch import nn
from deepxube.pytorch.nnet_utils import load_nnet
import time
from numpy.typing import NDArray

import matplotlib.pyplot as plt


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--digit", type=int, default=None)
    parser.add_argument("--digit_color", type=str, default=None)
    parser.add_argument("--background", type=str, default=None)
    parser.add_argument("--rotation", type=float, default=None)
    parser.add_argument("--prompt", type=str, default=None)

    parser.parse_args()
    args = parser.parse_args()

    if args.digit is not None:
        assert (0 <= args.digit) and (9 >= args.digit)
    if args.digit_color is not None:
        assert args.digit_color in ["white", "black", "red", "blue", "green"]
    if args.background is not None:
        assert args.background in ["white", "black", "classroom", "horseshoe"]
    if args.rotation is not None:
        assert (-90.0 <= args.rotation) and (90.0 >= args.rotation)


    # load nnet
    nnet: nn.Module = get_model()
    nnet = load_nnet(args.model, nnet)
    nnet.eval()

    start_time = time.time()
    imgs: List[NDArray] = []
    for _ in range(9):
        img: NDArray = generate_image(nnet, args.digit, args.digit_color, args.background, args.rotation, args.prompt)
        img = img.transpose([1, 2, 0])
        imgs.append(img)
    print(f"Image generation time: {time.time() - start_time}")

    fig, axs = plt.subplots(3, 3)
    axs = axs.flatten()

    for ax, img in zip(axs, imgs, strict=True):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.show()
    plt.show()

if __name__ == "__main__":
    main()