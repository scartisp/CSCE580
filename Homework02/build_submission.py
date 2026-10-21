"""Build the submitted weights file: the trained image network (from train.py) plus the pretrained all-MiniLM-L6-v2
text encoder converted into model_def.TextEncoder.

    python build_submission.py --unet hw2_unet_v7.pt --minilm_dir <folder with model.safetensors and vocab.txt> \
        --out hw2_model_v7.pt

The MiniLM files come from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2. The safetensors format is
read directly (a JSON header followed by raw arrays), so no package outside the class environment is needed.
"""
from typing import Dict
from pathlib import Path
from argparse import ArgumentParser
import json
import struct

import numpy as np
import torch
from torch import Tensor

import model_def as md

SAFETENSORS_DTYPES: Dict[str, np.dtype] = {"F32": np.float32, "F16": np.float16, "I64": np.int64}


def load_safetensors(path: Path) -> Dict[str, Tensor]:
    data: bytes = path.read_bytes()
    header_len: int = struct.unpack("<Q", data[:8])[0]
    header: dict = json.loads(data[8:8 + header_len])
    body: memoryview = memoryview(data)[8 + header_len:]
    tensors: Dict[str, Tensor] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        arr: np.ndarray = np.frombuffer(body[start:end], dtype=SAFETENSORS_DTYPES[info["dtype"]]).reshape(info["shape"])
        tensors[name] = torch.from_numpy(arr.copy())
    return tensors


def text_encoder_state(minilm_dir: Path) -> Dict[str, Tensor]:
    """TextEncoder state dict from the MiniLM checkpoint (its module names match ours) and vocabulary."""
    weights: Dict[str, Tensor] = load_safetensors(minilm_dir / "model.safetensors")
    state: Dict[str, Tensor] = {k: v.float() for k, v in weights.items()
                                if k.startswith(("embeddings.", "encoder.")) and k != "embeddings.position_ids"}
    vocab: bytes = (minilm_dir / "vocab.txt").read_bytes()
    assert len(vocab) == md.TEXT_VOCAB_BYTES, len(vocab)
    state["vocab_bytes"] = torch.frombuffer(bytearray(vocab), dtype=torch.uint8)
    return state


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--unet", type=str, required=True, help="image network weights saved by train.py")
    parser.add_argument("--minilm_dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    model: md.HW2Model = md.get_model()
    missing, unexpected = model.unet.load_state_dict(torch.load(args.unet, map_location="cpu", weights_only=True),
                                                     strict=False)
    assert not unexpected and set(missing) <= {"cond_emb.mascot.weight"}, (missing, unexpected)
    model.text.load_state_dict(text_encoder_state(Path(args.minilm_dir)))
    torch.save(model.state_dict(), args.out)
    print(f"Saved {args.out} ({Path(args.out).stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
