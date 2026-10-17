"""Train the conditional diffusion model in model_def.py on images rendered by synth_data.py.

    python train.py                  # full run (~5 h on an RTX 4060), writes hw2_model.pt (EMA weights) as it goes
    python train.py --resume         # continue from checkpoints/last.pt

Every --eval_every steps it saves hw2_model.pt, a resumable checkpoint, and a grid of samples in previews/train/.
"""
from typing import Dict, List
from pathlib import Path
from argparse import ArgumentParser
import copy
import math
import time

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import model_def as md
from synth_data import DigitSynthesizer, save_grid, DIGIT_COLORS, BACKGROUNDS, HERE


def drop_conditions(labels: Dict[str, Tensor], p_drop: float, p_drop_all: float) -> List[Tensor]:
    """Network condition inputs from labels, with each condition replaced by its null value with prob p_drop
    (and all of them at once with prob p_drop_all), so the model learns to generate when a condition is None."""
    n: int = labels["digit"].shape[0]
    device: torch.device = labels["digit"].device
    drop_all: Tensor = torch.rand(n, device=device) < p_drop_all

    def dropped() -> Tensor:
        return drop_all | (torch.rand(n, device=device) < p_drop)

    return [
        torch.where(dropped(), md.NULL_DIGIT, labels["digit"]),
        torch.where(dropped(), md.NULL_COLOR, labels["digit_color"]),
        torch.where(dropped(), md.NULL_BACKGROUND, labels["background"]),
        labels["rotation"],
        ~dropped(),
    ]


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    for p_ema, p in zip(ema.parameters(), model.parameters()):
        p_ema.lerp_(p, 1 - decay)


def eval_grid(ema: nn.Module, path: Path) -> None:
    """One row per background; columns sweep color, digit and rotation, plus a fully unconditional column."""
    imgs: List[Tensor] = []
    titles: List[str] = []
    device: torch.device = next(ema.parameters()).device
    for background in BACKGROUNDS:
        for col in range(8):
            color = DIGIT_COLORS[col % len(DIGIT_COLORS)]
            rotation = [-90.0, -45.0, 0.0, 45.0, 90.0, None, 0.0, None][col]
            if col == 7:  # background only
                conds = md.condition_tensors(1, None, None, background, None, device)
                titles.append(f"{background}")
            else:
                conds = md.condition_tensors(1, col, color, background, rotation, device)
                titles.append(f"{col} {color}\n{background} {rotation}")
            imgs.append(md.sample(ema, conds))
    for _ in range(8):
        imgs.append(md.sample(ema, md.condition_tensors(1, None, None, None, None, device)))
        titles.append("none")
    save_grid(path, torch.cat(imgs), titles, n_cols=8)


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--ema_decay", type=float, default=0.9995)
    parser.add_argument("--p_drop", type=float, default=0.15)
    parser.add_argument("--p_drop_all", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=250)
    parser.add_argument("--out", type=str, default=str(HERE / "hw2_model.pt"))
    parser.add_argument("--ckpt_dir", type=str, default=str(HERE / "checkpoints"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device: torch.device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    synth: DigitSynthesizer = DigitSynthesizer(device, seed=args.seed)
    model: nn.Module = md.get_model().to(device)
    ema: nn.Module = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    ab_all: Tensor = md.alphas_cumprod(device)

    ckpt_dir: Path = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    preview_dir: Path = HERE / "previews" / "train"
    preview_dir.mkdir(parents=True, exist_ok=True)

    step: int = 0
    if args.resume:
        ckpt = torch.load(ckpt_dir / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        torch.set_rng_state(ckpt["cpu_rng"].cpu())
        torch.cuda.set_rng_state(ckpt["cuda_rng"].cpu())
        print(f"Resumed from step {step}")

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    loss_sum: float = 0.0
    start_time: float = time.time()
    while step < args.steps:
        # warmup then cosine decay to 10% of the peak learning rate
        lr: float = args.lr * min(1.0, (step + 1) / args.warmup)
        lr *= 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for group in opt.param_groups:
            group["lr"] = lr

        x0, labels = synth.sample(args.batch_size)
        x0 = x0 * 2 - 1
        conds: List[Tensor] = drop_conditions(labels, args.p_drop, args.p_drop_all)
        t: Tensor = torch.randint(md.NUM_TIMESTEPS, (args.batch_size,), device=device)
        ab: Tensor = ab_all[t].view(-1, 1, 1, 1)
        noise: Tensor = torch.randn_like(x0)
        x_t: Tensor = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        v_target: Tensor = ab.sqrt() * noise - (1 - ab).sqrt() * x0

        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_pred: Tensor = model(x_t, t, *conds)
        loss: Tensor = F.mse_loss(v_pred.float(), v_target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        update_ema(ema, model, args.ema_decay if step >= args.warmup else 0.0)
        step += 1

        loss_sum += loss.item()
        if step % args.log_every == 0:
            elapsed: float = time.time() - start_time
            print(f"step {step:6d}  loss {loss_sum / args.log_every:.4f}  lr {lr:.2e}  "
                  f"{args.log_every / elapsed:.1f} it/s", flush=True)
            loss_sum = 0.0
            start_time = time.time()

        if step % args.eval_every == 0 or step == args.steps:
            torch.save({k: v.float().cpu() for k, v in ema.state_dict().items()}, args.out)
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                        "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()},
                       ckpt_dir / "last.pt")
            eval_grid(ema, preview_dir / f"step_{step:06d}.png")
            print(f"Saved {args.out} and samples at step {step}", flush=True)
            start_time = time.time()


if __name__ == "__main__":
    main()
