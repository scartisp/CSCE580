"""Conditional diffusion model for generating 3x28x28 digit images.

The network is a small U-Net trained with v-prediction on a cosine noise schedule. It is conditioned on digit, digit
color, background and rotation. Each condition has a learned "null" value used when it is not given (training drops
conditions at random so the model learns these marginals). Images are sampled with DDIM and classifier-free guidance.
"""
from typing import List, Optional, Tuple
import math
import random

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from numpy.typing import NDArray

DIGIT_COLORS: List[str] = ["white", "black", "red", "blue", "green"]
BACKGROUNDS: List[str] = ["white", "black", "classroom", "horseshoe"]
NULL_DIGIT: int = 10
NULL_COLOR: int = len(DIGIT_COLORS)
NULL_BACKGROUND: int = len(BACKGROUNDS)
MAX_ROTATION: float = 90.0

NUM_TIMESTEPS: int = 1000
SAMPLE_STEPS: int = 50
GUIDANCE_SCALE: float = 1.5


# ---------------------------------------------------------------- network

def timestep_embedding(t: Tensor, dim: int) -> Tensor:
    half: int = dim // 2
    freqs: Tensor = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args: Tensor = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, 2 * out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h: Tensor = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.num_heads, c // self.num_heads, h * w).unbind(1)
        out: Tensor = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h, w))


class ConditionEmbedding(nn.Module):
    """Sum of embeddings for timestep, digit, color, background and rotation (each with a null value)."""

    def __init__(self, emb_dim: int, time_dim: int = 128, rot_freqs: int = 6):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.digit = nn.Embedding(NULL_DIGIT + 1, emb_dim)
        self.color = nn.Embedding(NULL_COLOR + 1, emb_dim)
        self.background = nn.Embedding(NULL_BACKGROUND + 1, emb_dim)
        self.rot_freqs = rot_freqs
        self.rot_mlp = nn.Sequential(nn.Linear(1 + 2 * rot_freqs, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.rot_null = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, t: Tensor, digit: Tensor, color: Tensor, background: Tensor, rotation: Tensor,
                has_rotation: Tensor) -> Tensor:
        a: Tensor = (rotation / MAX_ROTATION).clamp(-1.0, 1.0)[:, None]
        k: Tensor = torch.arange(1, self.rot_freqs + 1, device=a.device, dtype=a.dtype)[None]
        rot_feats: Tensor = torch.cat([a, torch.sin(math.pi * k * a / 2), torch.cos(math.pi * k * a / 2)], dim=1)
        rot_emb: Tensor = torch.where(has_rotation[:, None], self.rot_mlp(rot_feats), self.rot_null[None])
        return (self.time_mlp(timestep_embedding(t, self.time_dim)) + self.digit(digit) + self.color(color)
                + self.background(background) + rot_emb)


class UNet(nn.Module):
    """U-Net over 28x28 -> 14x14 -> 7x7, with self-attention at 7x7."""

    def __init__(self, base_ch: int = 64, ch_mults: List[int] = (1, 2, 4), blocks_per_level: int = 2,
                 emb_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.cond_emb = ConditionEmbedding(emb_dim)
        self.in_conv = nn.Conv2d(3, base_ch, 3, padding=1)
        attn_level: int = len(ch_mults) - 1

        self.down = nn.ModuleList()
        skip_chs: List[int] = [base_ch]
        ch: int = base_ch
        for level, mult in enumerate(ch_mults):
            for _ in range(blocks_per_level):
                out_ch: int = base_ch * mult
                self.down.append(nn.ModuleList([ResBlock(ch, out_ch, emb_dim, dropout),
                                                Attention(out_ch) if level == attn_level else nn.Identity()]))
                ch = out_ch
                skip_chs.append(ch)
            if level != len(ch_mults) - 1:
                self.down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                skip_chs.append(ch)

        self.mid1 = ResBlock(ch, ch, emb_dim, dropout)
        self.mid_attn = Attention(ch)
        self.mid2 = ResBlock(ch, ch, emb_dim, dropout)

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(ch_mults))):
            for _ in range(blocks_per_level + 1):
                out_ch = base_ch * mult
                self.up.append(nn.ModuleList([ResBlock(ch + skip_chs.pop(), out_ch, emb_dim, dropout),
                                              Attention(out_ch) if level == attn_level else nn.Identity()]))
                ch = out_ch
            if level != 0:
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                             nn.Conv2d(ch, ch, 3, padding=1)))

        self.out = nn.Sequential(nn.GroupNorm(32, ch), nn.SiLU(), nn.Conv2d(ch, 3, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x: Tensor, t: Tensor, digit: Tensor, color: Tensor, background: Tensor, rotation: Tensor,
                has_rotation: Tensor) -> Tensor:
        emb: Tensor = self.cond_emb(t, digit, color, background, rotation, has_rotation)
        h: Tensor = self.in_conv(x)
        hs: List[Tensor] = [h]
        for layer in self.down:
            if isinstance(layer, nn.ModuleList):
                h = layer[1](layer[0](h, emb))
            else:
                h = layer(h)
            hs.append(h)

        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)

        for layer in self.up:
            if isinstance(layer, nn.ModuleList):
                h = layer[1](layer[0](torch.cat([h, hs.pop()], dim=1), emb))
            else:
                h = layer(h)
        return self.out(h)


# ---------------------------------------------------------------- diffusion

def alphas_cumprod(device: torch.device, num_timesteps: int = NUM_TIMESTEPS, s: float = 0.008) -> Tensor:
    """Cosine noise schedule (Nichol & Dhariwal 2021): fraction of signal variance left at each timestep."""
    steps: Tensor = torch.arange(num_timesteps + 1, dtype=torch.float64, device=device) / num_timesteps
    f: Tensor = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    betas: Tensor = (1 - f[1:] / f[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0).float()


def condition_tensors(n: int, digit: Optional[int], digit_color: Optional[str], background: Optional[str],
                      rotation: Optional[float], device: torch.device) -> List[Tensor]:
    """Batched condition inputs for the network, using the null value for anything not given."""
    return [
        torch.full((n,), NULL_DIGIT if digit is None else int(digit), dtype=torch.long, device=device),
        torch.full((n,), NULL_COLOR if digit_color is None else DIGIT_COLORS.index(digit_color), dtype=torch.long,
                   device=device),
        torch.full((n,), NULL_BACKGROUND if background is None else BACKGROUNDS.index(background), dtype=torch.long,
                   device=device),
        torch.full((n,), 0.0 if rotation is None else float(rotation), device=device),
        torch.full((n,), rotation is not None, dtype=torch.bool, device=device),
    ]


@torch.no_grad()
def sample(model: nn.Module, conds: List[Tensor], steps: int = SAMPLE_STEPS,
           guidance_scale: float = GUIDANCE_SCALE) -> Tensor:
    """DDIM sampling with classifier-free guidance. Returns images (B, 3, 28, 28) in [0, 1]."""
    device: torch.device = conds[0].device
    n: int = conds[0].shape[0]
    ab_all: Tensor = alphas_cumprod(device)

    # guidance only makes sense when some condition is set; the unconditional branch has every condition nulled
    uncond: List[Tensor] = condition_tensors(n, None, None, None, None, device)
    use_guidance: bool = guidance_scale != 1.0 and not all(torch.equal(c, u) for c, u in zip(conds, uncond))
    if use_guidance:
        conds = [torch.cat([c, u]) for c, u in zip(conds, uncond)]

    timesteps: List[int] = torch.linspace(NUM_TIMESTEPS - 1, 0, steps).round().long().tolist()
    x: Tensor = torch.randn(n, 3, 28, 28, device=device)
    for i, t in enumerate(timesteps):
        ab: Tensor = ab_all[t]
        ab_prev: Tensor = ab_all[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=device)
        x_in: Tensor = torch.cat([x, x]) if use_guidance else x
        v: Tensor = model(x_in, torch.full((x_in.shape[0],), t, device=device), *conds).float()
        if use_guidance:
            v_cond, v_uncond = v.chunk(2)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        x0: Tensor = (ab.sqrt() * x - (1 - ab).sqrt() * v).clamp(-1.0, 1.0)
        eps: Tensor = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


# ---------------------------------------------------------------- required interface

def fill_missing_conditions(digit: Optional[int], digit_color: Optional[str], background: Optional[str],
                            rotation: Optional[float]) -> Tuple[int, str, str, float]:
    """Pick a random value for every condition that was not given.

    Sampling the conditions first and then generating with guidance gives much sharper images than letting the
    network fill in everything itself, with the same spread of digits, colors, backgrounds and angles. A random
    color never matches a plain white/black background (and vice versa), since that digit would be invisible.
    """
    if digit is None:
        digit = random.randrange(10)
    if rotation is None:
        rotation = random.uniform(-MAX_ROTATION, MAX_ROTATION)
    if background is None:
        background = random.choice([b for b in BACKGROUNDS if b != digit_color])
    if digit_color is None:
        digit_color = random.choice([c for c in DIGIT_COLORS if c != background])
    return digit, digit_color, background, rotation


def get_model() -> nn.Module:
    return UNet()


def generate_image(model: nn.Module, digit: Optional[int], digit_color: Optional[str], background: Optional[str],
                   rotation: Optional[float], prompt: Optional[str]) -> NDArray:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if next(model.parameters()).device != device:
        model.to(device)
    model.eval()

    if prompt is not None:  # prompts are not supported, and the spec says to ignore the other conditions then
        digit, digit_color, background, rotation = None, None, None, None

    digit, digit_color, background, rotation = fill_missing_conditions(digit, digit_color, background, rotation)
    conds: List[Tensor] = condition_tensors(1, digit, digit_color, background, rotation, device)
    return sample(model, conds)[0].cpu().numpy()


def main():
    nnet: nn.Module = get_model()
    print(f"Parameters: {sum(p.numel() for p in nnet.parameters()):,}")
    torch.save(nnet.state_dict(), "dummy.pt")


if __name__ == "__main__":
    main()
