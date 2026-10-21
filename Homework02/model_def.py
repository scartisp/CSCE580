"""Conditional diffusion model for generating 3x28x28 digit images.

The network is a small U-Net trained with v-prediction on a cosine noise schedule. It is conditioned on digit, digit
color, background and rotation. Each condition has a learned "null" value used when it is not given (training drops
conditions at random so the model learns these marginals). Images are sampled with DDIM and classifier-free guidance.

For the extra credit, the network is also conditioned on whether Cocky the mascot is holding the digit, and
parse_prompt turns a free-text prompt into these conditions using keywords plus a pretrained sentence encoder
(all-MiniLM-L6-v2, re-implemented here in PyTorch; see the text encoder section).
"""

"""
Written with Claude Code
"""
from typing import Dict, List, Optional, Tuple
import math
import random
import re
import unicodedata

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from numpy.typing import NDArray

DIGIT_COLORS: List[str] = ["white", "black", "red", "blue", "green"]
BACKGROUNDS: List[str] = ["white", "black", "classroom", "horseshoe", "library", "russell", "cocky"]
# backgrounds the assignment's --background option allows; the rest are extra-credit locations only used for prompts
STANDARD_BACKGROUNDS: List[str] = ["white", "black", "classroom", "horseshoe"]
NULL_DIGIT: int = 10
NULL_COLOR: int = len(DIGIT_COLORS)
NULL_BACKGROUND: int = len(BACKGROUNDS)
NULL_MASCOT: int = 2
MAX_ROTATION: float = 90.0

NUM_TIMESTEPS: int = 1000
SAMPLE_STEPS: int = 30  # tested as good as 50 steps, and keeps CPU-only generation of 9 images around 9 s
GUIDANCE_SCALE: float = 1.5
# the small digit Cocky holds needs stronger guidance to come out readable (61% -> 79% read correctly by an MNIST
# classifier, vs 91% for the training images)
MASCOT_GUIDANCE_SCALE: float = 3.5


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
        # Cocky holding the digit: 0 = no, 1 = yes, NULL_MASCOT = not given. Starts at zero so a model fine-tuned
        # from weights without it starts out behaving exactly like that model.
        self.mascot = nn.Embedding(NULL_MASCOT + 1, emb_dim)
        nn.init.zeros_(self.mascot.weight)

    def forward(self, t: Tensor, digit: Tensor, color: Tensor, background: Tensor, rotation: Tensor,
                has_rotation: Tensor, mascot: Tensor) -> Tensor:
        a: Tensor = (rotation / MAX_ROTATION).clamp(-1.0, 1.0)[:, None]
        k: Tensor = torch.arange(1, self.rot_freqs + 1, device=a.device, dtype=a.dtype)[None]
        rot_feats: Tensor = torch.cat([a, torch.sin(math.pi * k * a / 2), torch.cos(math.pi * k * a / 2)], dim=1)
        rot_emb: Tensor = torch.where(has_rotation[:, None], self.rot_mlp(rot_feats), self.rot_null[None])
        return (self.time_mlp(timestep_embedding(t, self.time_dim)) + self.digit(digit) + self.color(color)
                + self.background(background) + rot_emb + self.mascot(mascot))


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
                has_rotation: Tensor, mascot: Tensor) -> Tensor:
        emb: Tensor = self.cond_emb(t, digit, color, background, rotation, has_rotation, mascot)
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


# ---------------------------------------------------------------- text encoder (extra credit)
#
# A PyTorch re-implementation of the pretrained all-MiniLM-L6-v2 sentence encoder (a 6-layer BERT with mean pooling),
# so prompts can be understood by meaning without any package outside the class environment. Module names mirror the
# original checkpoint so its weights load directly (see build_submission.py). The WordPiece vocabulary is stored in
# the weights file as a byte buffer, since the submission is only this file and the .pt.

TEXT_VOCAB_SIZE: int = 30522
TEXT_DIM: int = 384
TEXT_LAYERS: int = 6
TEXT_HEADS: int = 12
TEXT_FFN: int = 1536
TEXT_MAX_LEN: int = 128
TEXT_VOCAB_BYTES: int = 231508  # size of the vocabulary file, one token per line


class BertSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(TEXT_DIM, TEXT_DIM)
        self.key = nn.Linear(TEXT_DIM, TEXT_DIM)
        self.value = nn.Linear(TEXT_DIM, TEXT_DIM)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        b, n, _ = x.shape
        q, k, v = [proj(x).view(b, n, TEXT_HEADS, -1).transpose(1, 2) for proj in (self.query, self.key, self.value)]
        out: Tensor = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        return out.transpose(1, 2).reshape(b, n, TEXT_DIM)


class BertDenseNorm(nn.Module):
    """Linear projection, then LayerNorm of (projection + residual)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.dense = nn.Linear(in_dim, TEXT_DIM)
        self.LayerNorm = nn.LayerNorm(TEXT_DIM, eps=1e-12)

    def forward(self, x: Tensor, residual: Tensor) -> Tensor:
        return self.LayerNorm(self.dense(x) + residual)


class BertAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.self = BertSelfAttention()
        self.output = BertDenseNorm(TEXT_DIM)


class BertIntermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(TEXT_DIM, TEXT_FFN)


class BertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = BertAttention()
        self.intermediate = BertIntermediate()
        self.output = BertDenseNorm(TEXT_FFN)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.attention.output(self.attention.self(x, mask), x)
        return self.output(F.gelu(self.intermediate.dense(x)), x)


class BertEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(TEXT_VOCAB_SIZE, TEXT_DIM)
        self.position_embeddings = nn.Embedding(512, TEXT_DIM)
        self.token_type_embeddings = nn.Embedding(2, TEXT_DIM)
        self.LayerNorm = nn.LayerNorm(TEXT_DIM, eps=1e-12)

    def forward(self, ids: Tensor) -> Tensor:
        pos: Tensor = torch.arange(ids.shape[1], device=ids.device)[None]
        return self.LayerNorm(self.word_embeddings(ids) + self.position_embeddings(pos)
                              + self.token_type_embeddings(torch.zeros_like(ids)))


class BertEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer() for _ in range(TEXT_LAYERS)])


class TextEncoder(nn.Module):
    """Sentence embeddings (unit length, 384-d) for lists of strings."""

    def __init__(self):
        super().__init__()
        self.embeddings = BertEmbeddings()
        self.encoder = BertEncoder()
        self.register_buffer("vocab_bytes", torch.zeros(TEXT_VOCAB_BYTES, dtype=torch.uint8))
        self._vocab: Optional[Dict[str, int]] = None

    def vocab(self) -> Dict[str, int]:
        if self._vocab is None:
            text: str = bytes(self.vocab_bytes.cpu().numpy().tobytes()).decode("utf-8")
            self._vocab = {tok: i for i, tok in enumerate(text.split("\n")) if tok}
        return self._vocab

    def tokenize(self, text: str) -> List[int]:
        """BERT uncased tokenization: lowercase, strip accents, split off punctuation, then greedy WordPiece."""
        vocab: Dict[str, int] = self.vocab()
        text = unicodedata.normalize("NFD", text.lower())
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        words: List[str] = re.findall(r"\w+|[^\w\s]", text)
        ids: List[int] = [vocab["[CLS]"]]
        for word in words:
            pieces: List[int] = []
            start: int = 0
            while start < len(word):
                end: int = len(word)
                while end > start and (("##" if start else "") + word[start:end]) not in vocab:
                    end -= 1
                if end == start:  # no piece matches: the whole word is unknown
                    pieces = [vocab["[UNK]"]]
                    break
                pieces.append(vocab[("##" if start else "") + word[start:end]])
                start = end
            ids.extend(pieces)
        return ids[:TEXT_MAX_LEN - 1] + [vocab["[SEP]"]]

    @torch.no_grad()
    def forward(self, texts: List[str]) -> Tensor:
        device: torch.device = self.vocab_bytes.device
        token_ids: List[List[int]] = [self.tokenize(t) for t in texts]
        length: int = max(len(t) for t in token_ids)
        ids: Tensor = torch.zeros(len(texts), length, dtype=torch.long, device=device)
        mask: Tensor = torch.zeros(len(texts), length, dtype=torch.bool, device=device)
        for i, t in enumerate(token_ids):
            ids[i, :len(t)] = torch.tensor(t, device=device)
            mask[i, :len(t)] = True
        x: Tensor = self.embeddings(ids)
        for layer in self.encoder.layer:
            x = layer(x, mask)
        pooled: Tensor = (x * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return F.normalize(pooled, dim=-1)


class HW2Model(nn.Module):
    """What get_model returns: the image network plus the text encoder used for prompts."""

    def __init__(self):
        super().__init__()
        self.unet = UNet()
        self.text = TextEncoder()


# ---------------------------------------------------------------- diffusion

def alphas_cumprod(device: torch.device, num_timesteps: int = NUM_TIMESTEPS, s: float = 0.008) -> Tensor:
    """Cosine noise schedule (Nichol & Dhariwal 2021): fraction of signal variance left at each timestep."""
    steps: Tensor = torch.arange(num_timesteps + 1, dtype=torch.float64, device=device) / num_timesteps
    f: Tensor = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    betas: Tensor = (1 - f[1:] / f[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0).float()


def condition_tensors(n: int, digit: Optional[int], digit_color: Optional[str], background: Optional[str],
                      rotation: Optional[float], device: torch.device, mascot: Optional[bool] = None) -> List[Tensor]:
    """Batched condition inputs for the network, using the null value for anything not given."""
    return [
        torch.full((n,), NULL_DIGIT if digit is None else int(digit), dtype=torch.long, device=device),
        torch.full((n,), NULL_COLOR if digit_color is None else DIGIT_COLORS.index(digit_color), dtype=torch.long,
                   device=device),
        torch.full((n,), NULL_BACKGROUND if background is None else BACKGROUNDS.index(background), dtype=torch.long,
                   device=device),
        torch.full((n,), 0.0 if rotation is None else float(rotation), device=device),
        torch.full((n,), rotation is not None, dtype=torch.bool, device=device),
        torch.full((n,), NULL_MASCOT if mascot is None else int(mascot), dtype=torch.long, device=device),
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
        background = random.choice([b for b in STANDARD_BACKGROUNDS if b != digit_color])
    if digit_color is None:
        digit_color = random.choice([c for c in DIGIT_COLORS if c != background])
    return digit, digit_color, background, rotation


# ---------------------------------------------------------------- prompts (extra credit)

NUMBER_WORDS: Dict[str, int] = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                                "eight": 8, "nine": 9}
COLOR_WORDS: Dict[str, str] = {
    "white": "white", "ivory": "white",
    "black": "black", "ebony": "black",
    "red": "red", "garnet": "red", "crimson": "red", "scarlet": "red", "maroon": "red", "cardinal": "red",
    "blue": "blue", "navy": "blue", "azure": "blue", "cobalt": "blue", "royal blue": "blue",
    "green": "green", "emerald": "green", "lime": "green", "forest green": "green",
}
# phrases naming each background; matched as whole words, longest phrase first
LOCATION_WORDS: Dict[str, List[str]] = {
    "horseshoe": ["horseshoe", "the shoe", "historic horseshoe", "maxcy monument", "rutledge chapel",
                  "mckissick museum", "president's house", "presidents house"],
    "classroom": ["classroom", "class room", "lecture hall", "lecture room", "our class", "in class",
                  "csce 580", "csce580", "where we have class"],
    "library": ["thomas cooper library", "thomas cooper", "cooper library", "tcl", "library", "the library"],
    "russell": ["russell house", "russell", "student union", "the union"],
    "cocky": ["cocky statue", "statue of cocky", "cocky's statue", "cocky on a bench", "cocky on the bench",
              "gamecock statue", "mascot statue", "statue"],
}
# words that put Cocky in the picture, holding the digit (the statue phrases above choose the statue background)
MASCOT_WORDS: List[str] = ["cocky", "mascot", "gamecock", "gamecocks"]

# For prompts the keywords above miss, the text encoder compares short spans of the prompt with these descriptions
# and picks the closest one if it is similar enough (thresholds set on example prompts, see extra_credit_prompts.txt).
LOCATION_CONCEPTS: Dict[str, List[str]] = {
    "classroom": ["a classroom", "a lecture hall with desks and chairs", "a college class", "a room where students "
                  "take notes"],
    "horseshoe": ["the historic horseshoe", "a grassy campus quad with old oak trees", "a college lawn with brick "
                  "paths", "the oldest part of campus", "a lawn under big old trees"],
    "library": ["thomas cooper library", "the main campus library", "a library full of books", "a place to study "
                "and check out books"],
    "russell": ["russell house", "the student union", "the student center", "a building where students eat lunch"],
    "cocky": ["the statue of cocky", "a bronze statue sitting on a bench"],
}
MASCOT_CONCEPTS: List[str] = ["cocky the gamecock mascot", "the usc mascot", "a cartoon rooster", "a gamecock bird",
                              "a chicken"]
LOCATION_MATCH: float = 0.67
MASCOT_MATCH: float = 0.6
STOPWORDS: set = {"a", "an", "the", "at", "in", "on", "of", "to", "by", "with", "and", "is", "it", "its", "near",
                  "next", "front", "behind", "outside", "inside", "holding", "holds", "number", "digit", "numeral",
                  "picture", "image", "photo", "drawing", "standing", "sitting", "from", "for", "as", "up", "this",
                  "student", "students", "wrote", "write", "writes", "written"}
ROTATION_WORDS: Dict[str, float] = {"upright": 0.0, "straight up": 0.0, "not rotated": 0.0, "unrotated": 0.0,
                                    "level": 0.0}
TILT_DEGREES: float = 30.0
SLIGHT_TILT_DEGREES: float = 15.0
PROMPT_DEFAULT_TILT: float = 15.0  # max random tilt when a prompt does not mention rotation


def _remove(text: str, match: re.Match) -> str:
    return text[:match.start()] + " " + text[match.end():]


def _parse_rotation(text: str) -> Tuple[Optional[float], str]:
    """Rotation in degrees (positive = counter-clockwise) and the prompt with the rotation phrase removed."""
    # explicit angle, e.g. "rotated 45 degrees clockwise", "tilted -30°", "at a 60 degree angle to the left"
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:°|degrees?|deg\b)"
                  r"(?:\s*(?:angle\s*)?(?:to\s*the\s*|towards?\s*the\s*)?"
                  r"(counter[\s-]?clockwise|anti[\s-]?clockwise|clockwise|left|right|ccw|cw)\b)?", text)
    if m is not None:
        angle: float = float(m.group(1))
        direction: Optional[str] = m.group(2)
        if direction is None:  # direction may come before the angle: "turned left 45 degrees"
            before: str = text[max(0, m.start() - 40):m.start()]
            dirs = re.findall(r"(counter[\s-]?clockwise|anti[\s-]?clockwise|clockwise|left|right)", before)
            direction = dirs[-1] if dirs else None
        if direction is not None:
            ccw: bool = direction.startswith(("counter", "anti", "ccw", "left"))
            angle = abs(angle) if ccw else -abs(angle)
        return max(-MAX_ROTATION, min(MAX_ROTATION, angle)), _remove(text, m)

    for phrase, angle in ROTATION_WORDS.items():
        m = re.search(rf"\b{re.escape(phrase)}\b", text)
        if m is not None:
            return angle, _remove(text, m)

    # "sideways", "on its side", "lying down": a quarter turn either way
    m = re.search(r"\b(sideways|on (?:its|it's|his|her|their) side|lying down|laying down|horizontal(?:ly)?)\b", text)
    if m is not None:
        return random.choice([-MAX_ROTATION, MAX_ROTATION]), _remove(text, m)

    # "tilted left", "leaning to the right", "slightly slanted"
    m = re.search(r"\b(slightly |a little |a bit )?(tilt(?:ed|ing)?|lean(?:s|ed|ing)?|slant(?:ed|ing)?|tipped|"
                  r"crooked|rotated|turned|askew)\b(?:\s*(?:to\s*the\s*|towards?\s*the\s*)?"
                  r"(left|right|counter[\s-]?clockwise|anti[\s-]?clockwise|clockwise))?", text)
    if m is not None:
        size: float = SLIGHT_TILT_DEGREES if m.group(1) else TILT_DEGREES
        direction = m.group(3)
        if direction is None:
            return random.choice([-size, size]), _remove(text, m)
        ccw = direction.startswith(("counter", "anti", "left"))
        return (size if ccw else -size), _remove(text, m)
    return None, text


def _parse_background(text: str) -> Tuple[Optional[str], str]:
    """Background name and the prompt with the background phrase removed."""
    # plain backgrounds: "white background", "on a black backdrop", "background is white", "on plain white"
    m = re.search(r"\b(white|black)\s+(?:background|backdrop|backround|bg)\b", text)
    if m is None:
        m = re.search(r"\b(?:background|backdrop)\s+(?:is\s+|of\s+)?(?:plain\s+|solid\s+|all\s+)?(white|black)\b", text)
    if m is None:
        m = re.search(r"\bon\s+(?:a\s+)?(?:plain|solid|blank|empty)\s+(white|black)\b", text)
    if m is not None:
        return m.group(1), _remove(text, m)

    phrases: List[Tuple[str, str]] = sorted(((p, name) for name, ps in LOCATION_WORDS.items() if name in BACKGROUNDS
                                             for p in ps), key=lambda pn: -len(pn[0]))
    for phrase, name in phrases:
        m = re.search(rf"\b{re.escape(phrase)}\b", text)
        if m is not None:
            return name, _remove(text, m)
    return None, text


def _parse_mascot(text: str) -> Tuple[bool, str]:
    """Whether the prompt names Cocky / the mascot, and the prompt with that word removed."""
    for word in MASCOT_WORDS:
        m = re.search(rf"\b{re.escape(word)}\b", text)
        if m is not None:
            return True, _remove(text, m)
    return False, text


def _spans(text: str, max_words: int = 5) -> List[str]:
    """Every run of up to max_words consecutive words that is not just filler or numbers."""
    words: List[str] = re.findall(r"[a-z0-9']+", text)
    return [" ".join(words[i:j]) for i in range(len(words)) for j in range(i + 1, min(len(words), i + max_words) + 1)
            if not all(w in STOPWORDS or w.isdigit() for w in words[i:j])]


def _closest_concept(encoder: "TextEncoder", text: str, concepts: Dict[str, List[str]]) -> Tuple[Optional[str], float]:
    """The concept whose descriptions best match some span of text, and that cosine similarity."""
    spans: List[str] = _spans(text)
    if not spans:
        return None, 0.0
    key: Tuple[str, ...] = tuple(p for ps in concepts.values() for p in ps)
    cache: Dict[Tuple[str, ...], Tensor] = encoder.__dict__.setdefault("_concept_cache", {})
    if key not in cache:
        cache[key] = encoder(list(key))
    sims: Tensor = encoder(spans) @ cache[key].T  # (spans, phrases)
    best: Tensor = sims.max(dim=0).values
    names: List[str] = [name for name, ps in concepts.items() for _ in ps]
    i: int = int(best.argmax())
    return names[i], float(best[i])


def _parse_color(text: str) -> Tuple[Optional[str], str]:
    """Digit color named first in the prompt, and the prompt with it removed."""
    best: Optional[re.Match] = None
    best_color: Optional[str] = None
    for word, color in COLOR_WORDS.items():
        m = re.search(rf"\b{re.escape(word)}\b", text)
        if m is not None and (best is None or m.start() < best.start()):
            best, best_color = m, color
    return (best_color, _remove(text, best)) if best is not None else (None, text)


def _parse_digit(text: str) -> Optional[int]:
    """A single digit written as a numeral ("a 5", "#7") or a word ("seven"). 'one' is only used if nothing else
    matches, since it is often not a number ("one of the desks")."""
    m = re.search(r"(?<![\d.])(\d)(?![\d.])", text)
    if m is not None:
        return int(m.group(1))
    words: List[Tuple[int, int]] = [(m.start(), NUMBER_WORDS[w]) for w in NUMBER_WORDS if w != "one"
                                    for m in [re.search(rf"\b{w}\b", text)] if m is not None]
    if words:
        return min(words)[1]
    return 1 if re.search(r"\bone\b", text) else None


def parse_prompt(prompt: str, encoder: Optional["TextEncoder"] = None
                 ) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[float], bool]:
    """Turn a free-text prompt into (digit, digit_color, background, rotation, mascot); anything not mentioned is
    None (mascot False).

    Rotation and background phrases are parsed (and removed) first, so "rotated 45 degrees" is not read as a digit
    and "black background" is not read as a black digit. Keywords decide whatever they can; the text encoder (if
    given) then looks for a place or the mascot described in other words.
    """
    text: str = " " + prompt.lower().replace("’", "'") + " "
    rotation, text = _parse_rotation(text)
    background, text = _parse_background(text)
    mascot, text = _parse_mascot(text)
    digit_color, text = _parse_color(text)
    digit: Optional[int] = _parse_digit(text)
    if encoder is not None:
        if background is None:
            name, sim = _closest_concept(encoder, text, LOCATION_CONCEPTS)
            if sim >= LOCATION_MATCH:
                background = name
        if not mascot:
            _, sim = _closest_concept(encoder, text, {"mascot": MASCOT_CONCEPTS})
            mascot = sim >= MASCOT_MATCH
    return digit, digit_color, background, rotation, mascot


# ---------------------------------------------------------------- required interface

def get_model() -> nn.Module:
    return HW2Model()


def generate_image(model: nn.Module, digit: Optional[int], digit_color: Optional[str], background: Optional[str],
                   rotation: Optional[float], prompt: Optional[str]) -> NDArray:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if next(model.parameters()).device != device:
        model.to(device)
    model.eval()

    mascot: bool = False
    if prompt is not None:  # the prompt replaces the other conditions
        digit, digit_color, background, rotation, mascot = parse_prompt(prompt, model.text)
        if rotation is None:  # a scene description implies a roughly upright digit
            rotation = random.uniform(-PROMPT_DEFAULT_TILT, PROMPT_DEFAULT_TILT)

    digit, digit_color, background, rotation = fill_missing_conditions(digit, digit_color, background, rotation)
    conds: List[Tensor] = condition_tensors(1, digit, digit_color, background, rotation, device, mascot=mascot)
    guidance: float = MASCOT_GUIDANCE_SCALE if mascot else GUIDANCE_SCALE
    return sample(model.unet, conds, guidance_scale=guidance)[0].cpu().numpy()


def main():
    nnet: nn.Module = get_model()
    print(f"Parameters: {sum(p.numel() for p in nnet.parameters()):,}")
    torch.save(nnet.state_dict(), "dummy.pt")


if __name__ == "__main__":
    main()
