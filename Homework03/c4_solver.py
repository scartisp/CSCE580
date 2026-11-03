"""ctypes wrapper around book_solver.dll, the exact solver used offline to build the opening book.

Also usable as an arena agent ("python hw3_arena.py --b c4_solver.py") that plays perfectly; it can
be very slow in the first few moves of a game.
"""
import ctypes
import json
import os
import random
from typing import Dict, List

from connect_four import ConnectFourState, ConnectFour

WIDTH = 7
HEIGHT = 6
H1 = HEIGHT + 1
SIZE = WIDTH * HEIGHT
BOTTOM = sum(1 << (col * H1) for col in range(WIDTH))
BOARD = BOTTOM * ((1 << HEIGHT) - 1)
COL_MASKS = [((1 << HEIGHT) - 1) << (col * H1) for col in range(WIDTH)]
MOVE_ORDER = [3, 2, 4, 1, 5, 0, 6]

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "book_solver.dll"))
_lib.solve.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]
_lib.solve.restype = ctypes.c_int
_lib.reset.argtypes = []
_lib.reset.restype = None
_lib.node_count.argtypes = []
_lib.node_count.restype = ctypes.c_uint64


def solve(pos: int, mask: int, weak: bool = False) -> int:
    """ Score for the player to move: 22 - k for a win on their k-th stone, negative for a loss,
    0 for a draw. With weak=True only the sign is meaningful. """
    return _lib.solve(pos, mask, int(weak))


def reset():
    _lib.reset()


def node_count() -> int:
    return _lib.node_count()


def possible(mask: int) -> int:
    return (mask + BOTTOM) & BOARD


def is_win(pos: int) -> bool:
    for shift in (1, H1, HEIGHT, H1 + 1):
        pair = pos & (pos >> shift)
        if pair & (pair >> (2 * shift)):
            return True
    return False


def play(pos: int, mask: int, col: int):
    """ Returns (pos, mask) after the player to move drops a stone in col """
    move = possible(mask) & COL_MASKS[col]
    return pos ^ mask, mask | move


def legal_cols(mask: int) -> List[int]:
    return [col for col in range(WIDTH) if possible(mask) & COL_MASKS[col]]


def mirror(board: int) -> int:
    out = 0
    for col in range(WIDTH):
        out |= ((board >> (col * H1)) & 0x7F) << ((WIDTH - 1 - col) * H1)
    return out


def move_scores(pos: int, mask: int, weak: bool = False) -> Dict[int, int]:
    """ Score of each legal column for the player to move """
    moves = mask.bit_count()
    scores: Dict[int, int] = {}
    for col in legal_cols(mask):
        move = possible(mask) & COL_MASKS[col]
        if is_win(pos | move):
            scores[col] = (SIZE + 1 - moves) // 2
        elif moves + 1 == SIZE:
            scores[col] = 0
        else:
            opp, new_mask = pos ^ mask, mask | move
            scores[col] = -solve(opp, new_mask, weak)
            if weak:
                scores[col] = (scores[col] > 0) - (scores[col] < 0)
    return scores


def grid_to_bitboards(state: ConnectFourState):
    """ (pos, mask) with Max (1) as the player to move """
    pos, mask = 0, 0
    for col in range(WIDTH):
        for row in range(HEIGHT):
            if state.grid[col, row] != 0:
                bit = 1 << (col * H1 + (HEIGHT - 1 - row))
                mask |= bit
                if state.grid[col, row] > 0:
                    pos |= bit
    return pos, mask


_cache: Dict[int, Dict[int, int]] = {}


def _cached_scores(pos: int, mask: int) -> Dict[int, int]:
    """ Exact move scores from opening_book_cache.json, which skips the slow solves of the first
    few plies. Returns {} if the position is not cached with exact scores. """
    if not _cache:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opening_book_cache.json")
        if os.path.exists(path):
            with open(path) as f:
                for key, entry in json.load(f).items():
                    if entry["exact"]:
                        _cache[int(key)] = {int(c): s for c, s in entry["exact"].items()}
        _cache[-1] = {}  # mark as loaded
    key = pos + mask
    if key in _cache:
        return _cache[key]
    mirror_key = mirror(key)
    if mirror_key in _cache:
        return {WIDTH - 1 - c: s for c, s in _cache[mirror_key].items()}
    return {}


def make_move(state: ConnectFourState, env: ConnectFour) -> int:
    """ Perfect play: best exact score, ties broken at random """
    pos, mask = grid_to_bitboards(state)
    if mask == 0:
        return 3
    scores = _cached_scores(pos, mask) or move_scores(pos, mask)
    best = max(scores.values())
    return random.choice([col for col, score in scores.items() if score == best])
