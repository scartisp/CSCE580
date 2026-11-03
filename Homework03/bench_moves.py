"""Measures how often agents pick a move from the best class (exact win > draw > loss) on
solver-labelled positions from solver_data.py.

Only positions where the choice matters are used: not every move has the same class, the player
to move has no immediate win, there is no forced block, and the position is not in the book.
By default only one game in five (id % 5 == 0) is used, so the rest stays available as training
data for any tuning.

Example:
    python bench_moves.py --agents versions/v4_book14.py connect_four_alg.py --time 1.0
"""
import importlib.util
import json
import os
import random
import time
from argparse import ArgumentParser
from multiprocessing import Pool
from typing import Dict, List, Tuple

import c4_solver as S

_module = None


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _init(path: str):
    global _module
    _module = load(path, "bench_agent")


def _run(job: Tuple[int, int, List, float]) -> Tuple[int, bool]:
    pos, mask, classes, limit = job
    _module._tt.clear()
    col = _module._choose_column(pos, mask, time.perf_counter() + limit)
    best = max(c for c in classes if c is not None)
    return mask.bit_count(), classes[col] == best


def select_positions(paths: List[str], min_ply: int, max_ply: int, test_only: bool, limit: int,
                     seed: int) -> List[Dict]:
    import connect_four_alg as alg
    records: List[Dict] = []
    for path in paths:
        with open(path) as f:
            records.extend(json.load(f))
    chosen = []
    for rec in records:
        pos, mask, classes = rec["pos"], rec["mask"], rec["classes"]
        if test_only and rec["game"] % 5 != 0:
            continue
        if not (min_ply <= mask.bit_count() <= max_ply):
            continue
        legal = [c for c in classes if c is not None]
        if len(set(legal)) < 2:
            continue
        possible = S.possible(mask)
        if alg._winning_spots(pos, mask) & possible or alg._winning_spots(pos ^ mask, mask) & possible:
            continue
        if alg._book_move(pos, mask) >= 0:
            continue
        chosen.append(rec)
    random.Random(seed).shuffle(chosen)
    return chosen[:limit]


def main():
    parser = ArgumentParser()
    parser.add_argument('--agents', nargs='+', default=["connect_four_alg.py"])
    parser.add_argument('--data', nargs='+', default=["eval_data.json", "eval_data_close.json"])
    parser.add_argument('--time', type=float, default=1.0)
    parser.add_argument('--min-ply', type=int, default=12)
    parser.add_argument('--max-ply', type=int, default=30)
    parser.add_argument('--limit', type=int, default=600)
    parser.add_argument('--all-games', action='store_true', help="include training games")
    parser.add_argument('--workers', type=int, default=15)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    positions = select_positions(args.data, args.min_ply, args.max_ply, not args.all_games, args.limit,
                                 args.seed)
    print("%d critical positions, %.1fs per move" % (len(positions), args.time))
    jobs = [(rec["pos"], rec["mask"], rec["classes"], args.time) for rec in positions]
    for path in args.agents:
        with Pool(args.workers, initializer=_init, initargs=(path,)) as pool:
            results = pool.map(_run, jobs)
        buckets: Dict[str, List[bool]] = {}
        for ply, ok in results:
            lo = (ply // 6) * 6
            buckets.setdefault("ply %2d-%2d" % (lo, lo + 5), []).append(ok)
        correct = sum(ok for _, ok in results)
        detail = ", ".join("%s: %.1f%%" % (k, 100 * sum(v) / len(v)) for k, v in sorted(buckets.items()))
        print("%-28s %5.1f%% best-class moves (%d/%d)   [%s]" % (path, 100 * correct / len(results),
                                                                  correct, len(results), detail), flush=True)


if __name__ == "__main__":
    main()
