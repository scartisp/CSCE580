"""Generates solver-labelled positions for benchmarking connect_four_alg.py (see bench_moves.py).

Games are played by a noisy strong policy: our agent (short time limit) for the first plies, then a
random move from the solver's best class (win > draw > loss). Either way, a fraction (--epsilon) of
moves is replaced by a random legal move so the games cover the kinds of mistakes real opponents
make. Every
position from --min-ply on is recorded with the class of each legal move for the player to move.

Example:
    python solver_data.py --games 1000 --epsilon 0.05 --out eval_data_close.json --workers 15
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

AGENT_PLIES = 10
EPSILON = 0.2
AGENT_TIME = 0.05
# a fixed snapshot of the agent, so retuning connect_four_alg.py does not change the data
POLICY_AGENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "versions", "v4_book14.py")
_agent_module = None


def _policy_agent():
    global _agent_module
    if _agent_module is None:
        spec = importlib.util.spec_from_file_location("policy_agent", POLICY_AGENT)
        _agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_agent_module)
    return _agent_module


def play_game(job: Tuple[int, int, float]) -> List[Dict]:
    seed, min_ply, epsilon = job
    rng = random.Random(seed)
    alg = _policy_agent()

    records: List[Dict] = []
    pos, mask = 0, 0
    while True:
        legal = S.legal_cols(mask)
        ply = mask.bit_count()
        classes: Dict[int, int] = {}
        if ply >= min_ply:
            classes = S.move_scores(pos, mask, weak=True)
            # immediate wins come back as distance scores; the class is just the sign
            classes = {c: (s > 0) - (s < 0) for c, s in classes.items()}
            records.append({"game": seed, "pos": pos, "mask": mask,
                            "classes": [classes.get(c) for c in range(S.WIDTH)]})

        if rng.random() < epsilon:
            col = rng.choice(legal)
        elif ply < max(AGENT_PLIES, min_ply):
            col = alg._choose_column(pos, mask, time.perf_counter() + AGENT_TIME)
        else:
            best = max(classes.values())
            col = rng.choice([c for c in legal if classes[c] == best])

        move = S.possible(mask) & S.COL_MASKS[col]
        if S.is_win(pos | move) or (mask | move) == S.BOARD:
            return records
        pos, mask = S.play(pos, mask, col)


def main():
    parser = ArgumentParser()
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--min-ply', type=int, default=10)
    parser.add_argument('--epsilon', type=float, default=EPSILON, help="chance of a random move")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=15)
    parser.add_argument('--out', type=str, default="eval_data.json")
    args = parser.parse_args()

    records: List[Dict] = []
    start = time.perf_counter()
    jobs = [(args.seed + idx, args.min_ply, args.epsilon) for idx in range(args.games)]
    with Pool(args.workers) as pool:
        for idx, recs in enumerate(pool.imap_unordered(play_game, jobs)):
            records.extend(recs)
            if (idx + 1) % 50 == 0 or idx + 1 == len(jobs):
                print("%d/%d games, %d positions, %.0fs" % (idx + 1, len(jobs), len(records),
                                                            time.perf_counter() - start), flush=True)
    with open(args.out, "w") as f:
        json.dump(records, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
