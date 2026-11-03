"""Plays Connect Four agents against each other and reports results and move times.

Agents are "random", "basic" (depth-4 minimax counting pieces in a row, as described in hw3.pdf),
or a path to a .py file / module name that defines make_move(state, env).

Unlike run_connect_four.py, every agent is shown the board from its own side (its stones are 1 and
it is Max's turn), so minimax opponents play for themselves.

Examples:
    python hw3_arena.py --a connect_four_alg.py --b random --games 10
    python hw3_arena.py --a connect_four_alg.py --b basic --games 10 --opening 2 --workers 4
    python hw3_arena.py --a connect_four_alg.py --b versions/v1.py --time 1.0
"""
import importlib
import importlib.util
import os
import random
import time
from argparse import ArgumentParser
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from connect_four import ConnectFourState, ConnectFour

GRID_SHAPE = (7, 6)
Agent = Callable[[ConnectFourState, ConnectFour], int]


def random_agent(state: ConnectFourState, env: ConnectFour) -> int:
    return int(np.random.choice(env.get_actions(state)))


def basic_heuristic(state: ConnectFourState) -> float:
    """ Pieces in a row: total length of every run of 2+ stones, Max's minus Min's """
    score = 0.0
    for line in state.get_lines():
        run_player, run_len = 0, 0
        for val in list(line) + [0]:
            if val != 0 and val == run_player:
                run_len += 1
            else:
                if run_player != 0 and run_len >= 2:
                    score += run_player * run_len
                run_player, run_len = val, 1
    return score


def basic_agent(state: ConnectFourState, env: ConnectFour, depth: int = 4) -> int:
    """ Depth-limited minimax with alpha-beta (same choices as plain minimax), random tie-breaks """
    def value(s: ConnectFourState, d: int, alpha: float, beta: float, maximize: bool) -> float:
        if env.is_terminal(s):
            return env.utility(s)
        if d == 0:
            return basic_heuristic(s)
        if maximize:
            v = -np.inf
            for a in env.get_actions(s):
                v = max(v, value(env.next_state(s, a), d - 1, alpha, beta, False))
                alpha = max(alpha, v)
                if alpha >= beta:
                    break
        else:
            v = np.inf
            for a in env.get_actions(s):
                v = min(v, value(env.next_state(s, a), d - 1, alpha, beta, True))
                beta = min(beta, v)
                if alpha >= beta:
                    break
        return v

    actions = env.get_actions(state)
    vals = [value(env.next_state(state, a), depth - 1, -np.inf, np.inf, False) for a in actions]
    best = max(vals)
    return random.choice([a for a, v in zip(actions, vals) if v == best])


def load_agent(spec: str, role: str, time_limit: Optional[float]) -> Agent:
    if spec == "random":
        return random_agent
    if spec == "basic":
        return basic_agent

    if spec.endswith(".py"):
        # unique module name per role so an agent can play a copy of itself
        mod_spec = importlib.util.spec_from_file_location("agent_%s" % role, os.path.abspath(spec))
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(spec)

    if time_limit is not None and hasattr(module, "TIME_LIMIT"):
        module.TIME_LIMIT = time_limit
    return module.make_move


_agents: Dict[str, Agent] = {}


def _init_worker(spec_a: str, spec_b: str, time_limit: Optional[float]):
    _agents["a"] = load_agent(spec_a, "a", time_limit)
    _agents["b"] = load_agent(spec_b, "b", time_limit)


def random_opening(env: ConnectFour, num_plies: int, seed: int) -> List[int]:
    """ A random sequence of opening moves that does not end the game """
    rng = random.Random(seed)
    while True:
        state = ConnectFourState(np.zeros(GRID_SHAPE), 1)
        moves: List[int] = []
        for _ in range(num_plies):
            moves.append(rng.choice(env.get_actions(state)))
            state = env.next_state(state, moves[-1])
            if env.is_terminal(state):
                break
        if not env.is_terminal(state):
            return moves


def play_game(job: Tuple[bool, List[int], int]) -> Dict:
    """ Returns the winner from a's point of view (1, 0, -1) and per-agent move times """
    a_first, opening, seed = job
    np.random.seed(seed)
    random.seed(seed)
    env = ConnectFour()

    # in the real game the first player's stones are 1
    state = ConnectFourState(np.zeros(GRID_SHAPE), 1)
    for move in opening:
        state = env.next_state(state, move)
    sign_a = 1 if a_first else -1

    times: Dict[str, List[float]] = {"a": [], "b": []}
    moves: List[int] = list(opening)
    illegal: Optional[str] = None
    while not env.is_terminal(state):
        sign = state.player_turn
        role = "a" if sign == sign_a else "b"
        view = ConnectFourState(state.grid * sign, 1)

        start = time.perf_counter()
        action = _agents[role](view, env)
        times[role].append(time.perf_counter() - start)

        if action not in env.get_actions(state):
            illegal = role
            break
        moves.append(int(action))
        state = env.next_state(state, action)

    if illegal is not None:
        result = -1 if illegal == "a" else 1
    else:
        result = int(np.sign(env.utility(state) * sign_a))
    return {"a_first": a_first, "result": result, "times": times, "moves": moves, "illegal": illegal}


def main():
    parser = ArgumentParser()
    parser.add_argument('--a', type=str, default="connect_four_alg.py")
    parser.add_argument('--b', type=str, default="basic")
    parser.add_argument('--games', type=int, default=4, help="total games, split evenly by who goes first")
    parser.add_argument('--opening', type=int, default=0, help="random plies played before agents take over")
    parser.add_argument('--time', type=float, default=None, help="override TIME_LIMIT of search agents")
    parser.add_argument('--workers', type=int, default=1, help="parallel games (inflates move times)")
    parser.add_argument('--a-side', choices=["both", "first", "second"], default="both",
                        help="which side A plays (both: alternate)")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    env = ConnectFour()
    jobs: List[Tuple[bool, List[int], int]] = []
    if args.a_side == "both":
        for pair in range((args.games + 1) // 2):
            opening = random_opening(env, args.opening, args.seed + pair)
            jobs.append((True, opening, args.seed + 2 * pair))
            jobs.append((False, opening, args.seed + 2 * pair + 1))
        jobs = jobs[:args.games]
    else:
        for idx in range(args.games):
            opening = random_opening(env, args.opening, args.seed + idx)
            jobs.append((args.a_side == "first", opening, args.seed + idx))

    results: List[Dict] = []
    with Pool(args.workers, initializer=_init_worker, initargs=(args.a, args.b, args.time)) as pool:
        for res in pool.imap_unordered(play_game, jobs):
            results.append(res)
            outcome = {1: "A WINS", 0: "DRAW", -1: "B WINS"}[res["result"]]
            note = " (illegal move by %s)" % res["illegal"] if res["illegal"] else ""
            print("%s first: %s in %i moves%s  [%s]" % ("A" if res["a_first"] else "B", outcome,
                                                         len(res["moves"]), note,
                                                         "".join(map(str, res["moves"]))), flush=True)

    print("\nA = %s, B = %s" % (args.a, args.b))
    for a_first in (True, False):
        subset = [r["result"] for r in results if r["a_first"] == a_first]
        print("  A going %s: %i W / %i D / %i L" % ("first " if a_first else "second", subset.count(1),
                                                 subset.count(0), subset.count(-1)))
    all_res = [r["result"] for r in results]
    print("  Total: %i W / %i D / %i L  (A win rate %.1f%%)" % (all_res.count(1), all_res.count(0),
                                                            all_res.count(-1),
                                                            100 * all_res.count(1) / len(all_res)))
    for role, name in (("a", args.a), ("b", args.b)):
        ts = [t for r in results for t in r["times"][role]]
        if ts:
            print("  %s move time: avg %.2fs, max %.2fs" % (name, float(np.mean(ts)), max(ts)))


if __name__ == "__main__":
    main()
