"""Builds the opening book embedded in connect_four_alg.py, using our exact solver (book_solver.c).

For every position the agent can face in its first moves (any opponent replies, our moves taken
from the book itself), each legal move is classified as win / draw / loss with the solver. Among the
best class the move with the best exact score is kept (fastest win, slowest loss); remaining ties
are broken by our own heuristic search. Mirror images share one entry.

Optionally one more level per side is added that only keeps the positions the agent would handle
badly on its own: --hard-first-ply keeps those its exact solver would likely not finish within the
move time, --hard-second-ply those where its heuristic search picks a worse move than the best.

Results are cached in opening_book_cache.json / opening_book_hard_cache.json so the build can be
extended or resumed, and the OPENING_BOOK block of connect_four_alg.py is rewritten at the end.

Example (the current book):
    python build_opening_book.py --first-ply 14 --second-ply 11 --hard-first-ply 16 --hard-second-ply 13

"""
import json
import os
import re
import time
from argparse import ArgumentParser
from multiprocessing import Pool
from typing import Dict, List, Set, Tuple

import c4_solver as S

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "opening_book_cache.json")
HARD_CACHE_PATH = os.path.join(HERE, "opening_book_hard_cache.json")
ALG_PATH = os.path.join(HERE, "connect_four_alg.py")
TIEBREAK_DEPTH = 10


def canonical(pos: int, mask: int) -> Tuple[int, int, int]:
    """ (key, pos, mask) of the position or its mirror image, whichever has the smaller key """
    key = pos + mask
    mirror_key = S.mirror(key)
    if mirror_key < key:
        return mirror_key, S.mirror(pos), S.mirror(mask)
    return key, pos, mask


def choose_move(job: Tuple[int, int, bool]) -> Dict:
    pos, mask, strong = job
    start = time.perf_counter()
    weak_scores = S.move_scores(pos, mask, weak=True)
    best_class = max(weak_scores.values())
    candidates = [col for col in S.MOVE_ORDER if weak_scores.get(col) == best_class]

    scores: Dict[int, int] = {}
    if strong and len(candidates) > 1 and best_class != 0:
        exact = S.move_scores(pos, mask)
        best = max(exact[col] for col in candidates)
        candidates = [col for col in candidates if exact[col] == best]
        scores = exact

    if len(candidates) > 1:
        import connect_four_alg as alg
        alg._tt.clear()
        searcher = alg._Searcher(float("inf"))
        values = {}
        for col in candidates:
            opp, new_mask = S.play(pos, mask, col)
            values[col] = -searcher.negamax(opp, new_mask, TIEBREAK_DEPTH - 1, -alg.WIN - 1, alg.WIN + 1, 1)
        best_val = max(values.values())
        candidates = [col for col in candidates if values[col] == best_val]

    return {"key": pos + mask, "col": candidates[0], "class": best_class, "weak": weak_scores,
            "exact": scores, "secs": time.perf_counter() - start}


def measure_hard(job: Tuple[int, int, int]) -> Dict:
    """ Solver effort for the position (C nodes to classify every move, from an empty table). Slow
    positions also get a book move. """
    pos, mask, threshold = job
    S.reset()
    before = S.node_count()
    S.move_scores(pos, mask, weak=True)
    nodes = S.node_count() - before
    if nodes < threshold:
        return {"key": pos + mask, "hard": False, "nodes": nodes}
    res = choose_move((pos, mask, True))
    res.update(hard=True, nodes=nodes)
    return res


def check_heuristic(job: Tuple[int, int, float]) -> Dict:
    """ Whether the agent's heuristic search (given check_time seconds, less than it gets in a
    game) picks a move of the best class here. Positions where it does not get a book move. """
    pos, mask, check_time = job
    weak = S.move_scores(pos, mask, weak=True)
    best = max(weak.values())
    key = pos + mask
    if len(set(weak.values())) == 1 or best > 1:  # every move equal, or an immediate win
        return {"key": key, "hard": False}
    import connect_four_alg as alg
    alg._tt.clear()
    col = alg._choose_column(pos, mask, time.perf_counter() + check_time)
    if weak[col] == best:
        return {"key": key, "hard": False}
    res = choose_move((pos, mask, True))
    res.update(hard=True, heuristic_col=col)
    return res


def our_turn_positions(first: bool, max_ply: int, book: Dict[int, int]) -> List[List[Tuple[int, int]]]:
    """ Canonical positions where we move, by level, following book moves for our side.
    Levels whose positions are not all in the book yet stop the expansion. """
    if first:
        # the agent always opens in the center, so start after (center, any reply)
        start_pos, start_mask = S.play(0, 0, 3)
        start_ply = 1
    else:
        start_pos, start_mask, start_ply = 0, 0, 0
    frontier: Set[Tuple[int, int, int]] = set()
    for col in S.legal_cols(start_mask):
        pos, mask = S.play(start_pos, start_mask, col)
        frontier.add(canonical(pos, mask))

    levels: List[List[Tuple[int, int]]] = []
    ply = start_ply + 1
    while frontier and ply <= max_ply:
        level = sorted(frontier)
        levels.append([(pos, mask) for _, pos, mask in level])
        if any(key not in book for key, _, _ in level):
            break
        nxt: Set[Tuple[int, int, int]] = set()
        for key, pos, mask in level:
            col = book[key]
            move = S.possible(mask) & S.COL_MASKS[col]
            if S.is_win(pos | move):
                continue
            opp, mask2 = S.play(pos, mask, col)
            for reply in S.legal_cols(mask2):
                reply_move = S.possible(mask2) & S.COL_MASKS[reply]
                if S.is_win(opp | reply_move) or (mask2 | reply_move) == S.BOARD:
                    continue
                pos3, mask3 = S.play(opp, mask2, reply)
                if S.legal_cols(mask3):
                    nxt.add(canonical(pos3, mask3))
        frontier = nxt
        ply += 2
    return levels


def write_book(book: Dict[int, int]):
    lines = ["OPENING_BOOK: Dict[int, int] = {"]
    items = sorted(book.items())
    for idx in range(0, len(items), 6):
        lines.append("    " + " ".join("%d: %d," % kv for kv in items[idx:idx + 6]))
    lines.append("}")
    block = "\n".join(lines)

    with open(ALG_PATH, "r", encoding="utf-8") as f:
        src = f.read()
    pattern = re.compile(r"(# BEGIN OPENING BOOK[^\n]*\n).*?(\n# END OPENING BOOK)", re.S)
    assert pattern.search(src), "OPENING BOOK markers not found in connect_four_alg.py"
    src = pattern.sub(lambda m: m.group(1) + block + m.group(2), src)
    with open(ALG_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)


def main():
    parser = ArgumentParser()
    parser.add_argument('--first-ply', type=int, default=8, help="deepest ply (stones on board) as first player")
    parser.add_argument('--second-ply', type=int, default=7, help="deepest ply as second player")
    parser.add_argument('--strong-from', type=int, default=3,
                        help="use exact scores to rank winning/losing moves from this ply on")
    parser.add_argument('--hard-first-ply', type=int, default=0,
                        help="one more first-player level (first-ply + 2) that only keeps positions the "
                             "agent's exact solver would be too slow on")
    parser.add_argument('--hard-nodes', type=int, default=250000,
                        help="C-solver node count above which a position counts as too slow")
    parser.add_argument('--hard-second-ply', type=int, default=0,
                        help="one more second-player level (second-ply + 2) that only keeps positions where "
                             "the agent's heuristic search picks a worse move than the best")
    parser.add_argument('--check-time', type=float, default=1.0,
                        help="search time given to the agent when checking --hard-second-ply positions")
    parser.add_argument('--workers', type=int, default=15)
    parser.add_argument('--no-write', action='store_true', help="update the cache only")
    args = parser.parse_args()

    book: Dict[int, int] = {}
    info: Dict[str, Dict] = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            info = json.load(f)
        book = {int(k): v["col"] for k, v in info.items()}
        print("loaded %d cached positions" % len(book))
    hard_info: Dict[str, Dict] = {}
    if os.path.exists(HARD_CACHE_PATH):
        with open(HARD_CACHE_PATH) as f:
            hard_info = json.load(f)

    with Pool(args.workers) as pool:
        for first, max_ply in ((False, args.second_ply), (True, args.first_ply)):
            while True:
                levels = our_turn_positions(first, max_ply, book)
                todo = [(pos, mask) for level in levels for pos, mask in level if pos + mask not in book]
                if not todo:
                    break
                ply = todo[0][1].bit_count()
                print("%s player, ply %d: solving %d positions" % ("first" if first else "second", ply,
                                                                   len(todo)), flush=True)
                start = time.perf_counter()
                jobs = [(pos, mask, mask.bit_count() >= args.strong_from) for pos, mask in todo]
                for idx, res in enumerate(pool.imap_unordered(choose_move, jobs)):
                    book[res["key"]] = res["col"]
                    info[str(res["key"])] = res
                    if (idx + 1) % 50 == 0 or idx + 1 == len(jobs):
                        print("  %d/%d done, %.0fs elapsed" % (idx + 1, len(jobs), time.perf_counter() - start),
                              flush=True)
                        with open(CACHE_PATH, "w") as f:
                            json.dump(info, f)

        if args.hard_first_ply:
            assert args.hard_first_ply == args.first_ply + 2, "the hard level must follow the last full level"
            level = our_turn_positions(True, args.hard_first_ply, book)[-1]
            todo = [(pos, mask) for pos, mask in level if str(pos + mask) not in hard_info]
            assert all(mask.bit_count() == args.hard_first_ply for _, mask in level)
            print("first player, ply %d: measuring %d positions" % (args.hard_first_ply, len(todo)), flush=True)
            start = time.perf_counter()
            jobs = [(pos, mask, args.hard_nodes) for pos, mask in todo]
            for idx, res in enumerate(pool.imap_unordered(measure_hard, jobs, chunksize=4)):
                hard_info[str(res["key"])] = res
                if (idx + 1) % 1000 == 0 or idx + 1 == len(jobs):
                    num_hard = sum(r["hard"] for r in hard_info.values())
                    print("  %d/%d done, %d hard, %.0fs elapsed" % (idx + 1, len(jobs), num_hard,
                                                                    time.perf_counter() - start), flush=True)
                    with open(HARD_CACHE_PATH, "w") as f:
                        json.dump(hard_info, f)

        if args.hard_second_ply:
            assert args.hard_second_ply == args.second_ply + 2, "the hard level must follow the last full level"
            level = our_turn_positions(False, args.hard_second_ply, book)[-1]
            todo = [(pos, mask) for pos, mask in level if str(pos + mask) not in hard_info]
            assert all(mask.bit_count() == args.hard_second_ply for _, mask in level)
            print("second player, ply %d: checking %d positions" % (args.hard_second_ply, len(todo)), flush=True)
            start = time.perf_counter()
            jobs = [(pos, mask, args.check_time) for pos, mask in todo]
            for idx, res in enumerate(pool.imap_unordered(check_heuristic, jobs, chunksize=4)):
                hard_info[str(res["key"])] = res
                if (idx + 1) % 2000 == 0 or idx + 1 == len(jobs):
                    num_hard = sum(r["hard"] for r in hard_info.values() if "heuristic_col" in r)
                    print("  %d/%d done, %d need a book move, %.0fs elapsed"
                          % (idx + 1, len(jobs), num_hard, time.perf_counter() - start), flush=True)
                    with open(HARD_CACHE_PATH, "w") as f:
                        json.dump(hard_info, f)

        for key, res in hard_info.items():
            if res["hard"] and res.get("nodes", args.hard_nodes) >= args.hard_nodes:
                book[int(key)] = res["col"]

    classes = [v["class"] for v in info.values()]
    print("book: %d positions (%d winning, %d drawn, %d lost for us), plus %d targeted positions"
          % (len(info), classes.count(1), classes.count(0), classes.count(-1), len(book) - len(info)))
    if not args.no_write:
        write_book(book)
        print("wrote OPENING_BOOK into", ALG_PATH)


if __name__ == "__main__":
    main()
