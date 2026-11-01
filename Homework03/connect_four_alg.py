import time
from typing import Dict, List, Tuple

from connect_four import ConnectFourState, ConnectFour

# The search runs on bitboards instead of ConnectFourState, which is far too slow to expand deeply.
# Column c uses bits c*7 .. c*7+6: bit c*7 is the bottom row and bit c*7+6 is a sentinel that is
# always empty, so shifting a board never carries a line from one column into the next.
# A position is (pos, mask): pos holds the stones of the player to move, mask holds all stones.
WIDTH: int = 7
HEIGHT: int = 6
H1: int = HEIGHT + 1

TIME_LIMIT: float = 3.0  # seconds of search per move (the assignment allows 5)

BOTTOM: int = sum(1 << (col * H1) for col in range(WIDTH))
BOARD: int = BOTTOM * ((1 << HEIGHT) - 1)
COL_MASKS: List[int] = [((1 << HEIGHT) - 1) << (col * H1) for col in range(WIDTH)]
CENTER: int = COL_MASKS[WIDTH // 2]
ODD_ROWS: int = BOTTOM * 0b010101  # rows 1, 3, 5 counting from the bottom
EVEN_ROWS: int = BOTTOM * 0b101010  # rows 2, 4, 6
MOVE_ORDER: List[int] = [3, 2, 4, 1, 5, 0, 6]
DIRECTIONS: Tuple[int, ...] = (1, H1, HEIGHT, H1 + 1)  # vertical, horizontal, both diagonals

WIN: int = 1_000_000
WIN_THRESHOLD: int = WIN - 1000  # scores beyond this are proven wins/losses, not heuristics

EXACT, LOWER, UPPER = 0, 1, 2
TT_MAX_SIZE: int = 1_000_000

# position key -> (depth, flag, value, best move bit). Kept between moves since keys are absolute
# positions; mate scores are stored relative to the node so they stay valid at any ply.
_tt: Dict[int, Tuple[int, int, int, int]] = {}


class _Timeout(Exception):
    pass


def _winning_spots(pos: int, mask: int) -> int:
    """ Empty cells that would complete four in a row for the player owning pos """
    spots = (pos << 1) & (pos << 2) & (pos << 3)
    for shift in (H1, HEIGHT, H1 + 1):
        pair = (pos << shift) & (pos << (2 * shift))
        spots |= pair & (pos << (3 * shift))
        spots |= pair & (pos >> shift)
        pair = (pos >> shift) & (pos >> (2 * shift))
        spots |= pair & (pos << shift)
        spots |= pair & (pos >> (3 * shift))
    return spots & (BOARD ^ mask)


def _possible(mask: int) -> int:
    """ The lowest empty cell of every column that is not full """
    return (mask + BOTTOM) & BOARD


def _window_score(mine: int, theirs: int) -> int:
    """ Rewards windows of four cells that hold two or three of my stones and none of theirs """
    open_cells = BOARD & ~theirs
    score = 0
    for shift in DIRECTIONS:
        free = open_cells & (open_cells >> shift) & (open_cells >> (2 * shift)) & (open_cells >> (3 * shift))
        if not free:
            continue
        # bit-sliced count of my stones in the window starting at each bit
        b0, b1, b2, b3 = mine, mine >> shift, mine >> (2 * shift), mine >> (3 * shift)
        s1, c1 = b0 ^ b1, b0 & b1
        s2, c2 = b2 ^ b3, b2 & b3
        low = s1 ^ s2
        mid = c1 ^ c2 ^ (s1 & s2)
        score += 2 * (free & mid & ~low).bit_count() + 5 * (free & mid & low).bit_count()
    return score


def _evaluate(pos: int, opp: int, mask: int, my_threats: int, opp_threats: int) -> int:
    """ Heuristic value of a non-terminal position for the player to move """
    score = 3 * ((pos & CENTER).bit_count() - (opp & CENTER).bit_count())
    score += _window_score(pos, opp) - _window_score(opp, pos)

    # Zugzwang rule of thumb: the first player profits from threats on odd rows, the second on even.
    if mask.bit_count() % 2 == 0:
        my_good, opp_good = ODD_ROWS, EVEN_ROWS
    else:
        my_good, opp_good = EVEN_ROWS, ODD_ROWS
    score += 12 * (my_threats & my_good).bit_count() + 5 * (my_threats & opp_good).bit_count()
    score -= 12 * (opp_threats & opp_good).bit_count() + 5 * (opp_threats & my_good).bit_count()
    return score


def _order_moves(pos: int, mask: int, moves: int, first: int) -> List[int]:
    """ Moves as single-bit boards: the hinted move first, then those creating the most threats """
    scored: List[Tuple[int, int]] = []
    for col in MOVE_ORDER:
        move = moves & COL_MASKS[col]
        if move:
            if move == first:
                priority = 1000
            else:
                priority = _winning_spots(pos | move, mask | move).bit_count()
            scored.append((priority, move))
    scored.sort(key=lambda pm: -pm[0])  # stable, so ties keep the center-first order
    return [move for _, move in scored]


class _Searcher:
    def __init__(self, deadline: float):
        self.deadline: float = deadline
        self.nodes: int = 0

    def negamax(self, pos: int, mask: int, depth: int, alpha: int, beta: int, ply: int) -> int:
        """ Value of the position for the player to move. ply is the number of moves made since
        the root; a win on move k (counted from the root) scores WIN - k. """
        self.nodes += 1
        if (self.nodes & 1023) == 0 and time.perf_counter() > self.deadline:
            raise _Timeout()

        if mask == BOARD:
            return 0

        possible = _possible(mask)
        my_threats = _winning_spots(pos, mask)
        if my_threats & possible:
            return WIN - (ply + 1)

        opp = pos ^ mask
        opp_threats = _winning_spots(opp, mask)
        forced = possible & opp_threats
        if forced:
            if forced & (forced - 1):  # two immediate threats: cannot block both
                return -(WIN - (ply + 2))
            possible = forced
        moves = possible & ~(opp_threats >> 1)  # never play directly beneath an opponent threat
        if not moves:
            return -(WIN - (ply + 2))

        # we cannot win on this move, so the best outcome is a win on our next one
        best_possible = WIN - (ply + 3)
        if beta > best_possible:
            beta = best_possible
            if alpha >= beta:
                return beta

        if depth <= 0:
            return _evaluate(pos, opp, mask, my_threats, opp_threats)

        key = pos + mask
        tt_move = 0
        entry = _tt.get(key)
        if entry is not None:
            e_depth, e_flag, e_value, tt_move = entry
            if e_depth >= depth:
                if e_value > WIN_THRESHOLD:
                    e_value -= ply
                elif e_value < -WIN_THRESHOLD:
                    e_value += ply
                if e_flag == EXACT:
                    return e_value
                if e_flag == LOWER and e_value >= beta:
                    return e_value
                if e_flag == UPPER and e_value <= alpha:
                    return e_value

        alpha_orig = alpha
        best_value = -WIN - 1
        best_move = 0
        for idx, move in enumerate(_order_moves(pos, mask, moves, tt_move)):
            if idx == 0:
                value = -self.negamax(opp, mask | move, depth - 1, -beta, -alpha, ply + 1)
            else:
                # principal variation search: prove the move is no better with a null window
                value = -self.negamax(opp, mask | move, depth - 1, -alpha - 1, -alpha, ply + 1)
                if alpha < value < beta:
                    value = -self.negamax(opp, mask | move, depth - 1, -beta, -value, ply + 1)
            if value > best_value:
                best_value, best_move = value, move
                if value > alpha:
                    alpha = value
                    if alpha >= beta:
                        break

        if best_value <= alpha_orig:
            flag = UPPER
        elif best_value >= beta:
            flag = LOWER
        else:
            flag = EXACT
        stored = best_value
        if stored > WIN_THRESHOLD:
            stored += ply
        elif stored < -WIN_THRESHOLD:
            stored -= ply
        _tt[key] = (depth, flag, stored, best_move)
        return best_value

    def search_root(self, pos: int, mask: int, depth: int, ordered: List[int],
                    partial: List[Tuple[int, int]]) -> Tuple[int, int]:
        """ Searches every legal root move. Completed (value, move) pairs are appended to partial
        so a search cut off by the clock can still use the moves it finished. """
        alpha, beta = -WIN - 1, WIN + 1
        best_value, best_move = -WIN - 1, ordered[0]
        opp = pos ^ mask
        for idx, move in enumerate(ordered):
            if idx == 0:
                value = -self.negamax(opp, mask | move, depth - 1, -beta, -alpha, 1)
            else:
                value = -self.negamax(opp, mask | move, depth - 1, -alpha - 1, -alpha, 1)
                if value > alpha:
                    value = -self.negamax(opp, mask | move, depth - 1, -beta, -value, 1)
            partial.append((value, move))
            if value > best_value:
                best_value, best_move = value, move
                alpha = max(alpha, value)
        return best_value, best_move


def _move_to_col(move: int) -> int:
    return (move.bit_length() - 1) // H1


def _choose_column(pos: int, mask: int, deadline: float) -> int:
    possible = _possible(mask)
    legal: List[int] = [possible & COL_MASKS[col] for col in MOVE_ORDER if possible & COL_MASKS[col]]

    if mask == 0:
        return WIDTH // 2  # the center is the only winning first move
    if len(legal) == 1:
        return _move_to_col(legal[0])
    win_now = _winning_spots(pos, mask) & possible
    if win_now:
        return _move_to_col(win_now & -win_now)

    if len(_tt) > TT_MAX_SIZE:
        _tt.clear()

    searcher = _Searcher(deadline)
    best_move = _order_moves(pos, mask, possible, 0)[0]
    num_empty = WIDTH * HEIGHT - mask.bit_count()
    start = time.perf_counter()
    for depth in range(1, num_empty + 1):
        ordered = _order_moves(pos, mask, possible, best_move)
        partial: List[Tuple[int, int]] = []
        try:
            value, best_move = searcher.search_root(pos, mask, depth, ordered, partial)
        except _Timeout:
            # moves finished at this depth were searched deeper than anything from the last
            # iteration; the previous best is searched first, so it is among them if any are
            if partial:
                best_move = max(partial, key=lambda vm: vm[0])[1]
            break

        if abs(value) > WIN_THRESHOLD:
            break  # proven result, deeper search cannot change it
        # each iteration takes a few times longer than the last; skip one that cannot finish
        elapsed = time.perf_counter() - start
        if elapsed > 0.45 * (deadline - start):
            break

    return _move_to_col(best_move)


def make_move(state: ConnectFourState, env: ConnectFour) -> int:
    # grid[col, row] with row 0 at the top. make_move is always called on Max's turn, so Max's
    # stones (1) belong to the player to move.
    grid = state.grid
    pos, mask = 0, 0
    for col in range(WIDTH):
        for row in range(HEIGHT):
            if grid[col, row] != 0:
                bit = 1 << (col * H1 + (HEIGHT - 1 - row))
                mask |= bit
                if grid[col, row] > 0:
                    pos |= bit

    col = _choose_column(pos, mask, time.perf_counter() + TIME_LIMIT)

    actions = env.get_actions(state)
    if col not in actions:  # should never happen, but an illegal move would forfeit the game
        col = actions[0]
    return col
