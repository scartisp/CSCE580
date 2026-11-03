/* Exact Connect Four solver used offline by build_opening_book.py. It is not part of the
 * submission; it only exists to compute the opening book that gets embedded in
 * connect_four_alg.py.
 *
 * Same bitboard layout as connect_four_alg.py: column c uses bits c*7 .. c*7+6 with bit c*7 at
 * the bottom and bit c*7+6 an always-empty sentinel. A position is (pos, mask): pos holds the
 * stones of the player to move, mask holds all stones.
 *
 * Scores: if the player to move wins with their k-th stone, the score is 22 - k (positive); a
 * loss is the negative of the opponent's score; a draw is 0. So bigger magnitude = faster.
 *
 * Build: gcc -O3 -march=native -shared -static-libgcc -o book_solver.dll book_solver.c
 */
#include <stdint.h>
#include <string.h>

#define WIDTH 7
#define HEIGHT 6
#define H1 (HEIGHT + 1)
#define SIZE (WIDTH * HEIGHT)
#define MIN_SCORE (-(SIZE / 2) + 3)
#define MAX_SCORE ((SIZE + 1) / 2 - 3)

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

static const uint64_t BOTTOM = 0x40810204081ULL; /* bit c*7 for every column */
static const uint64_t BOARD = 0x40810204081ULL * 0x3FULL;
static const int MOVE_ORDER[WIDTH] = {3, 2, 4, 1, 5, 0, 6};

/* Transposition table. The key (pos + mask) fits in 49 bits; storing its low 32 bits at index
 * key % TT_SIZE identifies it uniquely because TT_SIZE is an odd prime and 2^32 * TT_SIZE > 2^49.
 * Values: 0 = empty, 1..37 = upper bound, 38..74 = lower bound (see negamax). */
#define TT_SIZE 8388593
static uint32_t tt_keys[TT_SIZE];
static uint8_t tt_vals[TT_SIZE];
static uint64_t nodes;

static inline void tt_put(uint64_t key, int val) {
    if (val <= 0 || val > 255) return;
    uint32_t idx = (uint32_t)(key % TT_SIZE);
    tt_keys[idx] = (uint32_t)key;
    tt_vals[idx] = (uint8_t)val;
}

static inline int tt_get(uint64_t key) {
    uint32_t idx = (uint32_t)(key % TT_SIZE);
    return tt_keys[idx] == (uint32_t)key ? tt_vals[idx] : 0;
}

static inline uint64_t col_mask(int col) { return 0x3FULL << (col * H1); }

static inline uint64_t possible(uint64_t mask) { return (mask + BOTTOM) & BOARD; }

/* Empty cells that would complete four in a row for the owner of pos */
static inline uint64_t winning_spots(uint64_t pos, uint64_t mask) {
    uint64_t r = (pos << 1) & (pos << 2) & (pos << 3);
    static const int shifts[3] = {H1, HEIGHT, H1 + 1};
    for (int i = 0; i < 3; i++) {
        int s = shifts[i];
        uint64_t p = (pos << s) & (pos << (2 * s));
        r |= p & (pos << (3 * s));
        r |= p & (pos >> s);
        p = (pos >> s) & (pos >> (2 * s));
        r |= p & (pos << s);
        r |= p & (pos >> (3 * s));
    }
    return r & (BOARD ^ mask);
}

/* Requires that the player to move cannot win immediately. moves = number of stones played. */
static int negamax(uint64_t pos, uint64_t mask, int moves, int alpha, int beta) {
    nodes++;
    uint64_t poss = possible(mask);
    uint64_t opp_win = winning_spots(pos ^ mask, mask);
    uint64_t forced = poss & opp_win;
    if (forced) {
        if (forced & (forced - 1)) return -(SIZE - moves) / 2; /* two threats, cannot block both */
        poss = forced;
    }
    uint64_t next = poss & ~(opp_win >> 1); /* never play directly under an opponent threat */
    if (!next) return -(SIZE - moves) / 2;
    if (moves >= SIZE - 2) return 0;

    int lo = -(SIZE - 2 - moves) / 2; /* the opponent cannot win on their next move */
    if (alpha < lo) {
        alpha = lo;
        if (alpha >= beta) return alpha;
    }
    int hi = (SIZE - 1 - moves) / 2; /* we cannot win on this move */
    uint64_t key = pos + mask;
    int val = tt_get(key);
    if (val) {
        if (val > MAX_SCORE - MIN_SCORE + 1) {
            lo = val + 2 * MIN_SCORE - MAX_SCORE - 2;
            if (alpha < lo) {
                alpha = lo;
                if (alpha >= beta) return alpha;
            }
        } else {
            hi = val + MIN_SCORE - 1;
        }
    }
    if (beta > hi) {
        beta = hi;
        if (alpha >= beta) return beta;
    }

    /* order moves by how many threats they create, ties broken center-first */
    uint64_t mv[WIDTH];
    int sc[WIDTH];
    int n = 0;
    for (int i = 0; i < WIDTH; i++) {
        uint64_t m = next & col_mask(MOVE_ORDER[i]);
        if (!m) continue;
        int s = __builtin_popcountll(winning_spots(pos | m, mask));
        int j = n++;
        while (j > 0 && sc[j - 1] < s) {
            mv[j] = mv[j - 1];
            sc[j] = sc[j - 1];
            j--;
        }
        mv[j] = m;
        sc[j] = s;
    }

    for (int k = 0; k < n; k++) {
        int score = -negamax(pos ^ mask, mask | mv[k], moves + 1, -beta, -alpha);
        if (score >= beta) {
            tt_put(key, score + MAX_SCORE - 2 * MIN_SCORE + 2);
            return score;
        }
        if (score > alpha) alpha = score;
    }
    tt_put(key, alpha - MIN_SCORE + 1);
    return alpha;
}

/* Exact score of the position for the player to move (weak != 0: only its sign is reliable) */
EXPORT int solve(uint64_t pos, uint64_t mask, int weak) {
    int moves = __builtin_popcountll(mask);
    if (winning_spots(pos, mask) & possible(mask)) return (SIZE + 1 - moves) / 2;

    int lo = -(SIZE - moves) / 2;
    int hi = (SIZE + 1 - moves) / 2;
    if (weak) {
        lo = -1;
        hi = 1;
    }
    /* narrow the score with null-window searches, probing near 0 first */
    while (lo < hi) {
        int med = lo + (hi - lo) / 2;
        if (med <= 0 && lo / 2 < med) med = lo / 2;
        else if (med >= 0 && hi / 2 > med) med = hi / 2;
        int r = negamax(pos, mask, moves, med, med + 1);
        if (r <= med) hi = r;
        else lo = r;
    }
    return lo;
}

EXPORT void reset(void) {
    memset(tt_keys, 0, sizeof(tt_keys));
    memset(tt_vals, 0, sizeof(tt_vals));
}

EXPORT uint64_t node_count(void) { return nodes; }
