#pragma once
#include "search_types.hpp"
#include "game_history.hpp"

#include "minimax.hpp"

#include <cstdint>
#include <cstring>

/* ================================================================
 * Transposition Table
 * ================================================================ */
enum TTFlag : uint8_t { TT_NONE = 0, TT_EXACT = 1, TT_ALPHA = 2, TT_BETA = 3 };

// Compact move: pack (fr, fc, tr, tc) into 16 bits (4 bits each)
struct CompactMove {
    uint16_t data = 0;

    CompactMove() = default;
    CompactMove(const Move& m){
        data = (uint16_t)(
            ((m.first.first & 0xF) << 12) |
            ((m.first.second & 0xF) << 8) |
            ((m.second.first & 0xF) << 4) |
            (m.second.second & 0xF)
        );
    }
    Move to_move() const {
        return Move(
            Point((data >> 12) & 0xF, (data >> 8) & 0xF),
            Point((data >> 4) & 0xF, data & 0xF)
        );
    }
    bool is_valid() const { return data != 0; }
};

struct TTEntry {
    uint32_t key32;      // Upper 32 bits of Zobrist hash for verification
    int16_t  score;      // search score
    int8_t   depth;      // search depth
    TTFlag   flag;       // bound type
    CompactMove best_move; // best move found (2 bytes)
    // Total: 4 + 2 + 1 + 1 + 2 = 10 bytes, padded to 12
};

// 2M entries × 12 bytes ≈ 24MB — well within memory budget
static constexpr int TT_SIZE = (1 << 21); // 2,097,152 entries
static constexpr int TT_MASK = TT_SIZE - 1;

/* ================================================================
 * Killer Moves & History Heuristic
 * ================================================================ */
static constexpr int MAX_PLY = 128;

class Submission{
public:
    // Transposition table (heap-allocated, persists across UBGI iterative deepening calls)
    static TTEntry* tt;
    static bool tt_initialized;

    // Killer moves: 2 slots per ply
    static Move killers[MAX_PLY][2];

    // History heuristic: indexed by [player][from_sq][to_sq]
    // from_sq = row * 5 + col, to_sq = row * 5 + col
    static int history_table[2][30][30];

    static void init_tt();
    static void clear_killers_and_history();

    static void store_tt(uint64_t key, int score, int depth, TTFlag flag, const Move& best_move, int ply);
    static bool probe_tt(uint64_t key, int depth, int alpha, int beta, int& score, Move& best_move, int ply);

    static void sort_moves(
        State* state,
        std::vector<Move>& moves,
        const Move& tt_move,
        int ply
    );

    static int quiescence(
        State *state,
        GameHistory& history,
        int ply,
        SearchContext& ctx,
        const MMParams& p,
        int alpha,
        int beta
    );

    static int eval_ctx(
        State *state,
        int depth,
        GameHistory& history,
        int ply,
        SearchContext& ctx,
        const MMParams& p,
        int alpha = M_MAX,
        int beta = P_MAX
    );
    static SearchResult search(
        State *state,
        int depth,
        GameHistory& history,
        SearchContext& ctx
    );

    static ParamMap default_params();
    static std::vector<ParamDef> param_defs();
};
