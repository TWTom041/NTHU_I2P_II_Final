#include <utility>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <unordered_map>
#include <chrono>
#include "state.hpp"
#include "submission.hpp"

/* === NNUE evaluation ===
 * The embedded neural net (nnue_weights.hpp, trained by tools/nnue_train.cpp
 * on self-play data) is blended with the hand-crafted evaluation. It is ON by
 * default and self-contained -- the standard build command activates it with
 * no extra flags. Compile with -DNO_NNUE to disable, or -DNNUE_BLEND=<0..100>
 * to override the mix (percentage weight given to the NNUE score; the
 * remainder goes to the hand-crafted eval). Tuned default: 30. */
#ifndef NO_NNUE
#define USE_NNUE
#endif
#ifdef USE_NNUE
#include "nnue.hpp"
#ifndef NNUE_BLEND
#define NNUE_BLEND 30
#endif
#endif

/* ================================================================
 * Static member definitions
 * ================================================================ */
TTEntry* Submission::tt = nullptr;
bool Submission::tt_initialized = false;
uint8_t Submission::tt_generation = 0;
Move Submission::killers[MAX_PLY][2];
int Submission::history_table[2][30][30];
Move Submission::counter_moves[2][30][30];
std::chrono::high_resolution_clock::time_point Submission::search_start_time;
int64_t Submission::search_time_limit_ms = 0;

/* ================================================================
 * Search node pool: a ply-indexed array of State objects reused across the
 * (single-threaded, depth-first) search instead of heap-allocating a child
 * State at every node. Because search is DFS, pool[child_ply] is free to
 * overwrite once a child's subtree returns, and a parent never reads a child
 * slot while recursing into it -- so this also removes the per-node move-list
 * allocation (each slot's legal_actions buffer is reused). Falls back to the
 * heap path beyond the pool depth (never hit in practice).
 * ================================================================ */
static const int NODE_POOL_SIZE = 256;
static State g_node_pool[NODE_POOL_SIZE];

static inline State* child_state(State* parent, const Move& action, int child_ply, bool& heap){
    if(child_ply < NODE_POOL_SIZE){
        heap = false;
        parent->apply_into(g_node_pool[child_ply], action);
        return &g_node_pool[child_ply];
    }
    heap = true;
    return static_cast<State*>(parent->next_state(action));
}

static inline State* null_child(State* parent, int child_ply, bool& heap){
    if(child_ply < NODE_POOL_SIZE){
        heap = false;
        parent->null_into(g_node_pool[child_ply]);
        return &g_node_pool[child_ply];
    }
    heap = true;
    return static_cast<State*>(parent->create_null_state());
}

void Submission::init_tt(){
    if(!tt_initialized){
        tt = new TTEntry[TT_SIZE]();  // value-initialize (zero)
        tt_initialized = true;
    }
}

void Submission::clear_killers_and_history(){
    std::memset(killers, 0, sizeof(killers));
    std::memset(history_table, 0, sizeof(history_table));
    std::memset(counter_moves, 0, sizeof(counter_moves));
}

/* ================================================================
 * Transposition Table Operations
 * ================================================================ */
void Submission::store_tt(uint64_t key, int score, int depth, TTFlag flag, const Move& best_move, int ply){
    int idx = (int)(key & TT_MASK);
    TTEntry& entry = tt[idx];
    uint32_t key32 = (uint32_t)(key >> 32);

    // Adjust mate scores for storage
    int adj_score = score;
    if(score > P_MAX - 500) adj_score += ply;
    else if(score < M_MAX + 500) adj_score -= ply;

    // Replacement: always replace if different generation, or if same gen and depth >= stored
    bool replace = (entry.key32 == 0) ||
                   (entry.age != tt_generation) ||
                   (entry.key32 == key32) ||
                   (depth >= entry.depth) ||
                   (flag == TT_EXACT && entry.flag != TT_EXACT);
    
    if(replace){
        entry.key32 = key32;
        entry.score = (int16_t)std::max(-32000, std::min(32000, adj_score));
        entry.depth = (int8_t)depth;
        entry.flag = flag;
        entry.best_move = CompactMove(best_move);
        entry.age = tt_generation;
    }
}

bool Submission::probe_tt(uint64_t key, int depth, int alpha, int beta, int& score, Move& best_move, int ply){
    int idx = (int)(key & TT_MASK);
    TTEntry& entry = tt[idx];
    uint32_t key32 = (uint32_t)(key >> 32);

    if(entry.key32 != key32) return false;

    if(entry.best_move.is_valid()){
        best_move = entry.best_move.to_move();
    }

    if(entry.depth >= depth){
        score = (int)entry.score;
        // Adjust mate scores for retrieval
        if(score > P_MAX - 500) score -= ply;
        else if(score < M_MAX + 500) score += ply;

        if(entry.flag == TT_EXACT) return true;
        if(entry.flag == TT_ALPHA && score <= alpha){ score = alpha; return true; }
        if(entry.flag == TT_BETA && score >= beta){ score = beta; return true; }
    }
    return false;
}

Move Submission::probe_tt_move(uint64_t key){
    int idx = (int)(key & TT_MASK);
    TTEntry& entry = tt[idx];
    uint32_t key32 = (uint32_t)(key >> 32);
    if(entry.key32 == key32 && entry.best_move.is_valid()){
        return entry.best_move.to_move();
    }
    return Move();
}

/* ================================================================
 * Time Management
 * ================================================================ */
bool Submission::should_stop(SearchContext& ctx){
    if(ctx.stop) return true;
    
    // Check every 2048 nodes for speed
    if((ctx.nodes & 2047) == 0){
        auto now = std::chrono::high_resolution_clock::now();
        int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - search_start_time).count();
        
        if(search_time_limit_ms > 0){
            // Use the movetime limit with a small safety margin
            int64_t margin = std::min((int64_t)300, search_time_limit_ms / 25);
            if(ms >= search_time_limit_ms - margin){
                ctx.stop = true;
                return true;
            }
        } else {
            // Hard limit: 9.5s
            if(ms >= 9500){
                ctx.stop = true;
                return true;
            }
        }
    }
    return false;
}

/* ================================================================
 * Check Detection (lightweight)
 * ================================================================ */
bool Submission::is_in_check(State* state){
    // A position is "in check" if the opponent has a king-capture available.
    // Since this is a king-capture game model, we detect this by checking
    // if any of the side-to-move's pieces attack the opponent's king.
    // Actually, we need to check if the OPPONENT can capture OUR king,
    // i.e., if the previous move left us in check.
    // In this game model, the opponent's moves are not generated for us,
    // but we can check if our king is attacked.
    
    int self = state->player;
    int opp = 1 - self;
    
    // Find our king
    int kr = -1, kc = -1;
    for(int r = 0; r < BOARD_H; r++){
        for(int c = 0; c < BOARD_W; c++){
            if(state->board.board[self][r][c] == 6){
                kr = r; kc = c;
                break;
            }
        }
        if(kr >= 0) break;
    }
    if(kr < 0) return false; // No king? Shouldn't happen
    
    // Check if opponent's pieces attack our king square
    auto opp_board = state->board.board[opp];
    auto self_board = state->board.board[self];
    
    // Check opponent pawns
    // White pawns (opp==0) capture toward row-1 from their position;
    // so a white pawn at (kr+1, kc±1) attacks our king at (kr, kc)
    if(opp == 0){
        if(kr < BOARD_H - 1){
            if(kc > 0 && opp_board[kr+1][kc-1] == 1) return true;
            if(kc < BOARD_W-1 && opp_board[kr+1][kc+1] == 1) return true;
        }
    } else {
        // Black pawns (opp==1) capture toward row+1 from their position;
        // so a black pawn at (kr-1, kc±1) attacks our king at (kr, kc)
        if(kr > 0){
            if(kc > 0 && opp_board[kr-1][kc-1] == 1) return true;
            if(kc < BOARD_W-1 && opp_board[kr-1][kc+1] == 1) return true;
        }
    }
    
    // Check opponent knights
    static const int kn_dr[8] = {1, 1, -1, -1, 2, 2, -2, -2};
    static const int kn_dc[8] = {2, -2, 2, -2, 1, -1, 1, -1};
    for(int d = 0; d < 8; d++){
        int nr = kr + kn_dr[d], nc = kc + kn_dc[d];
        if(nr >= 0 && nr < BOARD_H && nc >= 0 && nc < BOARD_W){
            if(opp_board[nr][nc] == 3) return true;
        }
    }
    
    // Check opponent king (adjacent)
    static const int ki_dr[8] = {1, 0, -1, 0, 1, 1, -1, -1};
    static const int ki_dc[8] = {0, 1, 0, -1, 1, -1, 1, -1};
    for(int d = 0; d < 8; d++){
        int nr = kr + ki_dr[d], nc = kc + ki_dc[d];
        if(nr >= 0 && nr < BOARD_H && nc >= 0 && nc < BOARD_W){
            if(opp_board[nr][nc] == 6) return true;
        }
    }
    
    // Check sliding pieces (rook/queen on straight lines, bishop/queen on diagonals)
    // Rook directions
    static const int rook_dr[4] = {0, 0, 1, -1};
    static const int rook_dc[4] = {1, -1, 0, 0};
    for(int d = 0; d < 4; d++){
        int cr = kr + rook_dr[d], cc = kc + rook_dc[d];
        while(cr >= 0 && cr < BOARD_H && cc >= 0 && cc < BOARD_W){
            int op = opp_board[cr][cc];
            int sp = self_board[cr][cc];
            if(sp) break; // own piece blocks
            if(op){
                if(op == 2 || op == 5) return true; // rook or queen
                break; // other piece blocks
            }
            cr += rook_dr[d]; cc += rook_dc[d];
        }
    }
    
    // Bishop directions
    static const int bish_dr[4] = {1, 1, -1, -1};
    static const int bish_dc[4] = {1, -1, 1, -1};
    for(int d = 0; d < 4; d++){
        int cr = kr + bish_dr[d], cc = kc + bish_dc[d];
        while(cr >= 0 && cr < BOARD_H && cc >= 0 && cc < BOARD_W){
            int op = opp_board[cr][cc];
            int sp = self_board[cr][cc];
            if(sp) break;
            if(op){
                if(op == 4 || op == 5) return true; // bishop or queen
                break;
            }
            cr += bish_dr[d]; cc += bish_dc[d];
        }
    }
    
    return false;
}

/* ================================================================
 * Custom State Value Function (Evaluation)
 * ================================================================ */

static const int TUNE_MATERIAL[7] = {0, 100, 500, 320, 330, 900, 10000};
static const int TUNE_PST[6][6][5] = {
    // Pawn
    {{ 0,  0,  0,  0,  0}, {50, 50, 50, 50, 50}, {10, 15, 25, 15, 10},
     { 5, 10, 15, 10,  5}, { 0,  5,  5,  5,  0}, { 0,  0,  0,  0,  0}},
    // Rook
    {{ 5,  5,  5,  5,  5}, {10, 10, 10, 10, 10}, { 0,  0,  5,  0,  0},
     { 0,  0,  5,  0,  0}, { 0,  0,  5,  0,  0}, { 0,  0,  0,  0,  0}},
    // Knight
    {{-20, -10,  0, -10, -20}, {-10,  5, 15,  5, -10}, { 0, 15, 20, 15,  0},
     { 0, 15, 20, 15,  0}, {-10,  5, 15,  5, -10}, {-20, -10,  0, -10, -20}},
    // Bishop
    {{-10,  0,  0,  0, -10}, { 0, 10, 15, 10,  0}, { 0, 15, 15, 15,  0},
     { 0, 15, 15, 15,  0}, { 0, 10, 15, 10,  0}, {-10,  0,  0,  0, -10}},
    // Queen
    {{-10,  0,  5,  0, -10}, { 0,  5, 10,  5,  0}, { 0, 10, 15, 10,  0},
     { 0, 10, 15, 10,  0}, { 0,  5, 10,  5,  0}, {-10,  0,  5,  0, -10}},
    // King
    {{-30, -30, -30, -30, -30}, {-20, -20, -20, -20, -20}, {-20, -20, -20, -20, -20},
     {-10, -10, -10, -10, -10}, {10, 10,  0, 10, 10}, {20, 20,  5, 20, 20}},
};

static const int TUNE_TROPISM_W[7] = {0, 2, 5, 8, 5, 10, 0};

static inline int custom_king_tropism(int piece_type, int pr, int pc, int ekr, int ekc) {
    int dist = std::max(std::abs(pr - ekr), std::abs(pc - ekc));
    return TUNE_TROPISM_W[piece_type] * std::max(0, 4 - dist);
}

// Pawn structure evaluation (doubled + isolated penalties)
static inline int eval_pawn_structure(const char self_board[BOARD_H][BOARD_W]) {
    int penalty = 0;
    for(int c = 0; c < BOARD_W; c++){
        int pawn_count = 0;
        bool has_neighbor = false;
        for(int r = 0; r < BOARD_H; r++){
            if(self_board[r][c] == 1) pawn_count++;
        }
        if(pawn_count > 1) penalty -= 15 * (pawn_count - 1);
        if(pawn_count > 0){
            for(int r = 0; r < BOARD_H; r++){
                if(c > 0 && self_board[r][c-1] == 1) has_neighbor = true;
                if(c < BOARD_W-1 && self_board[r][c+1] == 1) has_neighbor = true;
            }
            if(!has_neighbor) penalty -= 10;
        }
    }
    return penalty;
}

// Passed pawn bonus: a pawn with no enemy pawn blocking it or on adjacent files ahead
// player==0 is white (advances toward row 0), player==1 is black (advances toward row 5)
static inline int eval_passed_pawns(
    const char self_board[BOARD_H][BOARD_W],
    const char opp_board[BOARD_H][BOARD_W],
    int player
){
    int bonus = 0;
    for(int c = 0; c < BOARD_W; c++){
        for(int r = 0; r < BOARD_H; r++){
            if(self_board[r][c] != 1) continue;
            bool passed = true;
            if(player == 0){
                // White pawn at row r advances toward row 0; check rows 0..r-1
                for(int ar = 0; ar < r && passed; ar++){
                    for(int ac = std::max(0, c-1); ac <= std::min(BOARD_W-1, c+1); ac++){
                        if(opp_board[ar][ac] == 1) passed = false;
                    }
                }
                if(passed){
                    // Bonus scales with proximity to promotion (row 0)
                    int dist = r; // distance to promotion row
                    bonus += std::max(10, 50 - dist * 10);
                }
            } else {
                // Black pawn at row r advances toward row BOARD_H-1; check rows r+1..BOARD_H-1
                for(int ar = r + 1; ar < BOARD_H && passed; ar++){
                    for(int ac = std::max(0, c-1); ac <= std::min(BOARD_W-1, c+1); ac++){
                        if(opp_board[ar][ac] == 1) passed = false;
                    }
                }
                if(passed){
                    int dist = BOARD_H - 1 - r;
                    bonus += std::max(10, 50 - dist * 10);
                }
            }
        }
    }
    return bonus;
}

static int custom_evaluate(State* state) {
    if (state->game_state == WIN) {
        return P_MAX;
    } else if (state->game_state == DRAW) {
        return 0;
    }

    auto self_board = state->board.board[state->player];
    auto oppn_board = state->board.board[1 - state->player];
    int self_score = 0, oppn_score = 0;

    int self_kr = -1, self_kc = -1;
    int oppn_kr = -1, oppn_kc = -1;

    for (int r = 0; r < BOARD_H; r++) {
        for (int c = 0; c < BOARD_W; c++) {
            if (self_board[r][c] == 6) { self_kr = r; self_kc = c; }
            if (oppn_board[r][c] == 6) { oppn_kr = r; oppn_kc = c; }
        }
    }

    for (int r = 0; r < BOARD_H; r++) {
        for (int c = 0; c < BOARD_W; c++) {
            int p_self = self_board[r][c];
            if (p_self) {
                int pst_r = (state->player == 0) ? r : BOARD_H - 1 - r;
                self_score += TUNE_MATERIAL[p_self] + TUNE_PST[p_self - 1][pst_r][c];
                if (oppn_kr != -1 && p_self != 6) {
                    self_score += custom_king_tropism(p_self, r, c, oppn_kr, oppn_kc);
                }
            }
            int p_oppn = oppn_board[r][c];
            if (p_oppn) {
                int pst_r = (1 - state->player == 0) ? r : BOARD_H - 1 - r;
                oppn_score += TUNE_MATERIAL[p_oppn] + TUNE_PST[p_oppn - 1][pst_r][c];
                if (self_kr != -1 && p_oppn != 6) {
                    oppn_score += custom_king_tropism(p_oppn, r, c, self_kr, self_kc);
                }
            }
        }
    }

    // Pawn structure (doubled/isolated penalties)
    self_score += eval_pawn_structure(self_board);
    oppn_score += eval_pawn_structure(oppn_board);

    // Passed pawn bonus
    self_score += eval_passed_pawns(self_board, oppn_board, state->player);
    oppn_score += eval_passed_pawns(oppn_board, self_board, 1 - state->player);

    // Mobility bonus - use already-computed legal actions for self
    int self_mobility = (int)state->legal_actions.size();
    int bonus = self_mobility * 5;

    int hand_score = self_score - oppn_score + bonus;

#ifdef USE_NNUE
    int nn_score = NNUE::evaluate(state);
    return (nn_score * NNUE_BLEND + hand_score * (100 - NNUE_BLEND)) / 100;
#else
    return hand_score;
#endif
}

/* ================================================================
 * Move Ordering
 * ================================================================ */
static inline int move_to_sq(const Point& p){
    return (int)p.first * BOARD_W + (int)p.second;
}

void Submission::sort_moves(
    State* state,
    std::vector<Move>& moves,
    const Move& tt_move,
    int ply,
    const Move& prev_move
){
    int opp = 1 - state->player;
    int self = state->player;

    struct ScoredMove {
        Move move;
        int score;
    };

    int n = (int)moves.size();
    if(n <= 1) return;

    // Get counter move for this position
    Move counter = Move();
    if(prev_move.first != prev_move.second || prev_move.first.first != 0 || prev_move.first.second != 0){
        int prev_from = move_to_sq(prev_move.first);
        int prev_to = move_to_sq(prev_move.second);
        if(prev_from >= 0 && prev_from < 30 && prev_to >= 0 && prev_to < 30){
            counter = counter_moves[opp][prev_from][prev_to];
        }
    }

    ScoredMove scored[128];
    if(n > 128) n = 128;

    for(int i = 0; i < n; i++){
        auto& m = moves[i];
        int s = 0;

        if(m == tt_move && tt_move.first != tt_move.second){
            s = 10000000; // TT move is highest priority
        } else {
            int to_r = (int)m.second.first;
            int to_c = (int)m.second.second;
            int victim = state->board.board[opp][to_r][to_c];

            if(victim){
                int from_r = (int)m.first.first;
                int from_c = (int)m.first.second;
                int attacker = state->board.board[self][from_r][from_c];
                // MVV-LVA: prioritize high-value victims with low-value attackers
                s = 5000000 + TUNE_MATERIAL[victim] * 16 - TUNE_MATERIAL[attacker];
            } else {
                // Check for pawn promotion
                int piece = state->board.board[self][(int)m.first.first][(int)m.first.second];
                if(piece == 1 && (to_r == 0 || to_r == BOARD_H - 1)){
                    s = 4500000; // Promotion is very good
                } else if(ply < MAX_PLY){
                    if(m == killers[ply][0]){
                        s = 4000000;
                    } else if(m == killers[ply][1]){
                        s = 3500000;
                    } else if(m == counter){
                        s = 3000000; // Counter move heuristic
                    }
                }
                int from_sq = move_to_sq(m.first);
                int to_sq = move_to_sq(m.second);
                if(from_sq >= 0 && from_sq < 30 && to_sq >= 0 && to_sq < 30){
                    s += history_table[self][from_sq][to_sq];
                }
            }
        }

        scored[i] = {m, s};
    }

    std::sort(scored, scored + n, [](const ScoredMove& a, const ScoredMove& b){
        return a.score > b.score;
    });

    for(int i = 0; i < n; i++){
        moves[i] = scored[i].move;
    }
}

/* ================================================================
 * Quiescence Search (with SEE-like pruning)
 * ================================================================ */
int Submission::quiescence(
    State *state,
    GameHistory& hist,
    int ply,
    SearchContext& ctx,
    const MMParams& p,
    int alpha,
    int beta
){
    (void)p;
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }

    if(should_stop(ctx)){
        return 0;
    }

    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    if(state->game_state == WIN){
        return P_MAX - ply;
    }
    if(state->game_state == DRAW){
        return 0;
    }

    int rep_score;
    if(state->check_repetition(hist, rep_score)){
        return rep_score;
    }

    int stand_pat = custom_evaluate(state);
    if(stand_pat >= beta){
        return beta;
    }

    // Delta pruning with bigger margin
    const int DELTA_MARGIN = TUNE_MATERIAL[5] + 200; // queen value + margin
    if(stand_pat + DELTA_MARGIN < alpha){
        return alpha;
    }

    if(alpha < stand_pat){
        alpha = stand_pat;
    }

    hist.push(state->hash());

    int opp = 1 - state->player;

    struct CaptureMove {
        Move move;
        int mvv_score;
    };
    CaptureMove captures[64];
    int num_captures = 0;

    for(auto& action : state->legal_actions){
        int to_r = action.second.first;
        int to_c = action.second.second;
        bool is_capture = state->board.board[opp][to_r][to_c] != 0;
        int piece = state->board.board[state->player][action.first.first][action.first.second];
        bool is_promo = (piece == 1 && (to_r == 0 || to_r == BOARD_H - 1));

        if(is_capture || is_promo){
            int mvv_lva = 0;
            if(is_capture){
                int victim = state->board.board[opp][to_r][to_c];
                int attacker = piece;
                mvv_lva = TUNE_MATERIAL[victim] * 16 - TUNE_MATERIAL[attacker];
                
                // SEE-like pruning: skip captures where we lose material
                // (capturing with high-value piece against low-value piece)
                if(stand_pat + TUNE_MATERIAL[victim] + 100 < alpha && !is_promo){
                    continue; // Delta pruning for individual captures
                }
            }
            if(is_promo) mvv_lva += 90000;

            if(num_captures < 64){
                captures[num_captures++] = {action, mvv_lva};
            }
        }
    }

    std::sort(captures, captures + num_captures, [](const CaptureMove& a, const CaptureMove& b){
        return a.mvv_score > b.mvv_score;
    });

    for(int i = 0; i < num_captures; i++){
        auto& action = captures[i].move;

        bool heap;
        State* next = child_state(state, action, ply + 1, heap);
        bool same = next->same_player_as_parent();

        int score = quiescence(next, hist, ply + 1, ctx, p, same ? alpha : -beta, same ? beta : -alpha);
        if(!same) score = -score;

        if(heap) delete next;

        if(ctx.stop){
            hist.pop(state->hash());
            return 0;
        }

        if(score >= beta){
            hist.pop(state->hash());
            return beta;
        }
        if(score > alpha){
            alpha = score;
        }
    }

    hist.pop(state->hash());
    return alpha;
}

/* ================================================================
 * Principal Variation Search with TT, Null-Move, LMR, Futility
 * ================================================================ */
int Submission::eval_ctx(
    State *state,
    int depth,
    GameHistory& hist,
    int ply,
    SearchContext& ctx,
    const MMParams& p,
    int alpha,
    int beta,
    bool null_move_allowed,
    const Move& prev_move
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }

    if(should_stop(ctx)){
        return 0;
    }

    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    if(state->game_state == WIN){
        return P_MAX - ply;
    }
    if(state->game_state == DRAW){
        return 0;
    }

    int rep_score;
    if(state->check_repetition(hist, rep_score)){
        return rep_score;
    }

    bool is_pv = (beta - alpha > 1);
    int orig_alpha = alpha;

    // Mate distance pruning
    if(!is_pv){
        int mating_score = P_MAX - ply;
        if(mating_score < beta){
            beta = mating_score;
            if(alpha >= mating_score) return mating_score;
        }
        int mated_score = M_MAX + ply;
        if(mated_score > alpha){
            alpha = mated_score;
            if(beta <= mated_score) return mated_score;
        }
    }

    uint64_t hash_key = state->hash();
    Move tt_move = {};
    int tt_score;
    if(probe_tt(hash_key, depth, alpha, beta, tt_score, tt_move, ply)){
        if(!is_pv){
            return tt_score;
        }
    }

    // Internal Iterative Deepening: BEFORE hist.push to avoid double-push
    // (doing IID after push causes the same hash to appear twice, triggering
    // false draws in any 2-move loop that returns to this position during IID)
    if(is_pv && tt_move.first == tt_move.second && tt_move.first.first == 0 && depth >= 4){
        eval_ctx(state, depth - 2, hist, ply, ctx, p, alpha, beta, false, prev_move);
        if(!ctx.stop){
            tt_move = probe_tt_move(hash_key);
        }
    }

    hist.push(hash_key);

    if(depth <= 0){
        int score = quiescence(state, hist, ply, ctx, p, alpha, beta);
        hist.pop(hash_key);
        return score;
    }

    // Check detection for extensions
    bool in_check = is_in_check(state);
    int extension = in_check ? 1 : 0;

    int static_eval = custom_evaluate(state);

    // Reverse Futility Pruning (static null move pruning)
    if(!is_pv && !in_check && depth <= 4 && ply > 0){
        int margin = 120 * depth;
        if(static_eval - margin >= beta){
            hist.pop(hash_key);
            return static_eval - margin;
        }
    }

    // Null Move Pruning
    if(!is_pv && null_move_allowed && depth >= 3 && ply > 0 && !in_check){
        if(static_eval >= beta){
            bool null_heap;
            State* ns = null_child(state, ply + 1, null_heap);
            if(ns){
                int R = std::min(depth - 1, 3 + (depth >= 7 ? 1 : 0));
                if(R < 2) R = 2;
                int null_score = -eval_ctx(ns, depth - 1 - R, hist, ply + 1, ctx, p, -beta, -beta + 1, false);
                if(null_heap) delete ns;

                if(null_score >= beta){
                    hist.pop(hash_key);
                    // Don't return unproven mate scores from null move
                    if(null_score >= P_MAX - 500) return beta;
                    return null_score;
                }
            }
        }
    }

    sort_moves(state, state->legal_actions, tt_move, ply, prev_move);

    int best_score = M_MAX;
    Move best_move = {};
    int move_index = 0;

    // Futility pruning threshold
    bool can_futility = (!is_pv && !in_check && depth <= 3 && ply > 0);
    int futility_margin = static_eval + 150 * depth;

    for(auto& action : state->legal_actions){
        int to_r = (int)action.second.first;
        int to_c = (int)action.second.second;
        int opp = 1 - state->player;
        bool is_capture = state->board.board[opp][to_r][to_c] != 0;
        int piece = state->board.board[state->player][(int)action.first.first][(int)action.first.second];
        bool is_promo = (piece == 1 && (to_r == 0 || to_r == BOARD_H - 1));
        bool is_killer = (ply < MAX_PLY && (action == killers[ply][0] || action == killers[ply][1]));

        // Futility pruning: skip quiet moves that can't raise alpha
        if(can_futility && move_index > 0 && !is_capture && !is_promo && !is_killer){
            if(futility_margin <= alpha){
                move_index++;
                continue;
            }
        }

        // Late move pruning: skip very late quiet moves at low depth
        if(!is_pv && !in_check && depth <= 2 && move_index >= 8 + depth * 4 && !is_capture && !is_promo){
            move_index++;
            continue;
        }

        bool heap;
        State* next = child_state(state, action, ply + 1, heap);
        bool same = next->same_player_as_parent();
        int score;

        int new_depth = depth - 1 + extension;

        if(move_index == 0){
            score = eval_ctx(next, new_depth, hist, ply + 1, ctx, p, same ? alpha : -beta, same ? beta : -alpha, true, action);
            if(!same) score = -score;
        }else{
            int reduction = 0;
            // LMR: Late Move Reductions
            if(depth >= 3 && move_index >= 2 && !is_capture && !is_promo && !in_check && !is_killer){
                // More aggressive reduction formula
                reduction = 1;
                if(move_index >= 4) reduction = 2;
                if(move_index >= 8) reduction = 3;
                if(depth >= 6 && move_index >= 12) reduction = 4;
                
                // Reduce less for PV nodes
                if(is_pv && reduction > 1) reduction--;
                
                // Don't reduce into qsearch
                if(new_depth - reduction < 1) reduction = std::max(0, new_depth - 1);
            }

            // Null window search with reduction
            score = eval_ctx(next, new_depth - reduction, hist, ply + 1, ctx, p, same ? alpha : -alpha - 1, same ? alpha + 1 : -alpha, true, action);
            if(!same) score = -score;

            // Re-search without reduction if reduced search fails high
            if(reduction > 0 && score > alpha && !ctx.stop){
                score = eval_ctx(next, new_depth, hist, ply + 1, ctx, p, same ? alpha : -alpha - 1, same ? alpha + 1 : -alpha, true, action);
                if(!same) score = -score;
            }

            // Re-search with full window if null window fails high
            if(score > alpha && score < beta && !ctx.stop){
                score = eval_ctx(next, new_depth, hist, ply + 1, ctx, p, same ? alpha : -beta, same ? beta : -alpha, true, action);
                if(!same) score = -score;
            }
        }

        if(heap) delete next;

        if(ctx.stop){
            hist.pop(hash_key);
            return best_score > M_MAX ? best_score : 0;
        }

        if(score > best_score){
            best_score = score;
            best_move = action;
        }
        if(score > alpha){
            alpha = score;
        }
        if(alpha >= beta){
            // Update killers and history for quiet moves that cause cutoff
            if(!is_capture){
                if(ply < MAX_PLY){
                    if(killers[ply][0] != action){
                        killers[ply][1] = killers[ply][0];
                        killers[ply][0] = action;
                    }
                }
                int from_sq = move_to_sq(action.first);
                int to_sq = move_to_sq(action.second);
                if(from_sq >= 0 && from_sq < 30 && to_sq >= 0 && to_sq < 30){
                    history_table[state->player][from_sq][to_sq] += depth * depth;
                    // Gravity: prevent overflow
                    if(history_table[state->player][from_sq][to_sq] > 100000){
                        for(int p2 = 0; p2 < 2; p2++){
                            for(int f = 0; f < 30; f++){
                                for(int t = 0; t < 30; t++){
                                    history_table[p2][f][t] /= 2;
                                }
                            }
                        }
                    }
                }
                // Counter move heuristic
                if(prev_move.first != prev_move.second || prev_move.first.first != 0 || prev_move.first.second != 0){
                    int prev_from = move_to_sq(prev_move.first);
                    int prev_to = move_to_sq(prev_move.second);
                    if(prev_from >= 0 && prev_from < 30 && prev_to >= 0 && prev_to < 30){
                        counter_moves[1 - state->player][prev_from][prev_to] = action;
                    }
                }
            }
            break;
        }

        move_index++;
    }

    // Store in TT
    TTFlag flag;
    if(best_score <= orig_alpha){
        flag = TT_ALPHA;
    } else if(best_score >= beta){
        flag = TT_BETA;
    } else {
        flag = TT_EXACT;
    }
    store_tt(hash_key, best_score, depth, flag, best_move, ply);

    hist.pop(hash_key);
    return best_score;
}

/* ================================================================
 * Root Search with Aspiration Windows and PV tracking
 * ================================================================ */
SearchResult Submission::search(
    State *state,
    int depth,
    GameHistory& hist,
    SearchContext& ctx
){
    ctx.reset();
    ctx.stop = false; // Ensure stop flag is cleared for each depth iteration
    init_tt();
    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if (depth == 1) {
        search_start_time = std::chrono::high_resolution_clock::now();
        search_time_limit_ms = ctx.movetime_ms;
        tt_generation = (tt_generation + 1) & 0xFF;
        // Don't clear history between iterative deepening calls - 
        // it's valuable information. Only clear killers.
        std::memset(killers, 0, sizeof(killers));
    }

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }

    uint64_t root_hash = state->hash();
    Move tt_move = {};
    int tt_score;
    bool tt_hit = probe_tt(root_hash, 0, M_MAX, P_MAX, tt_score, tt_move, 0);

    sort_moves(state, state->legal_actions, tt_move, 0, Move());

    int alpha = M_MAX;
    int beta = P_MAX;
    int best_score = M_MAX - 10;
    Move best_move = state->legal_actions.empty() ? Move() : state->legal_actions[0];
    int total_moves = (int)state->legal_actions.size();
    
    // Aspiration Windows (delta=50 with pawn=100: ~half pawn initial window)
    int aspiration_delta = 50;
    if (depth >= 4 && tt_hit) {
        alpha = std::max(M_MAX, tt_score - aspiration_delta);
        beta = std::min(P_MAX, tt_score + aspiration_delta);
    }

    int retries = 0;
    while (true) {
        int orig_alpha = alpha;
        int orig_beta = beta;
        best_score = M_MAX - 10;
        int move_index = 0;
        Move iter_best_move = best_move;
        
        for(auto& action : state->legal_actions){
            bool heap;
            State* next = child_state(state, action, 1, heap);
            bool same = next->same_player_as_parent();
            int score;

            if(move_index == 0){
                score = eval_ctx(next, depth - 1, hist, 1, ctx, p, same ? alpha : -beta, same ? beta : -alpha, true, action);
                if(!same) score = -score;
            }else{
                // Null window search
                score = eval_ctx(next, depth - 1, hist, 1, ctx, p, same ? alpha : -alpha - 1, same ? alpha + 1 : -alpha, true, action);
                if(!same) score = -score;

                if(score > alpha && score < beta && !ctx.stop){
                    score = eval_ctx(next, depth - 1, hist, 1, ctx, p, same ? alpha : -beta, same ? beta : -alpha, true, action);
                    if(!same) score = -score;
                }
            }

            if(heap) delete next;

            if(ctx.stop && depth > 1){
                break;
            }

            if(score > best_score){
                best_score = score;
                iter_best_move = action;
                result.best_move = action;
                result.score = best_score;

                // Build PV with just the best root move
                result.pv.clear();
                result.pv.push_back(action);

                if(p.report_partial && ctx.on_root_update){
                    ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
                }
            }
            if(score > alpha){
                alpha = score;
            }

            move_index++;
        }

        if (ctx.stop && depth > 1) {
            // Use whatever best move we found so far
            if(iter_best_move.first != iter_best_move.second || iter_best_move.first.first != 0){
                best_move = iter_best_move;
                result.best_move = best_move;
            }
            break;
        }

        // Check if aspiration window failed
        if (best_score <= orig_alpha && orig_alpha > M_MAX) {
            // Fail low: widen window downward
            aspiration_delta *= 4;
            alpha = std::max(M_MAX, best_score - aspiration_delta);
            beta = orig_beta;
            retries++;
            if(retries > 4){
                alpha = M_MAX;
                beta = P_MAX;
            }
            continue;
        } else if (best_score >= orig_beta && orig_beta < P_MAX) {
            // Fail high: widen window upward
            aspiration_delta *= 4;
            alpha = orig_alpha;
            beta = std::min(P_MAX, best_score + aspiration_delta);
            retries++;
            if(retries > 4){
                alpha = M_MAX;
                beta = P_MAX;
            }
            continue;
        }
        best_move = iter_best_move;
        break;
    }

    if(!ctx.stop || depth == 1){
        store_tt(root_hash, best_score, depth, TT_EXACT, best_move, 0);
    }

    result.best_move = best_move;
    result.nodes = ctx.nodes;
    result.seldepth = ctx.seldepth;

    return result;
}

ParamMap Submission::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "false"},
    };
}

std::vector<ParamDef> Submission::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "false"},
    };
}
