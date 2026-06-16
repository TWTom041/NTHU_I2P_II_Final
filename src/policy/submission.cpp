#include <utility>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <unordered_map>
#include <chrono>
#include "state.hpp"
#include "submission.hpp"
#include <iostream>

static std::chrono::high_resolution_clock::time_point g_start_time;

/* ================================================================
 * Static member definitions
 * ================================================================ */
TTEntry* Submission::tt = nullptr;
bool Submission::tt_initialized = false;
Move Submission::killers[MAX_PLY][2];
int Submission::history_table[2][30][30];

void Submission::init_tt(){
    if(!tt_initialized){
        tt = new TTEntry[TT_SIZE]();  // value-initialize (zero)
        tt_initialized = true;
    }
}

void Submission::clear_killers_and_history(){
    std::memset(killers, 0, sizeof(killers));
    std::memset(history_table, 0, sizeof(history_table));
}

/* ================================================================
 * Transposition Table Operations
 * ================================================================ */
void Submission::store_tt(uint64_t key, int score, int depth, TTFlag flag, const Move& best_move, int ply){
    int idx = (int)(key & TT_MASK);
    TTEntry& entry = tt[idx];
    uint32_t key32 = (uint32_t)(key >> 32);

    int adj_score = score;
    if(score > P_MAX - 200) adj_score += ply;
    else if(score < M_MAX + 200) adj_score -= ply;

    if(entry.key32 == 0 || entry.key32 == key32 || depth >= entry.depth){
        entry.key32 = key32;
        entry.score = (int16_t)adj_score;
        entry.depth = (int8_t)depth;
        entry.flag = flag;
        entry.best_move = CompactMove(best_move);
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
        if(score > P_MAX - 200) score -= ply;
        else if(score < M_MAX + 200) score += ply;

        if(entry.flag == TT_EXACT) return true;
        if(entry.flag == TT_ALPHA && score <= alpha){ score = alpha; return true; }
        if(entry.flag == TT_BETA && score >= beta){ score = beta; return true; }
    }
    return false;
}

/* ================================================================
 * Custom State Value Function (Evaluation)
 * ================================================================ */

static const int TUNE_MATERIAL[7] = {0, 20, 60, 70, 80, 200, 1000};
static const int TUNE_PST[6][6][5] = {
    // Pawn
    {{ 0,  0,  0,  0,  0}, {20, 20, 20, 20, 20}, { 5,  8, 12,  8,  5},
     { 2,  4,  6,  4,  2}, { 0,  2,  2,  2,  0}, { 0,  0,  0,  0,  0}},
    // Rook
    {{ 2,  2,  2,  2,  2}, { 4,  4,  4,  4,  4}, { 0,  0,  2,  0,  0},
     { 0,  0,  2,  0,  0}, { 0,  0,  2,  0,  0}, { 0,  0,  0,  0,  0}},
    // Knight
    {{-4, -2,  0, -2, -4}, {-2,  2,  4,  2, -2}, { 0,  4,  6,  4,  0},
     { 0,  4,  6,  4,  0}, {-2,  2,  4,  2, -2}, {-4, -2,  0, -2, -4}},
    // Bishop
    {{-2,  0,  0,  0, -2}, { 0,  3,  4,  3,  0}, { 0,  4,  4,  4,  0},
     { 0,  4,  4,  4,  0}, { 0,  3,  4,  3,  0}, {-2,  0,  0,  0, -2}},
    // Queen
    {{-2,  0,  2,  0, -2}, { 0,  2,  4,  2,  0}, { 0,  4,  6,  4,  0},
     { 0,  4,  6,  4,  0}, { 0,  2,  4,  2,  0}, {-2,  0,  2,  0, -2}},
    // King
    {{-8, -8, -8, -8, -8}, {-4, -4, -4, -4, -4}, {-4, -4, -4, -4, -4},
     {-4, -4, -4, -4, -4}, { 4,  4,  0,  4,  4}, { 6,  6,  2,  6,  6}},
};

static const int TUNE_TROPISM_W[7] = {0, 0, 3, 3, 2, 5, 0};

static inline int custom_king_tropism(int piece_type, int pr, int pc, int ekr, int ekc) {
    int dist = std::max(std::abs(pr - ekr), std::abs(pc - ekc));
    if(dist <= 2){
        return TUNE_TROPISM_W[piece_type] * (3 - dist);
    }
    return 0;
}

static int custom_evaluate(State* state) {
    if (state->game_state == WIN) {
        return P_MAX;
    } else if (state->game_state == DRAW) {
        return -20; // Contempt factor
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
                if (oppn_kr != -1) {
                    self_score += custom_king_tropism(p_self, r, c, oppn_kr, oppn_kc);
                }
            }
            int p_oppn = oppn_board[r][c];
            if (p_oppn) {
                int pst_r = (1 - state->player == 0) ? r : BOARD_H - 1 - r;
                oppn_score += TUNE_MATERIAL[p_oppn] + TUNE_PST[p_oppn - 1][pst_r][c];
                if (self_kr != -1) {
                    oppn_score += custom_king_tropism(p_oppn, r, c, self_kr, self_kc);
                }
            }
        }
    }

    // Faster mobility heuristic based on legal actions count without full null state generation
    int self_mobility = state->legal_actions.size();
    int bonus = self_mobility * 2; // Approximate mobility advantage

    return self_score - oppn_score + bonus;
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
    int ply
){
    int opp = 1 - state->player;
    int self = state->player;

    struct ScoredMove {
        Move move;
        int score;
    };

    int n = (int)moves.size();
    if(n <= 1) return;

    ScoredMove scored[128];
    if(n > 128) n = 128;

    for(int i = 0; i < n; i++){
        auto& m = moves[i];
        int s = 0;

        if(m == tt_move && tt_move.first != tt_move.second){
            s = 1000000;
        } else {
            int to_r = (int)m.second.first;
            int to_c = (int)m.second.second;
            int victim = state->board.board[opp][to_r][to_c];

            if(victim){
                int from_r = (int)m.first.first;
                int from_c = (int)m.first.second;
                int attacker = state->board.board[self][from_r][from_c];
                s = 100000 + TUNE_MATERIAL[victim] * 10 - TUNE_MATERIAL[attacker];
            } else {
                if(ply < MAX_PLY){
                    if(m == killers[ply][0]){
                        s = 90000;
                    } else if(m == killers[ply][1]){
                        s = 80000;
                    }
                }
                int from_sq = move_to_sq(m.first);
                int to_sq = move_to_sq(m.second);
                if(from_sq >= 0 && from_sq < 30 && to_sq >= 0 && to_sq < 30){
                    s += history_table[self][from_sq][to_sq];
                }
            }

            int piece = state->board.board[self][(int)m.first.first][(int)m.first.second];
            if(piece == 1 && (to_r == 0 || to_r == BOARD_H - 1)){
                s += 95000;
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
 * Quiescence Search
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

    // Periodically check if we are approaching the 10-second hard limit (9.5s safety margin)
    if ((ctx.nodes & 4095) == 0) {
        auto now = std::chrono::high_resolution_clock::now();
        int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_start_time).count();
        if (ctx.movetime_ms > 0) {
            int64_t margin = std::min((int64_t)500, ctx.movetime_ms / 20);
            if (ms >= ctx.movetime_ms - margin) {
                ctx.stop = true;
            }
        } else if (ms >= 9500) {
            ctx.stop = true;
        }
    }

    if(ctx.stop){
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

    const int DELTA_MARGIN = 250;
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
                mvv_lva = TUNE_MATERIAL[victim] * 10 - TUNE_MATERIAL[piece];
            }
            if(is_promo) mvv_lva += 9000;

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

        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int score = quiescence(next, hist, ply + 1, ctx, p, same ? alpha : -beta, same ? beta : -alpha);
        if(!same) score = -score;

        delete next;

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
 * Principal Variation Search with TT, Null-Move, LMR
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
    bool null_move_made
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }

    // Periodically check if we are approaching the 10-second hard limit (9.5s safety margin)
    if ((ctx.nodes & 4095) == 0) {
        auto now = std::chrono::high_resolution_clock::now();
        int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_start_time).count();
        if (ctx.movetime_ms > 0) {
            int64_t margin = std::min((int64_t)500, ctx.movetime_ms / 20);
            if (ms >= ctx.movetime_ms - margin) {
                ctx.stop = true;
            }
        } else if (ms >= 9500) {
            ctx.stop = true;
        }
    }

    if(ctx.stop){
        return 0;
    }

    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    if(state->game_state == WIN){
        return P_MAX - ply;
    }
    if(state->game_state == DRAW){
        return -20; // Contempt factor
    }

    int rep_score;
    if(state->check_repetition(hist, rep_score)){
        return -20; // Contempt factor to avoid repeating moves in even positions
    }

    bool is_pv = (beta - alpha > 1);
    int orig_alpha = alpha;

    uint64_t hash_key = state->hash();
    Move tt_move = {};
    int tt_score;
    if(probe_tt(hash_key, depth, alpha, beta, tt_score, tt_move, ply)){
        if(!is_pv){
            return tt_score;
        }
    }

    hist.push(hash_key);

    if(depth <= 0){
        int score = quiescence(state, hist, ply, ctx, p, alpha, beta);
        hist.pop(hash_key);
        return score;
    }

    if(!is_pv && depth >= 3 && ply > 0 && !null_move_made){
        int static_eval = custom_evaluate(state);
        if(static_eval >= beta){
            BaseState* null_state = state->create_null_state();
            if(null_state){
                State* ns = static_cast<State*>(null_state);
                int R = 2 + (depth >= 6 ? 1 : 0);
                int null_score = -eval_ctx(ns, depth - 1 - R, hist, ply + 1, ctx, p, -beta, -beta + 1, true);
                delete ns;

                if(null_score >= beta){
                    hist.pop(hash_key);
                    return beta;
                }
            }
        }
    }

    if(!is_pv && depth <= 3 && ply > 0){
        int static_eval = custom_evaluate(state);
        int margin = 200 * depth;
        if(static_eval - margin >= beta){
            hist.pop(hash_key);
            return static_eval - margin;
        }
    }

    sort_moves(state, state->legal_actions, tt_move, ply);

    int best_score = M_MAX;
    Move best_move = {};
    int move_index = 0;

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();
        int score;

        int to_r = (int)action.second.first;
        int to_c = (int)action.second.second;
        int opp = 1 - state->player;
        bool is_capture = state->board.board[opp][to_r][to_c] != 0;
        int piece = state->board.board[state->player][(int)action.first.first][(int)action.first.second];
        bool is_promo = (piece == 1 && (to_r == 0 || to_r == BOARD_H - 1));
        bool is_killer = (ply < MAX_PLY && (action == killers[ply][0] || action == killers[ply][1]));

        if(move_index == 0){
            score = eval_ctx(next, depth - 1, hist, ply + 1, ctx, p, same ? alpha : -beta, same ? beta : -alpha);
            if(!same) score = -score;
        }else{
            int reduction = 0;
            if(depth >= 3 && move_index >= 3 && !is_capture && !is_promo && !is_killer && !is_pv){
                reduction = 1;
                if(move_index >= 6) reduction = 2;
                if(depth - 1 - reduction < 1) reduction = std::max(0, depth - 2);
            }

            score = eval_ctx(next, depth - 1 - reduction, hist, ply + 1, ctx, p, same ? alpha : -alpha - 1, same ? alpha + 1 : -alpha);
            if(!same) score = -score;

            if(reduction > 0 && score > alpha){
                score = eval_ctx(next, depth - 1, hist, ply + 1, ctx, p, same ? alpha : -alpha - 1, same ? alpha + 1 : -alpha);
                if(!same) score = -score;
            }

            if(score > alpha && score < beta){
                score = eval_ctx(next, depth - 1, hist, ply + 1, ctx, p, same ? alpha : -beta, same ? beta : -alpha);
                if(!same) score = -score;
            }
        }

        delete next;

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
                    if(history_table[state->player][from_sq][to_sq] > 50000){
                        for(int p2 = 0; p2 < 2; p2++){
                            for(int f = 0; f < 30; f++){
                                for(int t = 0; t < 30; t++){
                                    history_table[p2][f][t] /= 2;
                                }
                            }
                        }
                    }
                }
            }
            break;
        }

        move_index++;
    }

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
 * Root Search
 * ================================================================ */
SearchResult Submission::search(
    State *state,
    int depth,
    GameHistory& hist,
    SearchContext& ctx
){
    ctx.reset();
    init_tt();
    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if (depth == 1) {
        g_start_time = std::chrono::high_resolution_clock::now();
    }

    std::memset(killers, 0, sizeof(killers));

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }

    uint64_t root_hash = state->hash();
    Move tt_move = {};
    int tt_score;
    bool tt_hit = probe_tt(root_hash, 0, M_MAX, P_MAX, tt_score, tt_move, 0);

    sort_moves(state, state->legal_actions, tt_move, 0);

    int alpha = M_MAX;
    int beta = P_MAX;
    int best_score = M_MAX - 10;
    Move best_move = {};
    int total_moves = (int)state->legal_actions.size();
    
    // Aspiration Windows
    if (depth >= 4 && tt_hit) {
        alpha = std::max(M_MAX, tt_score - 50);
        beta = std::min(P_MAX, tt_score + 50);
    }

    while (true) {
        int orig_alpha = alpha;
        int orig_beta = beta;
        best_score = M_MAX - 10;
        int move_index = 0;
        
        for(auto& action : state->legal_actions){
            State* next = state->next_state(action);
            bool same = next->same_player_as_parent();
            int score;

            if(move_index == 0){
                score = eval_ctx(next, depth - 1, hist, 1, ctx, p, same ? alpha : -beta, same ? beta : -alpha);
                if(!same) score = -score;
            }else{
                score = eval_ctx(next, depth - 1, hist, 1, ctx, p, same ? alpha : -alpha - 1, same ? alpha + 1 : -alpha);
                if(!same) score = -score;

                if(score > alpha && score < beta){
                    score = eval_ctx(next, depth - 1, hist, 1, ctx, p, same ? alpha : -beta, same ? beta : -alpha);
                    if(!same) score = -score;
                }
            }

            delete next;

            if(ctx.stop && depth > 1){
                break;
            }

            if(score > best_score){
                best_score = score;
                best_move = action;
                result.best_move = action;
                result.score = best_score;

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
            break;
        }

        // Check if aspiration window failed
        if (best_score <= orig_alpha && orig_alpha > M_MAX) {
            alpha = M_MAX;
            continue; // Re-search with full lower bound
        } else if (best_score >= orig_beta && orig_beta < P_MAX) {
            beta = P_MAX;
            continue; // Re-search with full upper bound
        }
        break;
    }

    if(!ctx.stop || depth == 1){
        store_tt(root_hash, best_score, depth, TT_EXACT, best_move, 0);
    }

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
