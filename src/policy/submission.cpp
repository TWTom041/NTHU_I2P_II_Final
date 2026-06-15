#include <utility>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include "state.hpp"
#include "submission.hpp"

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

    // Adjust mate scores for storage (ply-independent)
    int adj_score = score;
    if(score > P_MAX - 200) adj_score += ply;
    else if(score < M_MAX + 200) adj_score -= ply;

    // Replace if: empty, same position, or we searched deeper
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
        // Adjust mate scores for retrieval
        if(score > P_MAX - 200) score -= ply;
        else if(score < M_MAX + 200) score += ply;

        if(entry.flag == TT_EXACT) return true;
        if(entry.flag == TT_ALPHA && score <= alpha){ score = alpha; return true; }
        if(entry.flag == TT_BETA && score >= beta){ score = beta; return true; }
    }
    return false;
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

        // TT move gets highest priority
        if(m == tt_move && tt_move.first != tt_move.second){
            s = 1000000;
        } else {
            int to_r = (int)m.second.first;
            int to_c = (int)m.second.second;
            int victim = state->board.board[opp][to_r][to_c];

            if(victim){
                // MVV-LVA: prioritize capturing valuable pieces with cheap pieces
                int from_r = (int)m.first.first;
                int from_c = (int)m.first.second;
                int attacker = state->board.board[self][from_r][from_c];
                s = 100000 + PIECE_VALUES[victim] * 10 - PIECE_VALUES[attacker];
            } else {
                // Check killer moves
                if(ply < MAX_PLY){
                    if(m == killers[ply][0]){
                        s = 90000;
                    } else if(m == killers[ply][1]){
                        s = 80000;
                    }
                }
                // History heuristic
                int from_sq = move_to_sq(m.first);
                int to_sq = move_to_sq(m.second);
                if(from_sq >= 0 && from_sq < 30 && to_sq >= 0 && to_sq < 30){
                    s += history_table[self][from_sq][to_sq];
                }
            }

            // Pawn promotion bonus
            int piece = state->board.board[self][(int)m.first.first][(int)m.first.second];
            if(piece == 1 && (to_r == 0 || to_r == BOARD_H - 1)){
                s += 95000;
            }
        }

        scored[i] = {m, s};
    }

    // Sort by score descending
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
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
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

    int stand_pat = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &hist);
    if(stand_pat >= beta){
        return beta;
    }

    // Delta pruning
    const int DELTA_MARGIN = 250;
    if(stand_pat + DELTA_MARGIN < alpha){
        return alpha;
    }

    if(alpha < stand_pat){
        alpha = stand_pat;
    }

    hist.push(state->hash());

    int opp = 1 - state->player;

    // Build list of captures and promotions, sorted by MVV-LVA
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
                mvv_lva = PIECE_VALUES[victim] * 10 - PIECE_VALUES[piece];
            }
            if(is_promo) mvv_lva += 900;

            if(num_captures < 64){
                captures[num_captures++] = {action, mvv_lva};
            }
        }
    }

    // Sort captures by MVV-LVA descending
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
    int beta
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    /* === Lazy move generation (sets game_state) === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */
    if(state->game_state == WIN){
        return P_MAX - ply;
    }
    if(state->game_state == DRAW){
        return 0;
    }

    /* === Repetition check === */
    int rep_score;
    if(state->check_repetition(hist, rep_score)){
        return rep_score;
    }

    bool is_pv = (beta - alpha > 1);
    int orig_alpha = alpha;

    /* === Transposition Table Probe === */
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

    /* === Null-Move Pruning === */
    if(!is_pv && depth >= 3 && ply > 0){
        int static_eval = state->evaluate(p.use_kp_eval, false, &hist);
        if(static_eval >= beta){
            BaseState* null_state = state->create_null_state();
            if(null_state){
                State* ns = static_cast<State*>(null_state);
                int R = 2 + (depth >= 6 ? 1 : 0);
                int null_score = -eval_ctx(ns, depth - 1 - R, hist, ply + 1, ctx, p, -beta, -beta + 1);
                delete ns;

                if(null_score >= beta){
                    hist.pop(hash_key);
                    return beta;
                }
            }
        }
    }

    /* === Reverse Futility Pruning === */
    if(!is_pv && depth <= 3 && ply > 0){
        int static_eval = state->evaluate(p.use_kp_eval, false, &hist);
        int margin = 120 * depth;
        if(static_eval - margin >= beta){
            hist.pop(hash_key);
            return static_eval - margin;
        }
    }

    /* === Move Ordering === */
    sort_moves(state, state->legal_actions, tt_move, ply);

    /* === PVS loop === */
    int best_score = M_MAX;
    Move best_move = {};
    int move_index = 0;

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();
        int score;

        // Determine move characteristics for LMR
        int to_r = (int)action.second.first;
        int to_c = (int)action.second.second;
        int opp = 1 - state->player;
        bool is_capture = state->board.board[opp][to_r][to_c] != 0;
        int piece = state->board.board[state->player][(int)action.first.first][(int)action.first.second];
        bool is_promo = (piece == 1 && (to_r == 0 || to_r == BOARD_H - 1));
        bool is_killer = (ply < MAX_PLY && (action == killers[ply][0] || action == killers[ply][1]));

        if(move_index == 0){
            // First move: full window search
            score = eval_ctx(next, depth - 1, hist, ply + 1, ctx, p, same ? alpha : -beta, same ? beta : -alpha);
            if(!same) score = -score;
        }else{
            /* === Late Move Reductions (LMR) === */
            int reduction = 0;
            if(depth >= 3 && move_index >= 3 && !is_capture && !is_promo && !is_killer && !is_pv){
                reduction = 1;
                if(move_index >= 6) reduction = 2;
                if(depth - 1 - reduction < 1) reduction = std::max(0, depth - 2);
            }

            // Null window search (with possible LMR)
            score = eval_ctx(next, depth - 1 - reduction, hist, ply + 1, ctx, p, same ? alpha : -alpha - 1, same ? alpha + 1 : -alpha);
            if(!same) score = -score;

            // Re-search at full depth if LMR was applied and score is promising
            if(reduction > 0 && score > alpha){
                score = eval_ctx(next, depth - 1, hist, ply + 1, ctx, p, same ? alpha : -alpha - 1, same ? alpha + 1 : -alpha);
                if(!same) score = -score;
            }

            // Full-window re-search if null window failed high
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
            // Beta cutoff: update killer moves and history for quiet moves
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

    /* === Store in TT === */
    TTFlag flag;
    if(best_score <= orig_alpha){
        flag = TT_ALPHA;  // fail-low: upper bound
    } else if(best_score >= beta){
        flag = TT_BETA;   // fail-high: lower bound
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

    // Clear killers at start of each depth (but keep TT and history)
    std::memset(killers, 0, sizeof(killers));

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }

    /* === Probe TT for best move from previous iteration === */
    uint64_t root_hash = state->hash();
    Move tt_move = {};
    int tt_score;
    probe_tt(root_hash, 0, M_MAX, P_MAX, tt_score, tt_move, 0);

    /* === Move ordering at root === */
    sort_moves(state, state->legal_actions, tt_move, 0);

    int best_score = M_MAX - 10;
    int alpha = M_MAX;
    int beta = P_MAX;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();
    Move best_move = {};

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();
        int score;

        if(move_index == 0){
            score = eval_ctx(next, depth - 1, hist, 1, ctx, p, same ? alpha : -beta, same ? beta : -alpha);
            if(!same) score = -score;
        }else{
            // Null window
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

    // Store root result in TT
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
        {"ReportPartial", "true"},
    };
}

std::vector<ParamDef> Submission::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
    };
}
