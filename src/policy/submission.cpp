#include <utility>
#include "state.hpp"
#include "submission.hpp"

int Submission::quiescence(
    State *state,
    GameHistory& history,
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
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }

    int stand_pat = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
    if(stand_pat >= beta){
        return beta;
    }
    if(alpha < stand_pat){
        alpha = stand_pat;
    }

    history.push(state->hash());

    int opp = 1 - state->player;
    for(auto& action : state->legal_actions){
        int to_r = action.second.first;
        int to_c = action.second.second;
        // Check if capture or promotion
        bool is_capture = state->board.board[opp][to_r][to_c] != 0;
        int piece = state->board.board[state->player][action.first.first][action.first.second];
        bool is_promo = (piece == 1 && (to_r == 0 || to_r == BOARD_H - 1));
        
        if(is_capture || is_promo){
            State* next = state->next_state(action);
            bool same = next->same_player_as_parent();

            int score = quiescence(next, history, ply + 1, ctx, p, same ? alpha : -beta, same ? beta : -alpha);
            if(!same) score = -score;

            delete next;

            if(score >= beta){
                history.pop(state->hash());
                return beta;
            }
            if(score > alpha){
                alpha = score;
            }
        }
    }

    history.pop(state->hash());
    return alpha;
}

int Submission::eval_ctx(
    State *state,
    int depth,
    GameHistory& history,
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

    /* === Repetition check (game-specific) === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }
    
    history.push(state->hash());

    if(depth <= 0){
        int score = quiescence(state, history, ply, ctx, p, alpha, beta);
        history.pop(state->hash());
        return score;
    }

    /* === PVS (Principal Variation Search) loop === */
    int best_score = M_MAX;
    bool first = true;

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();
        int score;

        if(first){
            score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, same ? alpha : -beta, same ? beta : -alpha);
            if(!same) score = -score;
            first = false;
        }else{
            // Null window search
            score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, same ? alpha : -alpha - 1, same ? alpha + 1 : -alpha);
            if(!same) score = -score;

            if(score > alpha && score < beta){
                // Re-search with full window
                score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, same ? alpha : -beta, same ? beta : -score);
                if(!same) score = -score;
            }
        }

        delete next;

        if(score > best_score){
            best_score = score;
        }
        if(score > alpha){
            alpha = score;
        }
        if(alpha >= beta){
            break; // Alpha-beta pruning
        }
    }

    history.pop(state->hash());
    return best_score;
}

SearchResult Submission::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }

    int best_score = M_MAX - 10;
    int alpha = M_MAX;
    int beta = P_MAX;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();
    bool first = true;

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();
        int score;

        if(first){
            score = eval_ctx(next, depth - 1, history, 1, ctx, p, same ? alpha : -beta, same ? beta : -alpha);
            if(!same) score = -score;
            first = false;
        }else{
            score = eval_ctx(next, depth - 1, history, 1, ctx, p, same ? alpha : -alpha - 1, same ? alpha + 1 : -alpha);
            if(!same) score = -score;

            if(score > alpha && score < beta){
                score = eval_ctx(next, depth - 1, history, 1, ctx, p, same ? alpha : -beta, same ? beta : -score);
                if(!same) score = -score;
            }
        }
        
        delete next;

        if(score > best_score){
            best_score = score;
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
