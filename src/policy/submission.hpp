#pragma once
#include "search_types.hpp"
#include "game_history.hpp"

#include "minimax.hpp"

class Submission{
public:
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
