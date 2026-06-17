#pragma once
/* ================================================================
 * NNUE evaluation for MiniChess (6x5).
 *
 * Small fully-connected net  360 -> HIDDEN -> 1  with a tanh output,
 * trained on self-play WDL data by tools/nnue_train.cpp. Weights are
 * embedded in nnue_weights.hpp.
 *
 * Feature encoding MUST stay identical to tools/nnue_train.cpp:
 *   Board is canonicalised to the side-to-move's perspective. White (stm 0)
 *   keeps orientation; Black (stm 1) is rotated 180 deg (r->5-r, c->4-c).
 *   feature = base + (piece-1)*30 + sq      (piece in 1..6)
 *     own pieces:   base = 0
 *     enemy pieces: base = 180
 *     sq(stm=0) = r*5 + c ;  sq(stm=1) = (5-r)*5 + (4-c)
 *
 * The board is tiny (<= 20 pieces), so the accumulator is built sparsely
 * from scratch at every leaf -- no incremental update / make-unmake bugs.
 * ================================================================ */
#include <cmath>
#include "state.hpp"
#include "nnue_weights.hpp"

namespace NNUE {

inline int persp_sq(int r, int c, int stm){
    return (stm == 0) ? (r * BOARD_W + c)
                      : ((BOARD_H - 1 - r) * BOARD_W + (BOARD_W - 1 - c));
}

/* Evaluate from the side-to-move's perspective, in centipawn-like units. */
inline int evaluate(const State* s){
    const int H = nnue_data::HIDDEN;
    float acc[64];
    for(int h = 0; h < H; h++) acc[h] = nnue_data::B1[h];

    const int stm = s->player;
    const auto& self  = s->board.board[stm];
    const auto& enemy = s->board.board[1 - stm];

    for(int r = 0; r < BOARD_H; r++){
        for(int c = 0; c < BOARD_W; c++){
            int p = self[r][c];
            if(p){
                int f = (p - 1) * 30 + persp_sq(r, c, stm);
                const float* w = &nnue_data::W1[(size_t)f * H];
                for(int h = 0; h < H; h++) acc[h] += w[h];
            }
            int q = enemy[r][c];
            if(q){
                int f = 180 + (q - 1) * 30 + persp_sq(r, c, stm);
                const float* w = &nnue_data::W1[(size_t)f * H];
                for(int h = 0; h < H; h++) acc[h] += w[h];
            }
        }
    }

    float pre = nnue_data::B2;
    for(int h = 0; h < H; h++){
        float a = acc[h] > 0 ? acc[h] : 0.0f;   // ReLU
        pre += nnue_data::W2[h] * a;
    }
    float out = std::tanh(pre);
    return (int)std::lround(out * nnue_data::OUT_SCALE);
}

} // namespace NNUE
