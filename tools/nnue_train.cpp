/* ================================================================
 * NNUE trainer for MiniChess (6x5)
 *
 * Pipeline (all in C++, no external ML deps):
 *   1. Self-play games using the existing MiniMax search to label data.
 *      Random openings + epsilon-randomness give position diversity.
 *      Every non-terminal position is labelled with the eventual game
 *      result (WDL: +1 win / 0 draw / -1 loss) from the side-to-move's
 *      perspective.
 *   2. Train a small fully-connected net  360 -> HIDDEN -> 1  (Adam, MSE,
 *      tanh output) to predict that WDL value.
 *   3. Emit src/policy/nnue_weights.hpp (embedded float weights).
 *
 * Feature encoding (MUST stay identical to src/policy/nnue.hpp):
 *   Board is canonicalised to the side-to-move's perspective. White (stm 0)
 *   keeps orientation; Black (stm 1) is rotated 180 deg (r->5-r, c->4-c),
 *   because Black's starting array is White's rotated 180 deg.
 *   feature index = base + (piece-1)*30 + sq          (piece in 1..6)
 *     own pieces:   base = 0
 *     enemy pieces: base = 180
 *     sq(stm=0) = r*5 + c ;  sq(stm=1) = (5-r)*5 + (4-c)
 *   -> 2 * 6 * 30 = 360 features.
 * ================================================================ */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

#include "state.hpp"
#include "minimax.hpp"
#include "game_history.hpp"
#include "search_types.hpp"

static const int NF = 360;   // features
static int   HIDDEN = 32;
static float OUT_SCALE = 600.0f;

/* ---- shared feature extraction ---- */
static inline int persp_sq(int r, int c, int stm){
    return (stm == 0) ? (r * BOARD_W + c)
                      : ((BOARD_H - 1 - r) * BOARD_W + (BOARD_W - 1 - c));
}

// Fill `out` with the active feature indices for the position; returns count.
static int extract_features(const State* s, int* out){
    int stm = s->player;
    const auto& self = s->board.board[stm];
    const auto& enemy = s->board.board[1 - stm];
    int n = 0;
    for(int r = 0; r < BOARD_H; r++){
        for(int c = 0; c < BOARD_W; c++){
            int p = self[r][c];
            if(p){ out[n++] = (p - 1) * 30 + persp_sq(r, c, stm); }
            int q = enemy[r][c];
            if(q){ out[n++] = 180 + (q - 1) * 30 + persp_sq(r, c, stm); }
        }
    }
    return n;
}

/* ================= Self-play data generation ================= */

struct Sample {
    int feat[20];   // active feature indices (<= 20 pieces on a 6x5 board)
    int nfeat;
    int stm;        // side to move at this position
    float target;   // filled after game ends
};

static std::mt19937 rng(0xC0FFEEu);

int main(int argc, char** argv){
    int   games   = (argc > 1) ? atoi(argv[1]) : 3000;
    int   depth   = (argc > 2) ? atoi(argv[2]) : 4;
    int   epochs  = (argc > 3) ? atoi(argv[3]) : 40;
    HIDDEN        = (argc > 4) ? atoi(argv[4]) : 32;
    OUT_SCALE     = (argc > 5) ? (float)atof(argv[5]) : 600.0f;
    unsigned seed = (argc > 6) ? (unsigned)atoi(argv[6]) : 0xC0FFEEu;
    rng.seed(seed);

    const int MAX_PLIES = 120;
    const double EPS = 0.12;   // chance of a random move (diversity)

    std::vector<Sample> data;
    data.reserve((size_t)games * 30);

    auto t0 = std::chrono::high_resolution_clock::now();
    int win0 = 0, win1 = 0, draws = 0;

    for(int g = 0; g < games; g++){
        GameHistory hist;
        State* s = new State();
        s->get_legal_actions();
        hist.push(s->hash());

        std::uniform_int_distribution<int> open_dist(1, 8);
        int opening = open_dist(rng);

        std::vector<Sample> game_samples;
        int winner = -2;  // -2 unknown, -1 draw, 0/1 winner
        int ply = 0;

        while(true){
            if(s->game_state == WIN){ winner = s->player; break; }
            if(s->legal_actions.empty()){ winner = -1; break; }
            if(ply >= MAX_PLIES){ winner = -1; break; }
            if(hist.count(s->hash()) >= 3){ winner = -1; break; }

            // Record the position (non-terminal) for training.
            Sample smp;
            smp.nfeat = extract_features(s, smp.feat);
            smp.stm = s->player;
            smp.target = 0.0f;
            game_samples.push_back(smp);

            // Choose a move.
            std::uniform_real_distribution<double> ur(0.0, 1.0);
            Move mv;
            if(ply < opening || ur(rng) < EPS){
                std::uniform_int_distribution<int> md(0, (int)s->legal_actions.size() - 1);
                mv = s->legal_actions[md(rng)];
            } else {
                SearchContext ctx;
                ctx.params = MiniMax::default_params();
                SearchResult r = MiniMax::search(s, depth, hist, ctx);
                mv = r.best_move;
            }

            State* ns = s->next_state(mv);
            ns->get_legal_actions();
            delete s;
            s = ns;
            hist.push(s->hash());
            ply++;
        }
        delete s;

        if(winner == 0) win0++;
        else if(winner == 1) win1++;
        else draws++;

        // Label samples by eventual result from each position's stm perspective.
        for(auto& smp : game_samples){
            if(winner < 0) smp.target = 0.0f;
            else smp.target = (smp.stm == winner) ? 1.0f : -1.0f;
            data.push_back(smp);
        }

        if((g + 1) % 200 == 0){
            auto t = std::chrono::high_resolution_clock::now();
            double el = std::chrono::duration_cast<std::chrono::milliseconds>(t - t0).count() / 1000.0;
            fprintf(stderr, "  games %d/%d  samples=%zu  W=%d B=%d D=%d  (%.1fs)\n",
                    g + 1, games, data.size(), win0, win1, draws, el);
        }
    }

    fprintf(stderr, "Generated %zu samples from %d games (W=%d B=%d D=%d)\n",
            data.size(), games, win0, win1, draws);
    if(data.empty()){ fprintf(stderr, "No data!\n"); return 1; }

    /* ================= Train  360 -> HIDDEN -> 1 ================= */
    int H = HIDDEN;
    std::vector<float> W1((size_t)NF * H), B1(H), W2(H);
    float B2 = 0.0f;

    std::uniform_real_distribution<float> iw(-0.10f, 0.10f);
    std::uniform_real_distribution<float> ow(-0.30f, 0.30f);
    for(auto& w : W1) w = iw(rng);
    for(auto& b : B1) b = 0.0f;
    for(auto& w : W2) w = ow(rng);

    // Adam state
    std::vector<float> mW1((size_t)NF * H, 0), vW1((size_t)NF * H, 0);
    std::vector<float> mB1(H, 0), vB1(H, 0), mW2(H, 0), vW2(H, 0);
    float mB2 = 0, vB2 = 0;
    const float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 1e-6f;

    std::vector<int> idx(data.size());
    for(size_t i = 0; i < idx.size(); i++) idx[i] = (int)i;

    const int BATCH = 256;
    std::vector<float> acc(H), a(H), gW2(H), gB1(H);
    int adam_t = 0;

    for(int ep = 0; ep < epochs; ep++){
        std::shuffle(idx.begin(), idx.end(), rng);
        double sse = 0.0;
        size_t nb = 0;

        for(size_t start = 0; start < idx.size(); start += BATCH){
            size_t end = std::min(idx.size(), start + (size_t)BATCH);
            int bs = (int)(end - start);
            adam_t++;

            // gradient accumulators for this batch
            // (W1 grad accumulated sparsely)
            std::vector<float> gW1((size_t)NF * H, 0.0f);
            std::fill(gW2.begin(), gW2.end(), 0.0f);
            std::fill(gB1.begin(), gB1.end(), 0.0f);
            float gB2 = 0.0f;

            for(size_t k = start; k < end; k++){
                const Sample& smp = data[idx[k]];
                // forward
                for(int h = 0; h < H; h++) acc[h] = B1[h];
                for(int f = 0; f < smp.nfeat; f++){
                    const float* w = &W1[(size_t)smp.feat[f] * H];
                    for(int h = 0; h < H; h++) acc[h] += w[h];
                }
                float pre = B2;
                for(int h = 0; h < H; h++){
                    a[h] = acc[h] > 0 ? acc[h] : 0.0f;   // ReLU
                    pre += W2[h] * a[h];
                }
                float out = std::tanh(pre);
                float diff = out - smp.target;
                sse += diff * diff;

                // backward
                float dpre = 2.0f * diff * (1.0f - out * out);
                gB2 += dpre;
                for(int h = 0; h < H; h++){
                    gW2[h] += dpre * a[h];
                    float da = dpre * W2[h];
                    float dacc = (acc[h] > 0) ? da : 0.0f;
                    gB1[h] += dacc;
                    if(dacc != 0.0f){
                        for(int f = 0; f < smp.nfeat; f++){
                            gW1[(size_t)smp.feat[f] * H + h] += dacc;
                        }
                    }
                }
            }
            nb++;

            // Adam update (gradients averaged over batch)
            float invb = 1.0f / bs;
            float bc1 = 1.0f - std::pow(b1, (float)adam_t);
            float bc2 = 1.0f - std::pow(b2, (float)adam_t);
            auto adam = [&](float& w, float& m, float& v, float gr){
                gr = gr * invb + wd * w;
                m = b1 * m + (1 - b1) * gr;
                v = b2 * v + (1 - b2) * gr * gr;
                float mh = m / bc1, vh = v / bc2;
                w -= lr * mh / (std::sqrt(vh) + eps);
            };
            for(size_t i = 0; i < W1.size(); i++) adam(W1[i], mW1[i], vW1[i], gW1[i]);
            for(int h = 0; h < H; h++){
                adam(B1[h], mB1[h], vB1[h], gB1[h]);
                adam(W2[h], mW2[h], vW2[h], gW2[h]);
            }
            adam(B2, mB2, vB2, gB2);
        }
        if((ep + 1) % 5 == 0 || ep == 0){
            fprintf(stderr, "  epoch %d/%d  mse=%.5f\n", ep + 1, epochs, sse / data.size());
        }
    }

    /* ================= Emit weights header ================= */
    const char* outpath = "src/policy/nnue_weights.hpp";
    FILE* fp = fopen(outpath, "w");
    if(!fp){ fprintf(stderr, "cannot open %s\n", outpath); return 1; }
    fprintf(fp, "#pragma once\n");
    fprintf(fp, "// Auto-generated by tools/nnue_train.cpp. Do not edit.\n");
    fprintf(fp, "// games trained: %d, hidden: %d, samples: %zu\n", games, H, data.size());
    fprintf(fp, "namespace nnue_data {\n");
    fprintf(fp, "constexpr int   NUM_FEATURES = %d;\n", NF);
    fprintf(fp, "constexpr int   HIDDEN       = %d;\n", H);
    fprintf(fp, "constexpr float OUT_SCALE    = %.1ff;\n", OUT_SCALE);

    fprintf(fp, "inline const float W1[%d] = {\n", NF * H);
    for(size_t i = 0; i < W1.size(); i++){
        fprintf(fp, "%.8gf,", W1[i]);
        if((i + 1) % 8 == 0) fprintf(fp, "\n");
    }
    fprintf(fp, "};\n");

    fprintf(fp, "inline const float B1[%d] = {", H);
    for(int h = 0; h < H; h++) fprintf(fp, "%.8gf,", B1[h]);
    fprintf(fp, "};\n");

    fprintf(fp, "inline const float W2[%d] = {", H);
    for(int h = 0; h < H; h++) fprintf(fp, "%.8gf,", W2[h]);
    fprintf(fp, "};\n");

    fprintf(fp, "inline const float B2 = %.8gf;\n", B2);
    fprintf(fp, "} // namespace nnue_data\n");
    fclose(fp);

    fprintf(stderr, "Wrote %s\n", outpath);
    return 0;
}
