/* ================================================================
 * NNUE self-play DATA GENERATOR for MiniChess (6x5)
 *
 * Plays self-play games using the strong `submission` search (internal
 * iterative deepening to a fixed depth) with random openings + epsilon
 * noise for diversity, then dumps every non-terminal position as a
 * training sample to a binary file. The PyTorch trainer (tools/train_nnue.py)
 * consumes these files and emits src/policy/nnue_weights.hpp.
 *
 * Feature encoding MUST stay identical to src/policy/nnue.hpp:
 *   Side-to-move perspective. White (stm 0) keeps orientation; Black (stm 1)
 *   is rotated 180 deg. feature = base + (piece-1)*30 + sq, piece in 1..6.
 *     own pieces:   base = 0 ; enemy pieces: base = 180
 *     sq(stm=0) = r*5 + c ; sq(stm=1) = (5-r)*5 + (4-c)   -> 360 features.
 *
 * Binary record format (little-endian, host):
 *   uint8  nfeat
 *   int16  feat[nfeat]      (0..359)
 *   uint8  stm              (0/1)
 *   int8   wdl              (-1/0/+1, from stm perspective)
 *   int16  score            (teacher search score, cp, clamped +-3000)
 *
 * Usage: nnue_gen <out_file> [games=4000] [depth=5] [seed=1] [eps=0.10]
 * ================================================================ */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

#include "state.hpp"
#include "submission.hpp"
#include "game_history.hpp"
#include "search_types.hpp"

static inline int persp_sq(int r, int c, int stm){
    return (stm == 0) ? (r * BOARD_W + c)
                      : ((BOARD_H - 1 - r) * BOARD_W + (BOARD_W - 1 - c));
}

static int extract_features(const State* s, int16_t* out){
    int stm = s->player;
    const auto& self = s->board.board[stm];
    const auto& enemy = s->board.board[1 - stm];
    int n = 0;
    for(int r = 0; r < BOARD_H; r++){
        for(int c = 0; c < BOARD_W; c++){
            int p = self[r][c];
            if(p){ out[n++] = (int16_t)((p - 1) * 30 + persp_sq(r, c, stm)); }
            int q = enemy[r][c];
            if(q){ out[n++] = (int16_t)(180 + (q - 1) * 30 + persp_sq(r, c, stm)); }
        }
    }
    return n;
}

struct Sample {
    int16_t feat[20];
    uint8_t nfeat;
    uint8_t stm;
    int16_t score;
};

/* Strong move selection: internal iterative deepening with submission. */
static SearchResult search_to_depth(State* s, GameHistory& hist, int depth){
    SearchContext ctx;
    ctx.params = Submission::default_params();
    ctx.movetime_ms = 0;        // depth-bounded; hard 9.5s cap never hit
    SearchResult r;
    for(int d = 1; d <= depth; d++){
        r = Submission::search(s, d, hist, ctx);
        if(ctx.stop) break;
    }
    return r;
}

int main(int argc, char** argv){
    if(argc < 2){
        fprintf(stderr, "usage: %s <out_file> [games] [depth] [seed] [eps]\n", argv[0]);
        return 1;
    }
    const char* outpath = argv[1];
    int    games = (argc > 2) ? atoi(argv[2]) : 4000;
    int    depth = (argc > 3) ? atoi(argv[3]) : 5;
    unsigned seed = (argc > 4) ? (unsigned)strtoul(argv[4], nullptr, 10) : 1u;
    double eps   = (argc > 5) ? atof(argv[5]) : 0.10;

    std::mt19937 rng(seed);
    const int MAX_PLIES = 120;

    FILE* fp = fopen(outpath, "wb");
    if(!fp){ fprintf(stderr, "cannot open %s\n", outpath); return 1; }

    Submission::init_tt();

    auto t0 = std::chrono::high_resolution_clock::now();
    int win0 = 0, win1 = 0, draws = 0;
    size_t total_samples = 0;

    std::uniform_int_distribution<int> open_dist(1, 8);
    std::uniform_real_distribution<double> ur(0.0, 1.0);

    for(int g = 0; g < games; g++){
        GameHistory hist;
        State* s = new State();
        s->get_legal_actions();
        hist.push(s->hash());

        int opening = open_dist(rng);
        std::vector<Sample> gs;
        int winner = -2;  // -2 unknown, -1 draw, 0/1 winner
        int ply = 0;

        while(true){
            if(s->game_state == WIN){ winner = s->player; break; }
            if(s->legal_actions.empty()){ winner = -1; break; }
            if(ply >= MAX_PLIES){ winner = -1; break; }
            if(hist.count(s->hash()) >= 3){ winner = -1; break; }

            Move mv;
            int16_t teacher = 0;
            bool record = true;

            if(ply < opening || ur(rng) < eps){
                std::uniform_int_distribution<int> md(0, (int)s->legal_actions.size() - 1);
                mv = s->legal_actions[md(rng)];
                record = false;  // skip noisy/forced-opening positions
            } else {
                SearchResult r = search_to_depth(s, hist, depth);
                mv = r.best_move;
                int sc = r.score;
                if(sc >  3000) sc =  3000;
                if(sc < -3000) sc = -3000;
                teacher = (int16_t)sc;
            }

            if(record){
                Sample smp;
                smp.nfeat = (uint8_t)extract_features(s, smp.feat);
                smp.stm = (uint8_t)s->player;
                smp.score = teacher;
                gs.push_back(smp);
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

        for(auto& smp : gs){
            int8_t wdl;
            if(winner < 0) wdl = 0;
            else wdl = (smp.stm == winner) ? 1 : -1;

            fwrite(&smp.nfeat, 1, 1, fp);
            fwrite(smp.feat, sizeof(int16_t), smp.nfeat, fp);
            fwrite(&smp.stm, 1, 1, fp);
            fwrite(&wdl, 1, 1, fp);
            fwrite(&smp.score, sizeof(int16_t), 1, fp);
            total_samples++;
        }

        if((g + 1) % 100 == 0){
            auto t = std::chrono::high_resolution_clock::now();
            double el = std::chrono::duration_cast<std::chrono::milliseconds>(t - t0).count() / 1000.0;
            fprintf(stderr, "[seed %u] games %d/%d  samples=%zu  W=%d B=%d D=%d  (%.1fs)\n",
                    seed, g + 1, games, total_samples, win0, win1, draws, el);
        }
    }

    fclose(fp);
    fprintf(stderr, "[seed %u] DONE %zu samples from %d games (W=%d B=%d D=%d) -> %s\n",
            seed, total_samples, games, win0, win1, draws, outpath);
    return 0;
}
