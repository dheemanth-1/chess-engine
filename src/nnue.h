#pragma once
// nnue.h
// ─────────────────────────────────────────────────────────────────────────────
// Self-contained NNUE inference for a HalfKP network trained in PyTorch.
// No LibTorch dependency — reads raw weights from nnue_weights.bin and
// performs the forward pass using plain float arithmetic.
//
// Architecture being inferred:
//   HalfKP (40 960 features) → FT (256) × 2 perspectives   [ClippedReLU]
//   concat  (512) → L1 (32) [ClippedReLU] → L2 (32) [ClippedReLU] → out (1)
// ─────────────────────────────────────────────────────────────────────────────

#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <stdexcept>

// ─── Network dimensions ───────────────────────────────────────────────────────
// These must match the constants used during Python training exactly.
// If you change FT_SIZE or hidden sizes in training, change them here too.
static constexpr int HALFKP_FEATURES = 40960;        // 64 king sq × 10 piece types × 64 piece sq
static constexpr int FT_SIZE = 256;           // accumulator width per perspective
static constexpr int INPUT_SIZE = FT_SIZE * 2;   // 512 — two perspectives concatenated
static constexpr int L1_SIZE = 32;
static constexpr int L2_SIZE = 32;


// ─────────────────────────────────────────────────────────────────────────────
// NNUEWeights
// Holds every parameter of the network, loaded from the binary file.
// Memory footprint: ft_weight alone is 40960×256×4 ≈ 42 MB.  The rest is <1 MB.
// Load once at engine startup and keep alive for the entire session.
// ─────────────────────────────────────────────────────────────────────────────
struct NNUEWeights {
    std::vector<float>                        ft_weight;   // [40960][256]
    std::array<float, FT_SIZE>                ft_bias;
    std::array<float, L1_SIZE* INPUT_SIZE>   l1_weight;   // [32][512]
    std::array<float, L1_SIZE>                l1_bias;
    std::array<float, L2_SIZE* L1_SIZE>      l2_weight;   // [32][32]
    std::array<float, L2_SIZE>                l2_bias;
    std::array<float, L2_SIZE>                out_weight;
    float                                     out_bias;

    bool load (const std::string& path);
};

// ─────────────────────────────────────────────────────────────────────────────
// NNUEAccumulator
// Stores the result of the feature-transformer layer for both perspectives.
// Lives on the search stack — one accumulator per ply, copied and updated
// incrementally as moves are made and unmade.
//
// dirty = true  → accumulator is stale; refresh_accumulator() must be called
//                 before running the forward pass.
// dirty = false → accumulator is valid for the current position.
// ─────────────────────────────────────────────────────────────────────────────
struct NNUEAccumulator {
    // alignas(32) guarantees 32-byte alignment required by AVX2 _mm256_load_ps.
    // Without this, the load instruction either falls back to a slower unaligned
    // form or faults on strict CPUs. std::array doesn't guarantee this by default.
    alignas(32) std::array<float, FT_SIZE> white;
    alignas(32) std::array<float, FT_SIZE> black;
    bool dirty = true;
};

// ─────────────────────────────────────────────────────────────────────────────
// NNUEPosition
// A board description that is independent of any chess library.
// Populate this from your engine's board object before calling nnue_evaluate().
//
// Square convention: a1=0, b1=1, … h1=7, a2=8, … h8=63  (same as python-chess)
// ─────────────────────────────────────────────────────────────────────────────
struct NNUEPosition {
    int  white_king_sq;
    int  black_king_sq;
    bool white_to_move;

    struct Piece {
        int  sq;
        int  type;      // 0=P 1=N 2=B 3=R 4=Q
        bool is_white;
    };
    std::vector<Piece> pieces;
};


// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

// Load weights from the binary file exported by export_weights() in Python.
// Call once at startup. Returns false on IO failure.

bool  load_nnue_weights (const std::string& path, NNUEWeights& weights);
// Recompute both perspective accumulators from scratch for a given position.
// Call this whenever you enter a new position that wasn't reached by making
// a single incremental move (e.g. at the root, or after a null move).
void  refresh_accumulator (const NNUEPosition& pos, const NNUEWeights& weights, NNUEAccumulator& acc);
// Run the full forward pass and return a centipawn-equivalent score.
// Positive = white is better, negative = black is better.
// Automatically refreshes the accumulator if acc.dirty == true.
float nnue_evaluate (const NNUEPosition& pos, const NNUEWeights& weights, NNUEAccumulator& acc);
// No ClippedReLU is applied — accumulator remains pre-activation.
void nnue_add_feature (NNUEAccumulator& acc, const NNUEWeights& w,
    int wk, int bk, int sq, int pt, bool is_white);

void nnue_remove_feature (NNUEAccumulator& acc, const NNUEWeights& w,
    int wk, int bk, int sq, int pt, bool is_white);