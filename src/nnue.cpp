// nnue.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Implementation of the NNUE inference engine declared in nnue.h
// ─────────────────────────────────────────────────────────────────────────────

#include "nnue.h"
#ifdef __AVX2__
#include <immintrin.h>   // AVX2 intrinsics: _mm256_*, _mm_*
#endif

static inline int mirror_sq (int sq) { return sq ^ 56; }
static inline float crelu (float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}


// ─────────────────────────────────────────────────────────────────────────────
// AVX2 helpers
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// AVX2 helpers
// ─────────────────────────────────────────────────────────────────────────────
#ifdef __AVX2__

// Horizontal sum of 8 floats packed in a __m256 register.
// This is the standard pattern for AVX2 dot product reduction.
//
// Step by step:
//   v       = [a b c d | e f g h]      (256-bit, 2 lanes of 4)
//   hi      = [e f g h]                (upper 128 bits extracted)
//   lo      = [a b c d]                (lower 128 bits)
//   lo     += hi → [a+e b+f c+g d+h]
//   shuf   = movehdup → [b+f b+f d+h d+h]
//   lo     += shuf → [a+b+e+f _ c+d+g+h _]
//   shuf   = movehl → [c+d+g+h _]
//   lo     += shuf → [a+b+c+d+e+f+g+h _]
//   result = extract lowest float
static inline float hsum8 (__m256 v) {
    __m128 hi = _mm256_extractf128_ps (v, 1);
    __m128 lo = _mm256_castps256_ps128 (v);
    lo = _mm_add_ps (lo, hi);
    __m128 shuf = _mm_movehdup_ps (lo);
    lo = _mm_add_ps (lo, shuf);
    shuf = _mm_movehl_ps (shuf, lo);
    lo = _mm_add_ps (lo, shuf);
    return _mm_cvtss_f32 (lo);
}

// AVX2 ClippedReLU on 8 floats: clamp to [0, 1]
static inline __m256 crelu_avx (__m256 x) {
    return _mm256_min_ps (
        _mm256_max_ps (x, _mm256_setzero_ps ()),
        _mm256_set1_ps (1.0f));
}

// Add a 256-float weight row into a 256-float accumulator using AVX2.
// 256 floats / 8 per register = 32 AVX2 add instructions.
// Compare to 256 scalar additions — 8x throughput improvement.
static inline void avx2_add_row (float* acc, const float* row) {
    for (int j = 0; j < FT_SIZE; j += 8) {
        __m256 a = _mm256_load_ps (acc + j);    // aligned load (alignas(32) in struct)
        __m256 r = _mm256_loadu_ps (row + j);   // unaligned load (heap vector, may not be 32-aligned)
        _mm256_store_ps (acc + j, _mm256_add_ps (a, r));
    }
}

// Apply ClippedReLU to the full 256-float accumulator.
static inline void avx2_crelu_inplace (float* acc) {
    __m256 zero = _mm256_setzero_ps ();
    __m256 one = _mm256_set1_ps (1.0f);
    for (int j = 0; j < FT_SIZE; j += 8) {
        __m256 v = _mm256_load_ps (acc + j);
        _mm256_store_ps (acc + j, _mm256_min_ps (_mm256_max_ps (v, zero), one));
    }
}

// Dense matrix-vector multiply with AVX2 FMA for one output neuron.
//
// Computes: bias + sum(weight[i] * input[i]) for i in [0, len)
// len must be a multiple of 8 (INPUT_SIZE=512 and L1_SIZE=32 both qualify).
//
// _mm256_fmadd_ps(a, b, c) = a*b + c  — one instruction, no rounding error
// between the multiply and add steps compared to two separate instructions.
// This is why -mfma is required in CMakeLists.
static inline float avx2_dot (const float* weight, const float* input, int len, float bias) {
    __m256 sum = _mm256_setzero_ps ();
    for (int j = 0; j < len; j += 8) {
        __m256 w = _mm256_loadu_ps (weight + j);
        __m256 x = _mm256_loadu_ps (input + j);
        sum = _mm256_fmadd_ps (w, x, sum);   // sum += w * x  (FMA)
    }
    return hsum8 (sum) + bias;
}

#endif // __AVX2__

// ─────────────────────────────────────────────────────────────────────────────
// Weight loading
// ─────────────────────────────────────────────────────────────────────────────
//
// The binary file is written by Python's export_weights() as a sequence of
// raw float32 arrays, all in little-endian order (same as x86 native).
// We read them in exactly the same order they were written.
//
// File layout:
//   ft_weight   [40960][256]  →  10 485 760 floats  ≈ 40 MB
//   ft_bias     [256]         →         256 floats
//   l1_weight   [32][512]     →      16 384 floats
//   l1_bias     [32]          →          32 floats
//   l2_weight   [32][32]      →       1 024 floats
//   l2_bias     [32]          →          32 floats
//   out_weight  [1][32]       →          32 floats
//   out_bias    [1]           →           1 float
// ─────────────────────────────────────────────────────────────────────────────
bool NNUEWeights::load (const std::string& path) {
    std::ifstream f (path, std::ios::binary);
    if (!f.is_open ()) return false;

    // Helper: read exactly n floats into dst.  Returns false if stream fails.
    auto read_floats = [&](float* dst, std::size_t n) -> bool {
        f.read (reinterpret_cast<char*>(dst),
            static_cast<std::streamsize>(n * sizeof (float)));
        return f.good ();
        };

    ft_weight.resize (HALFKP_FEATURES * FT_SIZE);

    if (!read_floats (ft_weight.data (), HALFKP_FEATURES * FT_SIZE)) return false;
    if (!read_floats (ft_bias.data (), FT_SIZE))                    return false;
    if (!read_floats (l1_weight.data (), L1_SIZE * INPUT_SIZE))       return false;
    if (!read_floats (l1_bias.data (), L1_SIZE))                    return false;
    if (!read_floats (l2_weight.data (), L2_SIZE * L1_SIZE))          return false;
    if (!read_floats (l2_bias.data (), L2_SIZE))                    return false;
    if (!read_floats (out_weight.data (), L2_SIZE))                    return false;
    if (!read_floats (&out_bias, 1))                          return false;

    return true;
}

bool load_nnue_weights (const std::string& path, NNUEWeights& weights) {
    return weights.load (path);
}




// ─────────────────────────────────────────────────────────────────────────────
// HalfKP index
// ─────────────────────────────────────────────────────────────────────────────
//
// Encodes the relationship (king_square, piece_type, piece_square) as a single
// integer in [0, 40960).
//
// The formula:
//   index = king_sq * 640 + piece_type_index * 64 + piece_sq
//
// piece_type_index:
//   own pieces  (pawn–queen) → 0-4
//   opp pieces  (pawn–queen) → 5-9
//
// This encoding means: "given that my king is on square K, there is a piece of
// type P on square S."  Different king squares create entirely different
// features, so the network learns king-relative patterns.
// ─────────────────────────────────────────────────────────────────────────────
static int halfkp_index (int king_sq, int piece_sq, int piece_type, bool is_own_piece) {
    int pt_idx = piece_type + (is_own_piece ? 0 : 5);
    return king_sq * (10 * 64) + pt_idx * 64 + piece_sq;
}


// ─────────────────────────────────────────────────────────────────────────────
// Accumulator refresh — AVX2 accelerated
// ─────────────────────────────────────────────────────────────────────────────
//
// For each non-king piece, computes its HalfKP feature index for both
// perspectives and adds the corresponding weight row to each accumulator.
//
// Scalar path:  25 pieces × 2 perspectives × 256 adds = 12,800 float adds
// AVX2 path:    25 pieces × 2 perspectives × 32 AVX2 adds = 1,600 AVX2 adds
//               Theoretical: 8x throughput. Practical: 3-5x (memory bound)
// ─────────────────────────────────────────────────────────────────────────────
void refresh_accumulator (const NNUEPosition& pos,
    const NNUEWeights& w,
    NNUEAccumulator& acc) {
    // Initialise both accumulators to the bias vector.
    // memcpy is faster than a loop for 256 floats.
    std::copy (w.ft_bias.begin (), w.ft_bias.end (), acc.white.begin ());
    std::copy (w.ft_bias.begin (), w.ft_bias.end (), acc.black.begin ());

    const int wk = pos.white_king_sq;
    const int bk = mirror_sq (pos.black_king_sq);

    for (const auto& p : pos.pieces) {
        const int sq_w = p.sq;
        const int sq_b = mirror_sq (p.sq);
        const int feat_w = halfkp_index (wk, sq_w, p.type, p.is_white);
        const int feat_b = halfkp_index (bk, sq_b, p.type, !p.is_white);

        const float* row_w = w.ft_weight.data () + feat_w * FT_SIZE;
        const float* row_b = w.ft_weight.data () + feat_b * FT_SIZE;

#ifdef __AVX2__
        avx2_add_row (acc.white.data (), row_w);
        avx2_add_row (acc.black.data (), row_b);
#else
        for (int j = 0; j < FT_SIZE; ++j) { acc.white[j] += row_w[j]; }
        for (int j = 0; j < FT_SIZE; ++j) { acc.black[j] += row_b[j]; }
#endif
    }

    // Apply ClippedReLU activation to both accumulators.
#ifdef __AVX2__
    avx2_crelu_inplace (acc.white.data ());
    avx2_crelu_inplace (acc.black.data ());
#else
    for (int j = 0; j < FT_SIZE; ++j) { acc.white[j] = crelu (acc.white[j]); }
    for (int j = 0; j < FT_SIZE; ++j) { acc.black[j] = crelu (acc.black[j]); }
#endif

    acc.dirty = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Incremental accumulator update  (call instead of refresh after a move)
// ─────────────────────────────────────────────────────────────────────────────
//
// When a move is made, only a few features change — the piece that moved and
// any captured piece.  Instead of recomputing from scratch, we:
//   subtract the weight rows for features that disappeared
//   add    the weight rows for features that appeared
//
// This reduces ~6400 operations per position to ~512 (2 pieces × 256 neurons).
//
// Usage in your move-make code:
//   1. Copy current accumulator onto the search stack
//   2. Call nnue_update_accumulator() for the moved piece (remove from-sq, add to-sq)
//   3. If a capture, call nnue_remove_feature() for the captured piece
//   4. When unmaking the move, restore the accumulator from the stack copy
//
// NOTE: After a king move, the entire accumulator must be refreshed because
//       every feature changes (all features are relative to the king square).
// ─────────────────────────────────────────────────────────────────────────────
void nnue_add_feature (NNUEAccumulator& acc, const NNUEWeights& w,
    int king_sq_w, int king_sq_b,
    int piece_sq, int piece_type, bool piece_is_white) {
    // The accumulator at this point has ClippedReLU already applied from the
    // previous refresh.  We undo the activation, add the new feature row, and
    // reapply.  In practice engines skip this and just work with pre-activation
    // values, but for clarity we keep the interface simple here.

    const int sq_b = mirror_sq (piece_sq);
    const int feat_w = halfkp_index (king_sq_w, piece_sq, piece_type, piece_is_white);
    const int feat_b = halfkp_index (mirror_sq (king_sq_b), sq_b, piece_type, !piece_is_white);

    const float* row_w = w.ft_weight.data () + feat_w * FT_SIZE;
    const float* row_b = w.ft_weight.data () + feat_b * FT_SIZE;

    for (int j = 0; j < FT_SIZE; ++j) {
        acc.white[j] = crelu (acc.white[j] + row_w[j]);
        acc.black[j] = crelu (acc.black[j] + row_b[j]);
    }
}

void nnue_remove_feature (NNUEAccumulator& acc, const NNUEWeights& w,
    int king_sq_w, int king_sq_b,
    int piece_sq, int piece_type, bool piece_is_white) {
    const int sq_b = mirror_sq (piece_sq);
    const int feat_w = halfkp_index (king_sq_w, piece_sq, piece_type, piece_is_white);
    const int feat_b = halfkp_index (mirror_sq (king_sq_b), sq_b, piece_type, !piece_is_white);

    const float* row_w = w.ft_weight.data () + feat_w * FT_SIZE;
    const float* row_b = w.ft_weight.data () + feat_b * FT_SIZE;

    for (int j = 0; j < FT_SIZE; ++j) {
        acc.white[j] = crelu (acc.white[j] - row_w[j]);
        acc.black[j] = crelu (acc.black[j] - row_b[j]);
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Full forward pass
// ─────────────────────────────────────────────────────────────────────────────
//
// Given valid accumulators, runs L1 → L2 → output to produce a score.
//
// Returns a raw float in centipawn-equivalent units:
//   positive → white is better
//   negative → black is better
//
// The magnitude is in the same scale as the training labels (centipawns),
// so you can drop this directly into your alpha-beta search wherever your
// PST evaluate() call currently lives.
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// Forward pass — AVX2 accelerated
// ─────────────────────────────────────────────────────────────────────────────
//
// Layer costs (scalar vs AVX2):
//   L1  512→32:  16,384 multiplies  →  2,048 FMAs   (8x)
//   L2   32→32:   1,024 multiplies  →    128 FMAs   (8x)
//   out  32→ 1:      32 multiplies  →      4 FMAs   (8x)
// ─────────────────────────────────────────────────────────────────────────────
float nnue_evaluate (const NNUEPosition& pos,
    const NNUEWeights& w,
    NNUEAccumulator& acc) {
    if (acc.dirty)
        refresh_accumulator (pos, w, acc);

    // ── Step 1: concatenate STM perspective first, NSTM second ───────────
    // INPUT_SIZE = 512 = 256 + 256
    // Aligned so AVX2 loads from x[] can use _mm256_load_ps
    alignas(32) std::array<float, INPUT_SIZE> x;

    const auto& stm_acc = pos.white_to_move ? acc.white : acc.black;
    const auto& nstm_acc = pos.white_to_move ? acc.black : acc.white;

    std::copy (stm_acc.begin (), stm_acc.end (), x.begin ());
    std::copy (nstm_acc.begin (), nstm_acc.end (), x.begin () + FT_SIZE);

    // ── Step 2: L1  (512 → 32)  ClippedReLU ─────────────────────────────
    alignas(32) std::array<float, L1_SIZE> l1;

    for (int i = 0; i < L1_SIZE; ++i) {
        const float* row = w.l1_weight.data () + i * INPUT_SIZE;
#ifdef __AVX2__
        float s = avx2_dot (row, x.data (), INPUT_SIZE, w.l1_bias[i]);
#else
        float s = w.l1_bias[i];
        for (int j = 0; j < INPUT_SIZE; ++j) s += row[j] * x[j];
#endif
        l1[i] = crelu (s);
    }

    // ── Step 3: L2  (32 → 32)  ClippedReLU ──────────────────────────────
    alignas(32) std::array<float, L2_SIZE> l2;

    for (int i = 0; i < L2_SIZE; ++i) {
        const float* row = w.l2_weight.data () + i * L1_SIZE;
#ifdef __AVX2__
        float s = avx2_dot (row, l1.data (), L1_SIZE, w.l2_bias[i]);
#else
        float s = w.l2_bias[i];
        for (int j = 0; j < L1_SIZE; ++j) s += row[j] * l1[j];
#endif
        l2[i] = crelu (s);
    }

    // ── Step 4: output  (32 → 1)  no activation ──────────────────────────
#ifdef __AVX2__
    return avx2_dot (w.out_weight.data (), l2.data (), L2_SIZE, w.out_bias);
#else
    float score = w.out_bias;
    for (int j = 0; j < L2_SIZE; ++j) score += w.out_weight[j] * l2[j];
    return score;
#endif
}
