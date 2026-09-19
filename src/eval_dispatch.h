#pragma once
// eval_dispatch.h
// ─────────────────────────────────────────────────────────────────────────────
// Provides a single evaluate(board) function that dispatches to either the
// PST evaluator or the NNUE evaluator based on a runtime flag.
//
// HOW TO USE IN YOUR SEARCH FILE:
//   Replace:  #include "evaluation.hpp"
//   With:     #include "eval_dispatch.h"
//
// That is the only change needed in search.cpp.
// The evaluate(board) call in quiescence_search stays exactly as-is.
// ─────────────────────────────────────────────────────────────────────────────

#include "nnue.h"
#include <chess.hpp>

// ── Forward declaration of the PST evaluator ─────────────────────────────────
// In your evaluation.hpp, rename the existing evaluate() to pst_evaluate().
// That is the only change needed in evaluation.hpp.
// The signature stays the same — just the name changes.
int pst_evaluate (const chess::Board& board);


// ── Global NNUE state ─────────────────────────────────────────────────────────
// Defined in eval_dispatch.cpp.
// Accessible from main.cpp via extern.
extern bool         g_use_nnue;
extern NNUEWeights  g_nnue_weights;
extern std::string  g_weights_path;
extern int g_multipv;
extern std::atomic<bool> g_stop_requested;


// ── Init function ─────────────────────────────────────────────────────────────
// Call this from main() when the --nnue flag is present.
// Returns true if weights loaded successfully, false otherwise.
bool init_nnue (const std::string& path = "nnue_weights.bin");


// ── Board → NNUEPosition converter ───────────────────────────────────────────
// Translates a chess::Board into the library-agnostic NNUEPosition struct.
// Uses Disservin's chess.hpp API — adapt if you use a different library.
NNUEPosition board_to_nnue_position (const chess::Board& board);


// ── The unified evaluate() ───────────────────────────────────────────────────
// This is what your search calls. It checks g_use_nnue and dispatches.
// Return value: centipawn score from the side-to-move's perspective.
//   positive = side to move is winning
//   negative = side to move is losing
// This convention matches your negamax search exactly.
int evaluate (const chess::Board& board);