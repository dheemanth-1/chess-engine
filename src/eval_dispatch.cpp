// eval_dispatch.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Implementation of the evaluation dispatcher.
// Compile this alongside search.cpp, main.cpp, nnue.cpp, evaluation.cpp.
// ─────────────────────────────────────────────────────────────────────────────

#include "eval_dispatch.h"
#include <iostream>


// ── Global state ──────────────────────────────────────────────────────────────
bool        g_use_nnue = false;
NNUEWeights g_nnue_weights;
std::string g_weights_path = "nnue_weights.bin";

bool init_nnue (const std::string& path) {
    if (!load_nnue_weights (path, g_nnue_weights)) {
        std::cerr << "[NNUE] Failed to load weights from: " << path << "\n";
        std::cerr << "[NNUE] Falling back to PST evaluation.\n";
        g_use_nnue = false;
        return false;
    }
    g_use_nnue = true;
    g_weights_path = path;
    std::cerr << "[NNUE] Weights loaded from: " << path << "\n";
    return true;
}


// ── board_to_nnue_position ────────────────────────────────────────────────────
//
// Piece type mapping — Disservin chess.hpp enum values happen to match the
// NNUE convention exactly:
//   chess::PieceType::PAWN   = 0   →  NNUE type 0
//   chess::PieceType::KNIGHT = 1   →  NNUE type 1
//   chess::PieceType::BISHOP = 2   →  NNUE type 2
//   chess::PieceType::ROOK   = 3   →  NNUE type 3
//   chess::PieceType::QUEEN  = 4   →  NNUE type 4
//   chess::PieceType::KING   = 5   →  skipped (kings are encoded separately)
//
// So static_cast<int>(piece.type()) gives the right NNUE index directly.
// No lookup table needed.
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// board_to_nnue_position — bitboard version
// ─────────────────────────────────────────────────────────────────────────────
//
// Previous version: scanned all 64 squares with board.at(sq) on each.
//   board.at() does a hash-map lookup internally — 64 calls per evaluation.
//
// This version: iterates the occupied-squares bitboard directly.
//   chess::Bitboard::pop() removes and returns the LSB square in one
//   instruction (hardware TZCNT/BSF). For a typical middlegame position
//   with ~28 pieces, this is 28 iterations instead of 64, and each
//   iteration is cheaper than a hash-map lookup.
//
// The piece type mapping exploits a happy coincidence: Disservin's
// chess.hpp PieceType enum values happen to match the NNUE convention:
//   PAWN=0  KNIGHT=1  BISHOP=2  ROOK=3  QUEEN=4  KING=5
// So static_cast<int>(piece.type()) gives the right NNUE index directly.
// ─────────────────────────────────────────────────────────────────────────────
NNUEPosition board_to_nnue_position (const chess::Board& board) {
    NNUEPosition pos;
    pos.white_king_sq = board.kingSq (chess::Color::WHITE).index ();
    pos.black_king_sq = board.kingSq (chess::Color::BLACK).index ();
    pos.white_to_move = (board.sideToMove () == chess::Color::WHITE);

    // Reserve for typical piece count — avoids reallocation mid-loop
    pos.pieces.reserve (30);

    // Iterate only occupied squares (typically 28-32 in middlegame)
    // instead of all 64 squares
    chess::Bitboard occ = board.occ ();
    while (occ) {
        chess::Square sq = occ.pop ();   // BSF: one CPU instruction
        chess::Piece  piece = board.at (sq);

        if (piece.type () == chess::PieceType::KING) continue;

        pos.pieces.push_back ({
            sq.index (),
            static_cast<int>(piece.type ()),
            piece.color () == chess::Color::WHITE
            });
    }

    return pos;
}


// ── evaluate ──────────────────────────────────────────────────────────────────
//
// This is the function your search calls at leaf nodes.
// It replaces the direct call to pst_evaluate() that was there before.
//
// The NNUE accumulator is created fresh on every call here (dirty=true by
// default, so refresh_accumulator runs automatically inside nnue_evaluate).
// This is the correct-but-not-incremental path. It works, and the MX450
// is fast enough at inference that this won't be your search bottleneck.
//
// When you want to add incremental updates later (for a speed boost), that
// change happens here — you'd pass an accumulator from the search stack
// instead of creating a local one.
// ─────────────────────────────────────────────────────────────────────────────
int evaluate (const chess::Board& board) {
    if (!g_use_nnue)
        return pst_evaluate (board);

    NNUEPosition    pos = board_to_nnue_position (board);
    NNUEAccumulator acc;
    float raw = nnue_evaluate (pos, g_nnue_weights, acc);
    return static_cast<int>(raw);
}