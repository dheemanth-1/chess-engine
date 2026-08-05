#include "evaluation.hpp"
#include <limits>
#include <vector>
#include <cstdint>
#include "eval_dispatch.h"

enum class TTFlag { EXACT, ALPHA, BETA };

struct TTEntry {
    uint64_t hash = 0;
    int depth = -1;
    int score = 0;
    TTFlag flag = TTFlag::EXACT;
    chess::Move best_move = chess::Move::NO_MOVE; // Default null move
};

class TranspositionTable {
    private:
    std::vector<TTEntry> table;
    size_t table_size;

    public:
    // Initialize table size (e.g., 2^20 elements ~ approx 32MB)
    TranspositionTable (size_t size = 1048576) : table_size (size) {
        table.resize (table_size);
    }

    void clear () {
        std::fill (table.begin (), table.end (), TTEntry{});
    }

    bool lookup (uint64_t hash, int depth, int alpha, int beta, int& tt_score, chess::Move& tt_move) {
        size_t index = hash % table_size;
        const TTEntry& entry = table[index];

        if (entry.hash == hash) {
            tt_move = entry.best_move; // Always grab the best move for ordering

            if (entry.depth >= depth) {
                if (entry.flag == TTFlag::EXACT) {
                    tt_score = entry.score;
                    return true;
                }
                if (entry.flag == TTFlag::ALPHA && entry.score <= alpha) {
                    tt_score = alpha;
                    return true;
                }
                if (entry.flag == TTFlag::BETA && entry.score >= beta) {
                    tt_score = beta;
                    return true;
                }
            }
        }
        return false;
    }

    void store (uint64_t hash, int depth, int score, TTFlag flag, chess::Move best_move) {
        size_t index = hash % table_size;
        // Always replace strategy or deeper depth replacement strategy
        if (table[index].hash != hash || depth >= table[index].depth) {
            table[index] = { hash, depth, score, flag, best_move };
        }
    }
};



int get_piece_value (chess::PieceType type) {
    switch (static_cast<int>(type)) {
    case static_cast<int>(chess::PieceType::PAWN):   return 100;
    case static_cast<int>(chess::PieceType::KNIGHT): return 300;
    case static_cast<int>(chess::PieceType::BISHOP): return 300;
    case static_cast<int>(chess::PieceType::ROOK):   return 500;
    case static_cast<int>(chess::PieceType::QUEEN):  return 900;
    default: return 0;
    }
}

// Pair up moves with an integer score for sorting
struct ScoredMove {
    chess::Move move;
    int score;
};

// Main scoring engine
int score_single_move (const chess::Board& board, chess::Move move, chess::Move tt_move, chess::Move killer_move) {
    // 1. TT Move gets absolute priority
    if (move == tt_move) return 10000000;

    // 2. MVV-LVA for Captures
    if (board.isCapture (move)) {
        chess::PieceType attacker = board.at (move.from ()).type ();
        // Handle normal captures vs en-passant safely
        chess::PieceType victim = chess::PieceType::PAWN;
        if (board.at (move.to ()) != chess::Piece::NONE) {
            victim = board.at (move.to ()).type ();
        }

        return 1000000 + (get_piece_value (victim) * 10) - get_piece_value (attacker);
    }

    // 3. Killer Move (Quiet move that previously caused a beta cutoff at this depth)
    if (move == killer_move) return 50000;

    // 4. Quiet moves get base value
    return 0;
}

// Sort function to execute before your loops
void order_moves (const chess::Board& board, chess::Movelist& moves, chess::Move tt_move, chess::Move killer_move) {
    std::vector<ScoredMove> scored_moves;
    scored_moves.reserve (moves.size ());

    for (const auto& move : moves) {
        int score = score_single_move (board, move, tt_move, killer_move);
        scored_moves.push_back ({ move, score });
    }

    // Sort descending based on score
    std::sort (scored_moves.begin (), scored_moves.end (), [](const ScoredMove& a, const ScoredMove& b) {
        return a.score > b.score;
        });

    // Write back ordered moves into standard movelist container
    for (chess::Movelist::size_type i = 0; i < scored_moves.size (); ++i) {
        moves[i] = scored_moves[i].move;
    }
}


// ─── Global move arrays — the key fix ────────────────────────────────────────
//
// PROBLEM: Declaring chess::Movelist as a local variable inside alphabeta() and
// quiescence_search() puts a ~1 KB array on the call stack for every recursive
// call. At depth 12 + 32 QS plies = 44 stack frames, each holding a Movelist,
// that is ~44 KB in release builds — manageable. But AddressSanitizer doubles
// effective frame sizes with its shadow/red-zone instrumentation, pushing it
// past the default 1 MB Windows stack limit and causing the stack-overflow crash
// you observed.
//
// FIX: Pre-allocate Movelists as globals, indexed by ply. Each recursive call
// uses the slot for its own ply, so there is no aliasing:
//   alphabeta at ply N   → g_ab_moves[N]   (N in [0,127])
//   quiescence at qply Q → g_qs_moves[Q]   (Q in [0, MAX_QUIESCENCE_PLY-1])
//
// The invariant that makes this safe: alphabeta at ply N calls alphabeta at
// ply N+1, which only ever touches g_ab_moves[N+1] — never g_ab_moves[N].
// Once the child returns, g_ab_moves[N] is still intact for the parent's loop
// to continue. This is the standard technique used in every production engine.
//
// Each array has 128 slots so the index never goes out of bounds even under
// unusual conditions (we clamp with std::min before using as index).
//
// Stack frame size drops from ~1 KB to ~40 bytes per call → problem eliminated.
// ─────────────────────────────────────────────────────────────────────────────
static chess::Movelist g_ab_moves[128];    // alphabeta, one slot per ply
static chess::Movelist g_qs_moves[128];    // quiescence search, one slot per qply
chess::Movelist g_qs_scratch[128];
constexpr int MAX_QUIESCENCE_PLY = 32;
const     int INF = 1000000;

TranspositionTable TT (1048576);
chess::Move        killer_moves[64];

int quiescence_search (chess::Board& board, int alpha, int beta, int qply = 0) {
    //  Hard Safety Ceiling to prevent infinite search loops
    if (qply >= MAX_QUIESCENCE_PLY)
        return evaluate (board);

    //  Standing Pat Evaluation acting as a baseline score
    int stand_pat = evaluate (board);
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    //  Memory Optimization: Establish global index tracking for this depth
    int sq = std::min (qply, 127);
    g_qs_moves[sq].clear ();

    //  Combined Move Generation Logic
    if (qply < 2) {
        // Use a global scratchpad to avoid placing a full Movelist container on the stack
        g_qs_scratch[sq].clear ();
        chess::movegen::legalmoves<chess::movegen::MoveGenType::ALL> (g_qs_scratch[sq], board);

        for (int i = 0; i < static_cast<int> (g_qs_scratch[sq].size ()); ++i) {
            chess::Move m = g_qs_scratch[sq][i];

            // Always include standard captures
            if (board.isCapture (m)) {
                g_qs_moves[sq].add (m);
            }
            // For quiet moves, test if they give a tactical check before accepting them
            else {
                board.makeMove (m);
                bool gives_check = board.inCheck ();
                board.unmakeMove (m);

                if (gives_check) {
                    g_qs_moves[sq].add (m);
                }
            }
        }
    }
    else {
        // Deep in the Q-search tree, generate ONLY captures to prevent endless explosions
        chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE> (g_qs_moves[sq], board);
    }

    // Move Ordering
    order_moves (board, g_qs_moves[sq], chess::Move::NO_MOVE, chess::Move::NO_MOVE);

    //  Search Loop
    for (int i = 0; i < static_cast<int> (g_qs_moves[sq].size ()); ++i) {
        chess::Move move = g_qs_moves[sq][i]; // Copy move by value before recursion

        //  Optimized Delta Pruning Guard
        // CRITICAL FIX: We must NOT delta prune if the move is a quiet check, 
        // because checks can easily force a win regardless of immediate material standing.
        if (!board.inCheck () && board.isCapture (move)) {
            chess::PieceType victim = (board.at (move.to ()) != chess::Piece::NONE)
                ? board.at (move.to ()).type () : chess::PieceType::PAWN;

            if (stand_pat + get_piece_value (victim) + 200 < alpha)
                continue;
        }

        // Recursive Execution
        board.makeMove (move);
        int score = -quiescence_search (board, -beta, -alpha, qply + 1);
        board.unmakeMove (move);

        // Alpha-Beta Cutoffs
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }

    return alpha;
}

// if (qply < 2) {
//     chess::Movelist all_moves;
//     chess::movegen::legalmoves<chess::movegen::MoveGenType::ALL> (all_moves, board);
//     for (const auto& m : all_moves) {
//         board.makeMove (m);
//         bool gives_check = board.inCheck ();
//         board.unmakeMove (m);
//         if (gives_check && !board.isCapture (m)) {
//             moves.add (m);
//         }
//     }
// }

// // Score and order captures using the exact same strategy
// order_moves (board, moves, chess::Move::NO_MOVE, chess::Move::NO_MOVE);

// for (const auto& move : moves) {
//     // Delta Pruning (Optional optimization: if capture can't even raise alpha, skip it)
//     if (!board.inCheck () && !board.isCapture (move)) {
//         chess::PieceType victim = (board.at (move.to ()) != chess::Piece::NONE) ? board.at (move.to ()).type () : chess::PieceType::PAWN;
//         if (stand_pat + get_piece_value (victim) + 200 < alpha) {
//             continue;
//         }
//     }

//     board.makeMove (move);
//     int score = -quiescence_search (board, -beta, -alpha);
//     board.unmakeMove (move);

//     if (score >= beta) return beta;
//     if (score > alpha) alpha = score;
// }

// return alpha;
//}

void new_game_reset () {
    TT.clear ();
    std::fill (std::begin (killer_moves), std::end (killer_moves), chess::Move::NO_MOVE);
}

int alphabeta (chess::Board& board, int depth, int alpha, int beta, int ply) {
    uint64_t    hash = board.hash ();
    chess::Move tt_move = chess::Move::NO_MOVE;
    int         tt_score = 0;


    if (TT.lookup (hash, depth, alpha, beta, tt_score, tt_move))
        return tt_score;


    // If the king is in check, we extend the search depth by 1 to avoid missing tactical refutations.
    bool in_check = board.inCheck ();
    int extension = (in_check && ply < 32) ? 1 : 0;

    //  Base Case: Drop into Quiescence Search when depth is fully exhausted
    if (depth <= 0)
        return quiescence_search (board, alpha, beta);

    // Memory Optimization: Use global slot for this ply (No stack allocations)
    int sp = std::min (ply, 127);
    g_ab_moves[sp].clear ();
    chess::movegen::legalmoves<chess::movegen::MoveGenType::ALL> (g_ab_moves[sp], board);

    // Terminal Node Handling (Checkmate / Stalemate)
    if (g_ab_moves[sp].empty ())
        return board.inCheck () ? -INF + ply : 0;

    // Move Ordering with Safe Array Indexing
    int killer_idx = std::min (ply, 63);
    order_moves (board, g_ab_moves[sp], tt_move, killer_moves[killer_idx]);

    chess::Move best_current_move = chess::Move::NO_MOVE;
    TTFlag      flag = TTFlag::ALPHA;

    // Search Loop
    for (int i = 0; i < static_cast<int> (g_ab_moves[sp].size ()); ++i) {
        chess::Move move = g_ab_moves[sp][i]; // Copy by value before making the move

        board.makeMove (move);
        // We pass the extension down here. If extended, depth effectively stays the same for this step.
        int score = -alphabeta (board, depth - 1 + extension, -beta, -alpha, ply + 1);
        board.unmakeMove (move);

        // Beta Cutoff
        if (score >= beta) {
            if (!board.isCapture (move))
                killer_moves[killer_idx] = move;
            TT.store (hash, depth, beta, TTFlag::BETA, move);
            return beta;
        }
        // Alpha Improvement
        if (score > alpha) {
            alpha = score;
            flag = TTFlag::EXACT;
            best_current_move = move;
        }
    }

    // Store Search Results in Transposition Table
    TT.store (hash, depth, alpha, flag, best_current_move);
    return alpha;
}


// int alphabeta (chess::Board& board, int depth, int alpha, int beta, int ply) {
//     uint64_t hash = board.hash ();
//     chess::Move tt_move = chess::Move::NO_MOVE;
//     int tt_score = 0;

//     // 1. TT Look Up
//     if (TT.lookup (hash, depth, alpha, beta, tt_score, tt_move)) {
//         return tt_score;
//     }

//     bool in_check = board.inCheck ();
//     int extension = (in_check && ply < 32) ? 1 : 0;
//     if (depth == 0 && extension == 0) {
//         return quiescence_search (board, alpha, beta);
//     }

//     chess::Movelist moves;
//     chess::movegen::legalmoves<chess::movegen::MoveGenType::ALL> (moves, board);

//     if (moves.empty ()) {
//         if (board.inCheck ()) {
//             return -INF + ply;
//         }
//         return 0;
//     }

//     order_moves (board, moves, tt_move, killer_moves[ply]);

//     chess::Move best_current_move = chess::Move::NO_MOVE;
//     TTFlag flag = TTFlag::ALPHA;

//     int best_score = -INF;

//     for (auto move : moves) {
//         board.makeMove (move);
//         int score = -alphabeta (board, depth - 1, -beta, -alpha, ply + 1);
//         board.unmakeMove (move);


//         if (score >= beta) {
//             if (!board.isCapture (move)) {
//                 killer_moves[ply] = move;
//             }
//             TT.store (hash, depth, beta, TTFlag::BETA, move);
//             return beta;
//         }
//         if (score > alpha) {
//             alpha = score;
//             flag = TTFlag::EXACT;
//             best_current_move = move;
//         }
//     }
//     TT.store (hash, depth, alpha, flag, best_current_move);
//     return alpha;
// }

std::pair<chess::Move, int> get_best_move (chess::Board& board, int depth) {
    chess::Movelist moves;
    chess::movegen::legalmoves (moves, board);

    if (moves.empty ()) {
        return { chess::Move (), board.inCheck () ? -INF : 0 };
    }

    chess::Move best_move = moves[0];
    int best_score = -INF;

    for (auto move : moves) {
        board.makeMove (move);
        int score = -alphabeta (board, depth - 1, -INF, INF, 1);
        board.unmakeMove (move);

        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
    }
    return { best_move, best_score };
}