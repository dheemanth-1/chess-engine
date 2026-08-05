#include "chess.hpp"
#include <array>

// Material Values (Centipawns)
const int PAWN_VAL = 100;
const int KNIGHT_VAL = 320;
const int BISHOP_VAL = 330;
const int ROOK_VAL = 500;
const int QUEEN_VAL = 900;
const int KING_VAL = 20000;

const std::array<int, 64> pawn_pst = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};

// Piece-Square Table 
// Higher values = better squares
const std::array<int, 64> knight_pst = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
};

// Bishops: Reward long diagonals and center control
const std::array<int, 64> bishop_pst = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
};

// Rooks: Neutral, but slightly prefer the 7th rank (row 1 in this array)
const std::array<int, 64> rook_pst = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0
};

// Queen: Keep it central but safe
const std::array<int, 64> queen_pst = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
};

// King: Encourages castling and staying in the corner for the middlegame
const std::array<int, 64> king_pst = {
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
};

// King Endgame PST: High values in the center, penalized in the corners
const std::array<int, 64> king_endgame_pst = {
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
};

constexpr int BONUS_CASTLED = 60;   // High incentive to get the king safe
constexpr int PENALTY_KING_WALKED = -70;  // Punishment for losing castling rights manually
constexpr int BONUS_MINOR_DEVELOPED = 20;   // Reward per minor piece leaving its home square
constexpr int PENALTY_EARLY_QUEEN = -50;  // Slap the wrist for aggressive early Queen lunges
constexpr int BONUS_CONNECTED_ROOKS = 25;

const std::array<int, 8> passed_pawn_bonus = { 0, 10, 20, 40, 80, 150, 300, 0 };

bool areWhiteRooksConnected (const chess::Board& board) {
    int rookCount = 0;
    int firstRookFile = -1;
    int secondRookFile = -1;

    // Scan Rank 1 (Files A through H)
    for (int file = 0; file < 8; ++file) {
        // Creates a square using (file, rank) indices [0-7]
        chess::Square sq = chess::Square (static_cast<chess::File> (file), static_cast<chess::Rank> (0));
        if (board.at (sq) == chess::Piece::WHITEROOK) {
            if (rookCount == 0) firstRookFile = file;
            else if (rookCount == 1) secondRookFile = file;
            rookCount++;
        }
    }

    if (rookCount != 2) return false;

    // Verify all squares between the two rooks are empty
    for (int file = firstRookFile + 1; file < secondRookFile; ++file) {
        if (board.at (chess::Square (static_cast<chess::File> (file), static_cast<chess::Rank> (0))) != chess::Piece::NONE) {
            return false;
        }
    }
    return true;
}

bool areBlackRooksConnected (const chess::Board& board) {
    int rookCount = 0;
    int firstRookFile = -1;
    int secondRookFile = -1;

    // Scan Rank 8 (Rank index 7)
    for (int file = 0; file < 8; ++file) {
        chess::Square sq = chess::Square (static_cast<chess::File> (file), static_cast<chess::Rank> (7));
        if (board.at (sq) == chess::Piece::BLACKROOK) {
            if (rookCount == 0) firstRookFile = file;
            else if (rookCount == 1) secondRookFile = file;
            rookCount++;
        }
    }

    if (rookCount != 2) return false;

    for (int file = firstRookFile + 1; file < secondRookFile; ++file) {
        if (board.at (chess::Square (static_cast<chess::File> (file), static_cast<chess::Rank> (7))) != chess::Piece::NONE) {
            return false;
        }
    }
    return true;
}

bool is_passed_pawn (const chess::Board& board, int sq, chess::Color color) {
    int file = sq % 8;
    int rank = sq / 8;

    // Create a mask of the files we care about (current, left, right)
    uint64_t file_mask = 0;
    for (int f = std::max (0, file - 1); f <= std::min (7, file + 1); ++f) {
        file_mask |= (0x0101010101010101ULL << f);
    }

    // Forward mask depends on color
    uint64_t forward_mask = 0;
    if (color == chess::Color::WHITE) {
        // Squares above the current rank
        forward_mask = ~((1ULL << ((rank + 1) * 8)) - 1);
    }
    else {
        // Squares below the current rank
        forward_mask = (1ULL << (rank * 8)) - 1;
    }

    // Get enemy pawns in front of us on adjacent or same files
    chess::Color enemy_color = (color == chess::Color::WHITE) ? chess::Color::BLACK : chess::Color::WHITE;
    uint64_t enemy_pawns = board.pieces (chess::PieceType::PAWN, enemy_color).getBits ();

    return (enemy_pawns & file_mask & forward_mask) == 0;
}

// Struct to hold split scores
struct Score {
    int mg = 0; // Middlegame score
    int eg = 0; // Endgame score
};

Score evaluate_side (const chess::Board& board, chess::Color color) {
    Score score;
    bool is_white = (color == chess::Color::WHITE);

    auto process_piece = [&](chess::PieceType type, int value, const std::array<int, 64>& pst) {
        chess::Bitboard bb = board.pieces (type, color);
        while (bb) {
            int sq = static_cast<int>(bb.pop ());
            int pst_sq = is_white ? sq : (sq ^ 56);
            int total_val = value + pst[pst_sq];
            score.mg += total_val;
            score.eg += total_val;
        }
        };

    process_piece (chess::PieceType::KNIGHT, KNIGHT_VAL, knight_pst);
    process_piece (chess::PieceType::BISHOP, BISHOP_VAL, bishop_pst);
    process_piece (chess::PieceType::ROOK, ROOK_VAL, rook_pst);
    process_piece (chess::PieceType::QUEEN, QUEEN_VAL, queen_pst);

    chess::Bitboard pawns = board.pieces (chess::PieceType::PAWN, color);
    while (pawns) {
        int sq = static_cast<int>(pawns.pop ());
        int pst_sq = is_white ? sq : (sq ^ 56);
        int rank = pst_sq / 8; // Normalized rank (0-7 from perspective)

        int pawn_value = PAWN_VAL + pawn_pst[pst_sq];
        score.mg += pawn_value;
        score.eg += pawn_value;

        // Check for passed pawn
        if (is_passed_pawn (board, sq, color)) {
            int bonus = passed_pawn_bonus[rank];
            score.mg += bonus / 2; // Passed pawns are useful in middlegame
            score.eg += bonus;     // Passed pawns are lethal in endgame
        }
    }


    chess::Bitboard king = board.pieces (chess::PieceType::KING, color);
    if (king) {
        int sq = static_cast<int>(king.pop ());
        int pst_sq = is_white ? sq : (sq ^ 56);

        score.mg += KING_VAL + king_pst[pst_sq];
        score.eg += KING_VAL + king_endgame_pst[pst_sq];

        // pst_sq == 6 (G1/G8) or 2 (C1/C8) means the King has successfully castled
        if (pst_sq == 6 || pst_sq == 2) {
            score.mg += 30; // Castling bonus in the middlegame
        }
        else if (pst_sq == 4) {
            score.mg -= 20; // Penalty for leaving the King exposed in the center
        }
    }
    int undeveloped_minors = 0;

    // Check if knights are still on starting squares (B1/G1 for White, B8/G8 for Black)
    chess::Bitboard temp_knights = board.pieces (chess::PieceType::KNIGHT, color);
    while (temp_knights) {
        int sq = static_cast<int>(temp_knights.pop ());
        if (is_white && (sq == 1 || sq == 6)) undeveloped_minors++;
        if (!is_white && (sq == 57 || sq == 62)) undeveloped_minors++;
    }

    // Check if bishops are still on starting squares (C1/F1 for White, C8/F8 for Black)
    chess::Bitboard temp_bishops = board.pieces (chess::PieceType::BISHOP, color);
    while (temp_bishops) {
        int sq = static_cast<int>(temp_bishops.pop ());
        if (is_white && (sq == 2 || sq == 5)) undeveloped_minors++;
        if (!is_white && (sq == 58 || sq == 61)) undeveloped_minors++;
    }

    // Apply development penalty to middlegame score
    score.mg -= undeveloped_minors * 10;

    // Punish early queen attacks if at least 2 minor pieces are still trapped back home
    chess::Bitboard temp_queen = board.pieces (chess::PieceType::QUEEN, color);
    if (temp_queen) {
        int q_sq = static_cast<int>(temp_queen.pop ());
        int q_pst_sq = is_white ? q_sq : (q_sq ^ 56);
        // Normalized index 3 represents D1/D8. If it moved early:
        if (q_pst_sq != 3 && undeveloped_minors >= 2) {
            score.mg -= 25;
        }
    }

    // Rook Connectivity Evaluation 
    chess::Bitboard rooks = board.pieces (chess::PieceType::ROOK, color);
    if (rooks.count () == 2) {
        int r1 = static_cast<int>(rooks.pop ());
        int r2 = static_cast<int>(rooks.pop ());
        // Give a bonus if they protect each other or coordinate on the same file/rank
        if ((r1 % 8 == r2 % 8) || (r1 / 8 == r2 / 8)) {
            score.mg += 15;
            score.eg += 15;
        }
    }

    return score;
}

int pst_evaluate (const chess::Board& board) {
    // Determine Game Phase
    int phase = 0;
    phase += board.pieces (chess::PieceType::KNIGHT).count () * 1;
    phase += board.pieces (chess::PieceType::BISHOP).count () * 1;
    phase += board.pieces (chess::PieceType::ROOK).count () * 2;
    phase += board.pieces (chess::PieceType::QUEEN).count () * 4;

    // Ensure phase stays bound between 0 and 24
    phase = std::min (phase, 24);

    // Evaluate both sides
    Score white_score = evaluate_side (board, chess::Color::WHITE);
    Score black_score = evaluate_side (board, chess::Color::BLACK);

    // Compute relative differences
    int mg_eval = white_score.mg - black_score.mg;
    int eg_eval = white_score.eg - black_score.eg;

    // Linear Interpolation (Tapering formula)
    int total_eval = ((mg_eval * phase) + (eg_eval * (24 - phase))) / 24;

    // Return perspective score for Negamax
    return (board.sideToMove () == chess::Color::WHITE) ? total_eval : -total_eval;
}