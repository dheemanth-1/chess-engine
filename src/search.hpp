#ifndef SEARCH_HPP
#define SEARCH_HPP

#include "chess.hpp"
#include <utility> // For std::pair

namespace SearchConstants {
    // A value higher than any possible board evaluation
    const int INF = 1000000;
    // Value for checkmate
    const int MATE = 900000;
}

/**
 * Searches the move tree using Alpha-Beta pruning to find the best move.
 * @param board The current chess board state.
 * @param depth How many half-moves (plies) to look ahead.
 * @return A pair containing the best Move and its evaluation score.
 */
std::pair<chess::Move, int> get_best_move (chess::Board& board, int depth);

/**
 * The core recursive NegaMax function with Alpha-Beta pruning.
 * @param alpha The lower bound of the search window.
 * @param beta The upper bound of the search window.
 */
int alphabeta (chess::Board& board, int depth, int alpha, int beta);

#endif