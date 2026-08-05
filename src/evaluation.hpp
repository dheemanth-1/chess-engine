#ifndef EVALUATION_HPP
#define EVALUATION_HPP

#include "chess.hpp"

/**
 * Evaluates the board position.
 * Returns a score in centipawns relative to the side whose turn it is.
 * Positive = Current player is winning.
 * Negative = Current player is losing.
 */
int pst_evaluate (const chess::Board& board);

#endif