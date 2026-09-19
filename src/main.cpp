#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include "chess.hpp"
#include "eval_dispatch.h"
#include <thread>
#include <atomic>
#include "search.hpp"
#include <iostream>
#include <thread>

// chess::Board board;
std::pair<chess::Move, int> get_best_move (chess::Board& board, int depth);
std::thread g_search_thread;

// ─────────────────────────────────────────────────────────────────────────────
// GoParams — parsed fields from a "go ..." command
// ─────────────────────────────────────────────────────────────────────────────
struct GoParams {
    bool has_depth = false;
    int  depth = 0;
    bool has_movetime = false;
    long movetime_ms = 0;
    bool has_time = false;   // wtime/btime present
    long wtime = 0;
    long btime = 0;
    long winc = 0;
    long binc = 0;
    int  movestogo = 30;      // cutechess default assumption if absent
    bool infinite = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// search_with_time_budget
// Iterative deepening: searches depth 1, 2, 3, ... using the existing
// get_best_move(board, depth), stopping once the time budget is consumed.
//
// get_best_move() has no internal abort mechanism, so we can't interrupt a
// search mid-depth. Instead we predict whether the NEXT depth will fit in
// the remaining budget using the previous iteration's elapsed time and a
// branching-factor estimate, and only attempt it if it plausibly fits.
// This is the standard technique simple engines use before implementing
// true mid-search time checks.
// ─────────────────────────────────────────────────────────────────────────────
constexpr int MAX_ID_DEPTH = 12;   // safety ceiling regardless of time budget

// ─────────────────────────────────────────────────────────────────────────────
// compute_time_budget
// Converts wtime/btime/winc/binc/movestogo into a millisecond budget for
// THIS move, using the side to move's clock.
// ─────────────────────────────────────────────────────────────────────────────
long compute_time_budget (const GoParams& g, bool white_to_move) {
    if (g.has_movetime) return g.movetime_ms;

    if (g.has_time) {
        long my_time = white_to_move ? g.wtime : g.btime;
        long my_inc = white_to_move ? g.winc : g.binc;

        // Standard heuristic: time for THIS move = remaining/movestogo + most of the increment.
        // Subtract a small safety margin to avoid flagging on overhead/IO latency.
        long budget = (my_time / std::max (1, g.movestogo)) + (long)(my_inc * 0.8);
        budget -= 50;                              // safety margin
        budget = std::max (budget, (long)50);        // never go below 50ms
        budget = std::min (budget, my_time / 2);     // never use more than half remaining clock
        return budget;
    }

    // Neither movetime nor wtime/btime given (e.g. manual "go depth N" testing,
    // or "go infinite" which we treat as a generous fixed budget here since we
    // don't implement true infinite/ponder search).
    return 5000;
}

void run_search_and_report (chess::Board board, GoParams g) {
    std::vector<RootMoveResult> result;
    try {
        if (g.has_depth && !g.has_time && !g.has_movetime) {
            result = run_iterative_search (board, g.depth, -1);
        }
        else {
            bool white_to_move = (board.sideToMove () == chess::Color::WHITE);
            long budget_ms = compute_time_budget (g, white_to_move);
            result = run_iterative_search (board, MAX_ID_DEPTH, budget_ms);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[FATAL-AVOIDED] search threw: " << e.what () << "\n";
    }

    if (result.empty ()) {
        // Either no legal moves at all, or stopped before depth 1 even
        // finished - same fallback philosophy as the old exception path,
        // so the engine always produces a valid bestmove.
        chess::Movelist fallback;
        chess::movegen::legalmoves (fallback, board);
        if (!fallback.empty ()) result.push_back ({ fallback[0], 0 });
    }

    chess::Move best = result.empty () ? chess::Move::NO_MOVE : result[0].move;
    std::cout << "bestmove " << chess::uci::moveToUci (best) << std::endl;
}


void new_game_reset ();
using Clock = std::chrono::steady_clock;
// ── parse_args ────────────────────────────────────────────────────────────────
// Reads argc/argv and sets g_use_nnue and g_weights_path before the UCI loop.
//
// Supported flags:
//   --nnue                use NNUE evaluation (loads nnue_weights.bin)
//   --pst                 use PST evaluation  (default, no flag needed)
//   --weights=path/to/file.bin   load NNUE weights from a custom path
//
// Usage examples:
//   ./engine                            → PST mode
//   ./engine --nnue                     → NNUE mode, default weights path
//   ./engine --nnue --weights=my.bin    → NNUE mode, custom weights path
// ─────────────────────────────────────────────────────────────────────────────
void parse_args (int argc, char* argv[]) {
    bool want_nnue = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--nnue") {
            want_nnue = true;
        }
        else if (arg == "--pst") {
            want_nnue = false;
        }
        else if (arg.rfind ("--weights=", 0) == 0) {
            g_weights_path = arg.substr (10);
        }
    }

    if (want_nnue) {
        // init_nnue prints its own success/failure message to stderr.
        // stderr goes to your terminal; stdout is reserved for UCI protocol.
        init_nnue (g_weights_path);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_uci_moves
// Applies a sequence of UCI move strings (e.g. "e2e4", "g1f3") to the board.
// Used after "position startpos moves ..." or "position fen ... moves ...".
// ─────────────────────────────────────────────────────────────────────────────
void apply_uci_moves (chess::Board& board, const std::vector<std::string>& tokens, size_t start_idx) {
    for (size_t i = start_idx; i < tokens.size (); i++) {
        try {
            chess::Move m = chess::uci::uciToMove (board, tokens[i]);
            if (m == chess::Move::NO_MOVE) break;   // malformed — stop rather than crash
            board.makeMove (m);
        }

        catch (const std::exception& e) {
            // uciToMove can throw on a move string it can't resolve against
            // the current position (e.g. unexpected promotion/castling
            // notation). Log it to stderr so it shows up in cutechess's
            // stderr= log file, and stop applying further moves rather
            // than letting the exception propagate and kill the process.
            std::cerr << "[FATAL-AVOIDED] apply_uci_moves failed on token '"
                << tokens[i] << "': " << e.what () << "\n";
            break;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// handle_position
// Supports both forms cutechess actually sends:
//   position startpos [moves m1 m2 ...]
//   position fen <fen> [moves m1 m2 ...]
// The previous version only handled "position fen " with no moves suffix,
// which silently dropped every move played after the opening — this also
// contributed to incorrect game state even on the rare cases it didn't stall.
// ─────────────────────────────────────────────────────────────────────────────
void handle_position (chess::Board& board, const std::string& line) {
    std::istringstream iss (line);
    std::vector<std::string> tokens;
    std::string tok;
    while (iss >> tok) tokens.push_back (tok);
    // tokens[0] == "position"

    size_t idx = 1;
    if (idx < tokens.size () && tokens[idx] == "startpos") {
        board = chess::Board ();   // default constructor = standard start position
        idx++;
    }
    else if (idx < tokens.size () && tokens[idx] == "fen") {
        idx++;
        // FEN is 6 space-separated fields; collect exactly those before "moves"
        std::string fen;
        int fields = 0;
        while (idx < tokens.size () && tokens[idx] != "moves" && fields < 6) {
            if (!fen.empty ()) fen += " ";
            fen += tokens[idx];
            idx++;
            fields++;
        }
        board = chess::Board (fen);
    }

    if (idx < tokens.size () && tokens[idx] == "moves") {
        idx++;
        apply_uci_moves (board, tokens, idx);
    }
}



GoParams parse_go (const std::string& line) {
    GoParams g;
    std::istringstream iss (line);
    std::string tok;
    iss >> tok;   // consume "go"

    while (iss >> tok) {
        if (tok == "depth") { iss >> g.depth;      g.has_depth = true; }
        else if (tok == "movetime") { iss >> g.movetime_ms; g.has_movetime = true; }
        else if (tok == "wtime") { iss >> g.wtime;      g.has_time = true; }
        else if (tok == "btime") { iss >> g.btime;      g.has_time = true; }
        else if (tok == "winc") { iss >> g.winc; }
        else if (tok == "binc") { iss >> g.binc; }
        else if (tok == "movestogo") { iss >> g.movestogo; }
        else if (tok == "infinite") { g.infinite = true; }
        // "ponder", "nodes", "mate" etc. are accepted but ignored
    }
    return g;
}


std::pair<chess::Move, int> search_with_time_budget (chess::Board& board, long budget_ms) {
    auto t_start = Clock::now ();

    chess::Move best_move = chess::Move::NO_MOVE;
    int         best_score = 0;
    long long   last_iter_ms = 0;

    for (int depth = 1; depth <= MAX_ID_DEPTH; depth++) {
        auto t_iter_start = Clock::now ();
        auto [move, score] = get_best_move (board, depth);
        auto t_iter_end = Clock::now ();

        last_iter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_iter_end - t_iter_start).count ();
        long long total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_iter_end - t_start).count ();

        best_move = move;
        best_score = score;

        std::cout << "info depth " << depth << " score cp " << score
            << " time " << total_ms << "\n";

        if (total_ms >= budget_ms) break;

        // Branching factor estimate: chess search trees typically grow
        // 3-6x per additional ply. Use 4x as a conservative middle estimate.
        long long predicted_next_ms = last_iter_ms * 4;
        if (total_ms + predicted_next_ms > budget_ms) break;
    }

    return { best_move, best_score };
}


int main (int argc, char* argv[]) {
    parse_args (argc, argv);

    std::string  line;
    chess::Board board;

    while (std::getline (std::cin, line)) {
        if (line == "uci") {
            std::cout << "id name MyEngine\n";
            std::cout << "id author You\n";
            std::cout << "option name EvalMode type combo default "
                << (g_use_nnue ? "NNUE" : "PST")
                << " var PST var NNUE\n";
            std::cout << "option name WeightsPath type string default "
                << g_weights_path << "\n";
            std::cout << "uciok" << std::endl;
        }
        else if (line == "isready") {
            std::cout << "readyok" << std::endl;
        }
        else if (line == "ucinewgame") {
            // Reset search state between games so stale TT/killer data from
            // a finished game doesn't influence the next one.
            if (g_search_thread.joinable ()) g_search_thread.join ();
            new_game_reset ();
            board = chess::Board ();
        }
        else if (line.rfind ("position", 0) == 0) {
            handle_position (board, line);
        }
        else if (line.rfind ("go", 0) == 0) {
            if (g_search_thread.joinable ()) g_search_thread.join ();
            g_stop_requested = false;
            GoParams g = parse_go (line);
            g_search_thread = std::thread ([board, g]() mutable {
                run_search_and_report (board, g);
                });
        }
        else if (line.rfind ("setoption name ", 0) == 0) {
            size_t name_start = 15;                          // length of "setoption name "
            size_t value_pos = line.find (" value ", name_start);

            if (value_pos == std::string::npos) {
                std::cerr << "[setoption] malformed line, ignoring: " << line << "\n";
            }
            else {
                std::string option_name = line.substr (name_start, value_pos - name_start);
                std::string option_value = line.substr (value_pos + 7);   // length of " value "

                if (option_name == "EngineType") {
                    std::string v = option_value;
                    std::transform (v.begin (), v.end (), v.begin (), ::tolower);

                    if (v == "nnue") {
                        std::cerr << "[EngineType] Switching to NNUE...\n";
                        // init_nnue() already sets g_use_nnue itself based on whether
                        // the file actually loaded. Trust its return value — do NOT
                        // force g_use_nnue = true afterward, or a failed load gets
                        // silently treated as a success and the engine runs inference
                        // on zero-initialized weights.
                        if (init_nnue (g_weights_path))
                            std::cerr << "[EngineType] NNUE active (weights: " << g_weights_path << ")\n";
                        else
                            std::cerr << "[EngineType] Failed to load weights — staying on PST.\n";
                    }
                    else if (v == "pst") {
                        std::cerr << "[EngineType] Switching to PST...\n";
                        g_use_nnue = false;
                    }
                    else {
                        std::cerr << "[EngineType] Unrecognized value '" << option_value
                            << "' (expected 'nnue' or 'pst') — ignoring.\n";
                    }
                }
                else if (option_name == "MultiPV") {
                    try {
                        int v = std::stoi (option_value);
                        g_multipv = std::max (1, v);
                        std::cerr << "[MultiPV] Set to " << g_multipv << "\n";
                    }
                    catch (const std::exception&) {
                        std::cerr << "[MultiPV] Invalid value '" << option_value << "' - ignoring.\n";
                    }
                }
                else if (option_name == "WeightsPath") {
                    g_weights_path = option_value;
                    std::cerr << "[WeightsPath] Set to: " << g_weights_path << "\n";

                    if (g_use_nnue) {
                        // Already active — reload immediately with the new file.
                        if (!init_nnue (g_weights_path))
                            std::cerr << "[WeightsPath] Reload failed — check the path.\n";
                    }
                    // If not currently in NNUE mode, the path is just stored and
                    // will be used the next time EngineType switches to nnue.
                }
                else {
                    std::cerr << "[setoption] Unknown option '" << option_name << "' — ignoring.\n";
                }
            }
        }
        else if (line == "eval") {
            // Non-standard debug command — NOT part of the UCI protocol,
            // cutechess will never send this. Purpose: call evaluate()
            // DIRECTLY on the current position with zero search, so its
            // output can be compared 1:1 against Python's raw model output
            // on the identical FEN. "go depth N" always evaluates positions
            // AFTER at least one move (via get_best_move's root move loop),
            // so it can't be used for this — this command is the only
            // direct path to a true apples-to-apples comparison.
            int raw_score = evaluate (board);
            std::cout << "eval_score " << raw_score
                << " string eval=" << (g_use_nnue ? "NNUE" : "PST") << std::endl;
        }
        else if (line == "stop") {
            g_stop_requested = true;
            if (g_search_thread.joinable ()) g_search_thread.join ();
        }
        else if (line == "quit") {
            g_stop_requested = true;
            if (g_search_thread.joinable ()) g_search_thread.join ();
            break;
        }
    }

    return 0;
}