"""
parse_match_results.py
─────────────────────────────────────────────────────────────────────────────
Parses a cutechess-cli PGN output file and produces:
  1. match_summary.csv   — one row per match run (for tracking Elo over time
                            as you retrain/improve the NNUE)
  2. games_detail.csv    — one row per individual game (for inspecting
                            streaks, color balance, game length, etc.)

Usage:
    python parse_match_results.py results.pgn --version "epoch20_avx2"

Run this after every cutechess match (e.g. after retraining with a bigger
dataset) with a different --version label, and match_summary.csv will
accumulate a row per run — perfect for plotting Elo progression over time.
─────────────────────────────────────────────────────────────────────────────
"""

import re
import csv
import math
import argparse
import os
from datetime import datetime


def elo_diff_from_score(wins: int, losses: int, draws: int):
    """
    Standard Elo difference formula from match score, with a 95% confidence
    interval using the normal approximation (the same method cutechess-cli
    itself uses internally).

    Returns (elo_diff, elo_error_margin) or (None, None) if undefined
    (e.g. all wins or all losses — division by zero in the logit).
    """
    n = wins + losses + draws
    if n == 0:
        return None, None

    score = (wins + 0.5 * draws) / n  # fraction of points won, in [0, 1]

    # Guard against log(0) at the extremes
    if score <= 0.0 or score >= 1.0:
        return None, None

    elo_diff = -400.0 * math.log10(1.0 / score - 1.0)

    # Standard error of the score, then propagated into Elo space
    # via the derivative of the logit transform.
    variance = (
        sum(
            [
                wins * (1 - score) ** 2,
                losses * (0 - score) ** 2,
                draws * (0.5 - score) ** 2,
            ]
        )
        / n
    )
    std_err_score = math.sqrt(variance / n)

    # Propagate to Elo space: d(elo)/d(score) = 400 / (ln(10) * score * (1-score))
    elo_error = std_err_score * 400.0 / (math.log(10) * score * (1 - score))

    return elo_diff, elo_error * 1.96  # 1.96 → 95% confidence interval


def parse_pgn(pgn_path: str):
    """
    Extracts per-game results from a multi-game PGN file.
    Returns a list of dicts: [{white, black, result, num_moves, eco}, ...]
    """
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Games are separated by blank lines after the movetext; split on
    # the pattern that starts a new game header block.
    game_blocks = re.split(r"\n(?=\[Event)", content)

    games = []
    for block in game_blocks:
        if not block.strip():
            continue

        def tag(name):
            m = re.search(rf'\[{name} "(.*?)"\]', block)
            return m.group(1) if m else None

        white = tag("White")
        black = tag("Black")
        result = tag("Result")
        eco = tag("ECO")

        if white is None or result is None:
            continue  # malformed block, skip

        # Count move pairs roughly by counting move-number markers "N."
        num_moves = len(re.findall(r"\d+\.\s", block))

        games.append(
            {
                "white": white,
                "black": black,
                "result": result,
                "eco": eco or "",
                "num_moves": num_moves,
            }
        )

    return games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn_file", help="Path to cutechess-cli PGN output")
    parser.add_argument(
        "--version", required=True, help='Label for this match run, e.g. "epoch20_avx2"'
    )
    parser.add_argument("--summary_csv", default="match_summary.csv")
    parser.add_argument("--detail_csv", default="games_detail.csv")
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip this many games from the START of the PGN file. "
        "Use this when cutechess keeps appending new games to the "
        "same results.pgn across multiple runs: pass the total "
        "game count from all PREVIOUS runs so only the newest "
        "batch gets counted (e.g. --skip 200, then --skip 400, ...).",
    )
    args = parser.parse_args()

    games = parse_pgn(args.pgn_file)
    print(f"Parsed {len(games)} games total from {args.pgn_file}")

    if not games:
        print("No games found — check the PGN file path/format.")
        return

    if args.skip > 0:
        if args.skip >= len(games):
            print(
                f"--skip {args.skip} is >= total games in file ({len(games)}) "
                f"— nothing new to analyze yet."
            )
            return
        print(
            f"Skipping first {args.skip} games (already counted in a previous run) "
            f"— analyzing the remaining {len(games) - args.skip} new game(s)"
        )
        games = games[args.skip :]

    # Identify the two engine names from the first game
    names = {games[0]["white"], games[0]["black"]}
    if len(names) != 2:
        print(f"Warning: expected 2 distinct engine names, found {names}")
    name_a, name_b = sorted(names) if len(names) == 2 else (list(names)[0], "Unknown")

    # ── Tally wins/losses/draws from name_a's perspective ──────────────────
    wins = losses = draws = 0
    for g in games:
        result = g["result"]
        is_white_a = g["white"] == name_a

        if result == "1/2-1/2":
            draws += 1
        elif result == "1-0":
            if is_white_a:
                wins += 1
            else:
                losses += 1
        elif result == "0-1":
            if is_white_a:
                losses += 1
            else:
                wins += 1
        # else: '*' (unfinished) — skip

    total = wins + losses + draws
    elo_diff, elo_margin = elo_diff_from_score(wins, losses, draws)
    avg_len = sum(g["num_moves"] for g in games) / len(games)

    print(f"\n{name_a} vs {name_b}")
    print(f"  W/L/D : {wins}/{losses}/{draws}  (n={total})")
    if elo_diff is not None:
        print(
            f"  Elo diff ({name_a} perspective): {elo_diff:+.1f} +/- {elo_margin:.1f}"
        )
    else:
        print(f"  Elo diff: undefined (one-sided result, need more games)")
    print(f"  Avg game length: {avg_len:.1f} move markers")

    # ── Write/append match_summary.csv ───────────────────────────────────
    summary_exists = os.path.exists(args.summary_csv)
    with open(args.summary_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not summary_exists:
            writer.writerow(
                [
                    "timestamp",
                    "version",
                    "engine_a",
                    "engine_b",
                    "wins",
                    "losses",
                    "draws",
                    "total_games",
                    "elo_diff",
                    "elo_margin_95",
                    "avg_game_length",
                ]
            )
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                args.version,
                name_a,
                name_b,
                wins,
                losses,
                draws,
                total,
                f"{elo_diff:.2f}" if elo_diff is not None else "",
                f"{elo_margin:.2f}" if elo_margin is not None else "",
                f"{avg_len:.1f}",
            ]
        )
    print(f"\nAppended summary row to {args.summary_csv}")

    # ── Write games_detail.csv (overwritten each run) ───────────────────
    with open(args.detail_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["game_num", "white", "black", "result", "eco", "num_moves"])
        for i, g in enumerate(games, args.skip + 1):
            writer.writerow(
                [i, g["white"], g["black"], g["result"], g["eco"], g["num_moves"]]
            )
    print(f"Wrote per-game detail to {args.detail_csv}")


if __name__ == "__main__":
    main()
