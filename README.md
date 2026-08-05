# Chess Engine with NNUE Evaluation

A UCI-compatible chess engine written in C++ with two evaluation modes:
a **Piece-Square Table (PST)** evaluator and a **NNUE (Efficiently
Updatable Neural Network)** evaluator. The NNUE is trained from scratch in PyTorch using
a HalfKP feature set and exported to a flat binary file that the C++ engine loads at runtime
with no external ML library dependency.

---

## Table of Contents

- [Chess Engine with NNUE Evaluation](#chess-engine-with-nnue-evaluation)
    - [Table of Contents](#table-of-contents)
    - [Architecture Overview](#architecture-overview)
    - [Project Structure](#project-structure)
    - [Prerequisites](#prerequisites)
        - [C++ Engine](#c-engine)
        - [Python Training Pipeline](#python-training-pipeline)
        - [Testing](#testing)
    - [Building the Engine](#building-the-engine)
    - [Running the Engine](#running-the-engine)
    - [Training the NNUE](#training-the-nnue)
        - [Datasets](#datasets)
        - [Initial Training](#initial-training)
        - [Continuing Training](#continuing-training)
        - [Verifying Weights](#verifying-weights)
    - [Testing with CutechessCLI](#testing-with-cutechesscli)
        - [Fixed Game Count](#fixed-game-count)
        - [SPRT Match](#sprt-match)
        - [Head-to-Head NNUE Comparison](#head-to-head-nnue-comparison)
        - [Parsing Results](#parsing-results)
    - [Viewing Training and Match Plots](#viewing-training-and-match-plots)
        - [Training Loss Curves (Jupyter)](#training-loss-curves-jupyter)
        - [Elo Progression Across Versions (Jupyter or plain Python)](#elo-progression-across-versions-jupyter-or-plain-python)
    - [UCI Options Reference](#uci-options-reference)
    - [Reference](#reference)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                     main.cpp                         │
│   UCI protocol loop — position/go/setoption/eval     │
└─────────────────┬───────────────────────────────────┘
                  │  get_best_move()
┌─────────────────▼───────────────────────────────────┐
│                    search.cpp                        │
│   Alpha-beta + quiescence search                     │
│   Transposition table, killer moves, MVV-LVA         │
│   Iterative deepening with time management           │
└─────────────────┬───────────────────────────────────┘
                  │  evaluate(board)
┌─────────────────▼───────────────────────────────────┐
│                 eval_dispatch.cpp                    │
│   Runtime flag selects PST or NNUE                   │
│   board_to_nnue_position() — bridge to chess library │
└──────────┬──────────────────────┬───────────────────┘
           │                      │
┌──────────▼──────┐    ┌──────────▼──────────────────┐
│  evaluation.cpp │    │         nnue.cpp              │
│  PST evaluator  │    │  HalfKP feature extraction   │
│  Tapered mg/eg  │    │  AVX2-accelerated inference  │
│  Passed pawns   │    │  Loads nnue_weights.bin       │
└─────────────────┘    └──────────────────────────────┘
```

**NNUE Architecture:** `HalfKP (40960 features) → FT(256) × 2 perspectives [ClippedReLU]
→ concat(512) → L1(32) [ClippedReLU] → L2(32) [ClippedReLU] → output(1)`

**AVX2 acceleration** is used for the accumulator refresh and the L1 forward pass,
processing 8 floats per instruction instead of 1. Verified on Intel i5-11320H (Tiger Lake).

---

## Project Structure

```
chess-engine/
├── CMakeLists.txt
├── extern/
│   └── chess-library-master/        ← Disservin's chess library (see Prerequisites)
│       └── include/
│           └── chess.hpp
├── src/
│   ├── main.cpp                     ← UCI loop, time management, eval mode switching
│   ├── search.cpp                   ← Alpha-beta, quiescence search, TT, iterative deepening
│   ├── evaluation.cpp               ← PST evaluator (pst_evaluate)
│   ├── eval_dispatch.h/.cpp         ← Dispatcher — routes evaluate() to PST or NNUE
│   └── nnue.h/.cpp                  ← NNUE inference engine, weight loading
├── torch/
│   ├── nnue_train.py                ← HalfKP feature extraction, NNUE model, training pipeline
│   │                                    (supports both CSV and 10-file parquet datasets)
│   ├── continue_train.py            ← Resume CSV training from a .pt checkpoint
│   ├── continue_train_parquet.py    ← Resume parquet training from a .pt checkpoint
│   ├── verify_weights.py            ← Three-layer parity check: PyTorch / .bin / C++ engine
│   ├── inspect_training.py          ← Jupyter cells: load checkpoints, plot loss curves
│   └── parse_match_results.py       ← Parse cutechess PGN output → Elo CSV
├── plot_elo_progression.py          ← Plot Elo across match runs from match_summary.csv
├── book.pgn                         ← Opening book used by cutechess (you supply this)
├── nnue_weights.bin                 ← Exported inference weights (you generate this)
└── results.pgn                      ← CutechessCLI match output (generated by matches)
```

---

## Prerequisites

### C++ Engine

- **CMake** 3.10 or newer — https://cmake.org/download/
- **Visual Studio 2022** (MSVC) or **MinGW-w64** (GCC) on Windows
- **Disservin's chess library** — the single-header C++ chess library used for board
  representation, legal move generation, and UCI move parsing

    ```bash
    # Clone into the extern/ folder
    git clone https://github.com/Disservin/chess-library.git extern/chess-library-master
    ```

    The library lives at `extern/chess-library-master/include/chess.hpp` — this path is
    already what `CMakeLists.txt` expects. No other installation is needed.

### Python Training Pipeline

- **Python 3.10+**
- **CUDA-capable GPU** strongly recommended (this training was done with MX450 (2GB VRAM); training
  on CPU is possible but significantly slower)
- **CUDA Toolkit** matching your GPU driver — https://developer.nvidia.com/cuda-downloads

Install Python dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install python-chess pandas numpy pyarrow
```

### Testing

- **CutechessCLI** — command-line chess tournament manager
    - Download from https://github.com/cutechess/cutechess/releases
    - Add `cutechess-cli` to your system PATH, or reference it by full path in the commands below

---

## Building the Engine

```bash
# Step 1 — Configure (run once per project, or after changing CMakeLists.txt)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Step 2 — Build (run after every source file change)
cmake --build build --config Release
```

The binary is produced at:

- **MSVC:** `build/Release/chess-engine.exe`
- **MinGW:** `build/chess-engine.exe`

**For debugging and AddressSanitizer builds** (used to diagnose crashes):

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="/fsanitize=address /Zi"
cmake --build build-asan --config Debug
```

---

## Running the Engine

The engine speaks the **UCI protocol** — it is designed to be launched by a GUI (Arena,
CutechessGUI, etc.) or driven by piped commands in a terminal.

**PST mode (default, no weights file needed):**

```bash
./build/Release/chess-engine.exe --pst
```

**NNUE mode:**

```bash
./build/Release/chess-engine.exe --nnue --weights=nnue_weights_epoch4.bin
```

**Manual UCI session for testing:**

```bash
echo -e "uci\nisready\nposition startpos moves e2e4 e7e5\ngo wtime 60000 btime 60000 winc 500 binc 500\nquit" | \
  ./build/Release/chess-engine.exe --nnue --weights=nnue_weights_epoch4.bin
```

**Direct evaluation of a position (for debugging — not part of UCI protocol):**

```bash
echo -e "uci\nisready\nposition fen rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\neval\nquit" | \
  ./build/Release/chess-engine.exe --nnue --weights=nnue_weights_epoch4.bin
```

The `eval` command returns the raw centipawn-equivalent score for the current board position
with zero search — useful for verifying that C++ inference matches PyTorch output.

**Switch eval mode at runtime via UCI `setoption`:**

```
setoption name EvalMode value NNUE
setoption name EvalMode value PST
setoption name WeightsPath value C:\path\to\new_weights.bin
```

---

## Training the NNUE

All training scripts live in the `torch/` folder. Run them from that folder or
from the project root — they import each other by name so they need to be
on the same Python path.

### Datasets

The training pipeline supports two dataset formats.

**Small CSV dataset (~13 million positions):**

| Field    | Value                                                        |
| -------- | ------------------------------------------------------------ |
| Source   | Lichess/Kaggle chess evaluations                             |
| Download | https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations |
| Format   | CSV with columns `FEN`, `Evaluation`                         |
| Size     | ~800 MB                                                      |

Place the downloaded file at: `dataset/chessData.csv`

**Large parquet dataset (~316 million unique positions):**

| Field    | Value                                                      |
| -------- | ---------------------------------------------------------- |
| Source   | Deduplicated Stockfish evaluations (depth 36)              |
| Download | https://huggingface.co/datasets/Lichess/chess-evaluations  |
| Format   | 10 parquet files with columns `fen`, `depth`, `cp`, `mate` |
| Size     | ~6–7 GB total                                              |

Place all 10 parquet files at: `dataset/parquet/*.parquet`

Both datasets store evaluations in **white-relative** centipawns (positive = white is
better). The training scripts handle the sign flip for black-to-move positions
automatically — this is the most critical correctness detail in the pipeline, as
getting it wrong produces a model that learns essentially nothing.

### Initial Training

```bash
cd torch
python nnue_train.py
```

**Before the full run begins**, the script runs an automatic smoke test on 300,000 rows
that checks three things:

1. Whether the `fen` column is `str` or `bytes` in your parquet files
2. Whether the filtering logic is dropping a sensible fraction of rows
3. Whether the sign convention is correct (via material-balance correlation)

It will ask `Proceed with the full training run? [y/N]` — inspect the smoke test output
before typing `y`.

**Expected checkpoints produced:**

- `nnue_parquet_epoch01.pt`, `nnue_parquet_epoch02.pt` — full epoch checkpoints
- `nnue_step10000.pt`, `nnue_step20000.pt`, ... — mid-epoch checkpoints every 10,000 steps
- `training_log_parquet.json` — full history of all metrics for plotting

**Export the trained weights for the C++ engine:**

```bash
# Already happens automatically at the end of nnue_train.py
# To export manually from any checkpoint:
python -c "
import torch
from nnue_train import NNUE, export_weights
model = NNUE()
ckpt = torch.load('nnue_parquet_epoch02.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state'])
export_weights(model, '../nnue_weights.bin')
"
```

### Continuing Training

To resume from a saved checkpoint for additional epochs:

**Parquet pipeline (recommended — 316M positions):**

```bash
cd torch
python continue_train_parquet.py
```

Edit the bottom of `continue_train_parquet.py` to point to your latest checkpoint
and set the number of additional epochs before running.

**CSV pipeline (13M positions):**

```bash
cd torch
python continue_train.py
```

Both continuation scripts:

- Restore model weights **and** the AdamW optimizer's momentum/variance buffers from the checkpoint
- Append new epoch metrics to the existing `training_log*.json` file so the full history
  is kept in one place
- Save new checkpoints with sequentially numbered names (`nnue_parquet_epoch03.pt`, etc.)
- Apply `ReduceLROnPlateau` — the learning rate is halved automatically if validation loss
  stops improving for 3 consecutive checkpoint checks (~30,000 steps)

### Verifying Weights

Before running any cutechess match with a new set of weights, verify that the C++
engine's inference produces the same scores as the PyTorch model on the same positions:

```bash
cd torch
python verify_weights.py nnue_parquet_epoch02.pt ../nnue_weights.bin
```

This performs a **three-layer parity check**:

1. **Layer 1 (PyTorch)** — runs the model directly in Python
2. **Layer 2 (NumPy from .bin)** — reads the raw bytes of `nnue_weights.bin` and
   re-implements the forward pass from scratch in NumPy, completely independent of
   PyTorch, verifying the export format
3. **Layer 3 (C++ engine)** — the script prints the exact `position fen ... eval`
   commands to run against the C++ binary and the expected scores to compare against

Expected output for a correctly working pipeline:

```
FEN (truncated)                                   PyTorch  NumPy(.bin)     diff
--------------------------------------------------------------------------------
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w      57.812       57.812   0.0000
rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w     128.942      128.942   0.0000
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w      -7.890       -7.890   0.0000
PASS — PyTorch and the raw .bin file agree on every test position.
```

---

## Testing with CutechessCLI

Cutechess communicates with your engine using the standard **UCI protocol** —
it sends `position startpos moves ...` and `go wtime ... btime ...` commands, and
your engine responds with `bestmove`. The engine handles both PST and NNUE modes
transparently; cutechess is not aware of which evaluator is active.

You need an opening book (`book.pgn`) — a small PGN file containing a variety of
opening lines so each game pair starts from a different position. A commonly used
free option is the `8moves_v3.pgn` suite, searchable on GitHub.

### Fixed Game Count

Standard 200-game match, NNUE vs PST:

```bash
cutechess-cli ^
  -engine name=PST  cmd=build\Release\chess-engine.exe arg=--pst proto=uci ^
  -engine name=NNUE cmd=build\Release\chess-engine.exe arg=--nnue arg=--weights=nnue_weights.bin proto=uci ^
  -each tc=40/60+0.5 ^
  -games 200 ^
  -repeat ^
  -concurrency 2 ^
  -recover ^
  -draw movenumber=40 movecount=8 score=10 ^
  -resign movecount=5 score=600 ^
  -openings file=book.pgn format=pgn order=random plies=8 ^
  -pgnout results.pgn ^
  -ratinginterval 10
```

**Time control:** `tc=40/60+0.5` — 40 moves in 60 seconds plus 0.5s increment. This
tests real playing strength under time pressure rather than fixed depth, which is
fairer given that NNUE and PST have different per-node speeds.

**`-concurrency 2`** — runs 2 games in parallel. Safe on a 4-core machine.
Do not go higher with single-threaded engine processes.

**`-repeat`** — plays each opening from both sides, cancelling out first-move advantage.

### SPRT Match

Sequential Probability Ratio Test — stops automatically once the result is
statistically significant. More efficient than a fixed game count for determining
whether a specific Elo threshold has been crossed.

```bash
cutechess-cli ^
  -engine name=PST  cmd=build\Release\chess-engine.exe arg=--pst proto=uci ^
  -engine name=NNUE cmd=build\Release\chess-engine.exe arg=--nnue arg=--weights=nnue_weights.bin proto=uci ^
  -each tc=40/60+0.5 ^
  -games 1000 ^
  -repeat ^
  -concurrency 2 ^
  -recover ^
  -draw movenumber=40 movecount=8 score=10 ^
  -resign movecount=5 score=600 ^
  -openings file=book.pgn format=pgn order=random plies=8 ^
  -sprt elo0=0 elo1=30 alpha=0.05 beta=0.05 ^
  -pgnout results_sprt.pgn
```

`-sprt elo0=0 elo1=30 alpha=0.05 beta=0.05` tests: "is NNUE at least 30 Elo
stronger than PST?" at a 95% confidence level. The match stops early as soon
as it has a confident answer — sometimes in 100 games, sometimes needing
the full 1000.

### Head-to-Head NNUE Comparison

To compare two different sets of NNUE weights (e.g. before/after a training improvement),
run them directly against each other — this eliminates PST as a confound and gives a
cleaner signal for whether the training change helped:

```bash
cutechess-cli ^
  -engine name=NNUE_v1 cmd=build\Release\chess-engine.exe arg=--nnue arg=--weights=nnue_weights_v1.bin proto=uci ^
  -engine name=NNUE_v2 cmd=build\Release\chess-engine.exe arg=--nnue arg=--weights=nnue_weights_v2.bin proto=uci ^
  -each tc=40/60+0.5 ^
  -games 600 ^
  -repeat ^
  -concurrency 2 ^
  -recover ^
  -draw movenumber=40 movecount=8 score=10 ^
  -resign movecount=5 score=600 ^
  -openings file=book.pgn format=pgn order=random plies=8 ^
  -sprt elo0=0 elo1=20 alpha=0.05 beta=0.05 ^
  -pgnout results_v1_vs_v2.pgn
```

### Parsing Results

After any match, parse the PGN output to compute Elo difference and append a
row to `match_summary.csv`:

```bash
python parse_match_results.py results.pgn --version "epoch2_parquet_rLRplat"
```

If cutechess appends games to the same PGN file across multiple runs (the default
behaviour), use `--skip` to count only the newest batch:

```bash
# Run 1: 200 games in file — no skip needed
python parse_match_results.py results.pgn --version "epoch2_300m"

# Run 2: file now has 400 games — skip the first 200 already counted
python parse_match_results.py results.pgn --version "epoch2_rLRplat" --skip 200

# Run 3: file now has 600 games — skip the first 400
python parse_match_results.py results.pgn --version "epoch4_continued" --skip 400
```

This produces two files:

- `match_summary.csv` — one row per version (appends across runs, perfect for plotting)
- `games_detail.csv` — one row per individual game (overwritten each run)

---

## Viewing Training and Match Plots

### Training Loss Curves (Jupyter)

Open `torch/inspect_training.py` as a Jupyter notebook in VS Code.
Run each cell in order:

1. **Cell 1** — imports everything from `nnue_train.py`
2. **Cell 2** — reads `training_log.json` or `training_log_parquet.json` and draws
   three side-by-side plots:

    ![Training curves showing MSE loss, Pearson r, and sign accuracy across epochs](docs/training_curves.png)
    - **MSE Loss** (train + val) — both curves should fall together. A rising val loss
      while train loss falls indicates overfitting.
    - **Pearson r** — correlation between predicted and target win probabilities.
      A well-trained model reaches 0.85+.
    - **Sign accuracy** — percentage of positions where the model correctly identifies
      which side is winning. Target: 85%+.

3. **Cell 3** — loads a specific checkpoint by path
4. **Cell 4** — re-runs `compute_metrics()` on the validation slice
5. **Cell 5** — evaluates arbitrary FEN strings and prints raw centipawn scores:

```python
evaluate_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
# Starting position — should be close to 0

evaluate_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
# White up a queen — should be strongly positive

evaluate_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
# Black up a queen — should be strongly negative
```

### Elo Progression Across Versions (Jupyter or plain Python)

After running multiple matches and parsing each one with `parse_match_results.py`,
plot how Elo has changed across training versions:

```bash
python plot_elo_progression.py
```

Or run `plot_elo_progression.py` as a Jupyter cell. This reads `match_summary.csv`
and produces two charts side by side:

![Elo progression chart showing Elo diff with error bars and W/L/D stacked bar by version](docs/elo_progression.png)

- **Left chart** — Elo difference with 95% confidence interval error bars, per version.
  The dashed line at 0 is the PST baseline. Points above it mean NNUE is winning.
- **Right chart** — Win/loss/draw stacked bar per version. Useful for spotting if
  a version has an unusual draw rate or one-sided result pattern that the Elo number
  alone doesn't reveal.

The chart is also saved to `elo_progression.png` automatically.

---

## UCI Options Reference

| Option        | Type                   | Default            | Description                 |
| ------------- | ---------------------- | ------------------ | --------------------------- |
| `EvalMode`    | combo (`PST` / `NNUE`) | `PST`              | Switch evaluator at runtime |
| `WeightsPath` | string                 | `nnue_weights.bin` | Path to NNUE weights binary |

These can be set from any UCI GUI (Arena, CutechessGUI, etc.) via its engine
configuration panel, or sent directly as `setoption name X value Y` commands.

---

## Reference

- **Disservin's chess library** — https://github.com/Disservin/chess-library
- **CutechessCLI** — https://github.com/cutechess/cutechess
- **Chess Evaluations Dataset (13M CSV)** — https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations
- **Lichess Chess Evaluations (316M parquet)** — https://huggingface.co/datasets/Lichess/chess-evaluations
- **NNUE paper** (Nasu, 2018) — https://www.chessprogramming.org/NNUE
- **Chess Programming Wiki — NNUE** — https://www.chessprogramming.org/Stockfish_NNUE
