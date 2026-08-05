"""
verify_weights.py
─────────────────────────────────────────────────────────────────────────────
Three-layer parity check across the full Python → .bin → C++ pipeline.

  Layer 1: PyTorch          — loads the .pt checkpoint, runs NNUE.forward()
  Layer 2: NumPy (from .bin) — reads nnue_weights.bin's RAW BYTES directly
                                and reimplements the exact forward math used
                                by nnue.cpp, completely independent of
                                PyTorch. This is the critical check: it
                                verifies the EXPORT FORMAT itself (float
                                order/layout), not just re-running the same
                                code that produced the file.
  Layer 3: C++ engine        — you run this manually (see printed
                                instructions at the bottom) using the new
                                'eval' debug command added to main.cpp.

If Layer 1 and Layer 2 disagree → bug in export_weights() or in how this
script reads the .bin file back.
If Layer 1/2 agree but Layer 3 disagrees → bug in nnue.cpp's C++ math, or
in how board_to_nnue_position()/halfkp_index() compute feature indices
differently than Python's halfkp_features() does for the same FEN.

Usage:
    python verify_weights.py nnue_epoch15.pt nnue_weights.bin
"""

import argparse
import numpy as np
import torch

from nnue_train import NNUE, halfkp_features, HALFKP_SIZE, FT_SIZE

INPUT_SIZE = 2 * FT_SIZE  # 512 — both perspectives concatenated
L1_SIZE = 32
L2_SIZE = 32

# ─────────────────────────────────────────────────────────────────────────────
# Test positions
# ─────────────────────────────────────────────────────────────────────────────
#
# NOTE: the very first material-imbalance test used earlier in this project
# had a bug — the FEN labeled "Black up a queen" actually removed a white
# PAWN, not a queen (rank 1 "PPPP1PPP" vs the rank-8 queen square). Fixed
# here: removing White's queen means changing rank 1 "RNBQKBNR" → "RNB1KBNR".
TEST_FENS = [
    # Starting position — should be close to 0 either way.
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # White up a queen (black's queen removed, rank 8 "rnb1kbnr") — strongly +
    "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # Black up a queen (white's queen removed, rank 1 "RNB1KBNR") — strongly -
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
    # A real position from the 316M-row dataset sample, with halfmove/fullmove
    # counters appended explicitly — the dataset's FENs omit them, and we
    # don't want to rely on python-chess and the C++ chess library defaulting
    # missing fields the SAME way (unverified assumption); appending " 0 1"
    # removes that variable entirely from this comparison.
    "2bq1rk1/pr3ppn/1p2p3/7P/2pP1B1P/2P5/PPQ2PB1/R3R1K1 w - - 0 1",
]


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: read nnue_weights.bin directly, reimplement nnue.cpp's math in NumPy
# ─────────────────────────────────────────────────────────────────────────────


def crelu_np(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def load_bin_weights(path: str) -> dict:
    """
    Reads the exact same byte layout export_weights() wrote and nnue.cpp's
    NNUEWeights::load() reads — same field order, same float32, same shapes.
    """
    with open(path, "rb") as f:
        ft_weight = np.fromfile(
            f, dtype=np.float32, count=HALFKP_SIZE * FT_SIZE
        ).reshape(HALFKP_SIZE, FT_SIZE)
        ft_bias = np.fromfile(f, dtype=np.float32, count=FT_SIZE)
        l1_weight = np.fromfile(
            f, dtype=np.float32, count=L1_SIZE * INPUT_SIZE
        ).reshape(L1_SIZE, INPUT_SIZE)
        l1_bias = np.fromfile(f, dtype=np.float32, count=L1_SIZE)
        l2_weight = np.fromfile(f, dtype=np.float32, count=L2_SIZE * L1_SIZE).reshape(
            L2_SIZE, L1_SIZE
        )
        l2_bias = np.fromfile(f, dtype=np.float32, count=L2_SIZE)
        out_weight = np.fromfile(f, dtype=np.float32, count=L2_SIZE)
        out_bias = float(np.fromfile(f, dtype=np.float32, count=1)[0])

    return dict(
        ft_weight=ft_weight,
        ft_bias=ft_bias,
        l1_weight=l1_weight,
        l1_bias=l1_bias,
        l2_weight=l2_weight,
        l2_bias=l2_bias,
        out_weight=out_weight,
        out_bias=out_bias,
    )


def numpy_forward(fen: str, w: dict) -> float:
    """
    Mirrors nnue.cpp's refresh_accumulator() + nnue_evaluate() exactly:
    sparse feature-row summation, ClippedReLU, STM-first concatenation,
    then two dense layers and a scalar output. Uses the SAME w_idx/b_idx/stm
    that training used (via halfkp_features), so this checks the .bin file's
    CONTENTS — not the feature-indexing logic, which Layer 3 checks instead.
    """
    w_idx, b_idx, stm = halfkp_features(fen)

    w_acc = w["ft_bias"].copy()
    for i in w_idx:
        w_acc += w["ft_weight"][i]
    w_acc = crelu_np(w_acc)

    b_acc = w["ft_bias"].copy()
    for i in b_idx:
        b_acc += w["ft_weight"][i]
    b_acc = crelu_np(b_acc)

    # STM perspective goes first — same convention as training and as
    # nnue.cpp's nnue_evaluate().
    x = np.concatenate([w_acc, b_acc]) if stm == 1 else np.concatenate([b_acc, w_acc])

    l1 = crelu_np(w["l1_weight"] @ x + w["l1_bias"])
    l2 = crelu_np(w["l2_weight"] @ l1 + w["l2_bias"])
    out = float(w["out_weight"] @ l2 + w["out_bias"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: PyTorch
# ─────────────────────────────────────────────────────────────────────────────


def pytorch_forward(fen: str, model: NNUE, device) -> float:
    w_idx, b_idx, stm = halfkp_features(fen)
    w_flat = torch.tensor(w_idx, dtype=torch.long, device=device)
    b_flat = torch.tensor(b_idx, dtype=torch.long, device=device)
    w_off = torch.zeros(1, dtype=torch.long, device=device)
    b_off = torch.zeros(1, dtype=torch.long, device=device)
    stm_t = torch.tensor([float(stm)], dtype=torch.float32, device=device)
    with torch.no_grad():
        raw = model(w_flat, w_off, b_flat, b_off, stm_t)
    return raw.item()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help=".pt checkpoint path (e.g. nnue_epoch15.pt)")
    parser.add_argument(
        "weights_bin", help="exported .bin weights path (e.g. nnue_weights.bin)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="Max allowed |PyTorch - NumPy| difference before flagging a mismatch",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = NNUE().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    w = load_bin_weights(args.weights_bin)
    print(f"Loaded weights binary: {args.weights_bin}\n")

    print(f"{'FEN (truncated)':46} {'PyTorch':>10} {'NumPy(.bin)':>12} {'diff':>8}")
    print("-" * 80)

    any_mismatch = False
    for fen in TEST_FENS:
        pt_score = pytorch_forward(fen, model, device)
        np_score = numpy_forward(fen, w)
        diff = abs(pt_score - np_score)
        flag = ""
        if diff > args.tolerance:
            flag = "  ⚠ MISMATCH"
            any_mismatch = True
        print(f"{fen[:46]:46} {pt_score:10.3f} {np_score:12.3f} {diff:8.4f}{flag}")

    print()
    if any_mismatch:
        print("⚠ Layer 1 vs Layer 2 MISMATCH detected — the bug is in export_weights()")
        print("  or in how this script reads nnue_weights.bin back. Fix this BEFORE")
        print("  bothering to test the C++ engine at all.")
    else:
        print("PASS — PyTorch and the raw .bin file agree on every test position.")
        print("Layer 1/2 are consistent. Now check Layer 3 (the C++ engine) manually:")

    print("\n" + "=" * 80)
    print("LAYER 3 — run these against your rebuilt C++ engine and compare eval_score")
    print("to the 'NumPy(.bin)' column above (same scale, both are raw pre-sigmoid")
    print("centipawn-equivalent scores — no transform needed for comparison):")
    print("=" * 80)
    for fen in TEST_FENS:
        print(f"\nposition fen {fen}\neval")
    print(
        "\n(pipe these through your engine, e.g.:\n"
        '  echo -e "uci\\nisready\\nposition fen <FEN>\\neval\\nquit" | '
        "./build/Release/chess-engine.exe --nnue --weights=nnue_weights.bin)"
    )


if __name__ == "__main__":
    main()
