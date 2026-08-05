"""
nnue_train.py
─────────────────────────────────────────────────────────────────────────────
Full training pipeline for a HalfKP-based NNUE.

Architecture
    HalfKP (40 960 features, sparse) ──► FT: 256 accumulators × 2 perspectives
                                          ClippedReLU
    concat (512) ──► L1: 32  ClippedReLU
                 ──► L2: 32  ClippedReLU
                 ──► out: 1  (raw centipawn score)

Dependencies
    pip install python-chess torch pandas numpy
"""

import time
import random
import json
import math
import chess
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NUM_SQ = 64
NUM_PT = 10  # 5 piece types × 2 colours (kings excluded)
HALFKP_SIZE = NUM_SQ * NUM_PT * NUM_SQ  # 40 960 features per perspective
FT_SIZE = 256  # accumulator width
SCORE_SCALE = 400  # centipawns → win-prob sigmoid scale
CLAMP_CP = 3000  # clip extreme evals before sigmoid

# HalfKP piece-type index mapping.
# From every side's own perspective:
#   own  pieces → indices 0-4  (P=0, N=1, B=2, R=3, Q=4)
#   opp  pieces → indices 5-9  (P=5, N=6, B=7, R=8, Q=9)
_OWN_PT = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
_OPP_PT = {pt: idx + 5 for pt, idx in _OWN_PT.items()}


# ─────────────────────────────────────────────────────────────────────────────
# HalfKP Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────


def _indices_for_perspective(board: chess.Board, our_colour: chess.Color) -> list:
    """
    Compute active HalfKP feature indices for one side's perspective.

    Coordinate convention
    ─────────────────────
    White's perspective  : squares are in standard orientation  (a1=0, h8=63)
    Black's perspective  : squares are vertically mirrored via sq ^ 56
                           (a8→0, h1→63) so that black's own back rank is
                           always "rank 0" — the same spatial meaning as
                           white's rank 0.  This lets the network share
                           pattern weights across both sides.

    Feature index formula
    ─────────────────────
    idx = king_sq * (NUM_PT * NUM_SQ)   # 64 buckets for the king
        + pt_idx  *  NUM_SQ             # 10 piece-type slots per bucket
        + piece_sq                      # 64 squares per slot
    → range [0, 40 960)
    """
    king_sq = board.king(our_colour)
    assert king_sq is not None, f"King of color {our_colour} is missing from the board"
    if our_colour == chess.BLACK:
        king_sq ^= 56  # mirror rank for black's perspective

    indices = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type == chess.KING:
            continue  # skip empty squares and kings

        # Choose own-piece or opp-piece encoding
        pt_map = _OWN_PT if piece.color == our_colour else _OPP_PT
        pt_idx = pt_map[piece.piece_type]

        piece_sq = sq if our_colour == chess.WHITE else sq ^ 56

        idx = king_sq * (NUM_PT * NUM_SQ) + pt_idx * NUM_SQ + piece_sq
        indices.append(idx)

    return indices


def halfkp_features(fen: str):
    """
    Parse a FEN and return the active HalfKP feature indices for both sides.

    Returns
    ───────
    white_indices : list[int]  active features from white's perspective
    black_indices : list[int]  active features from black's perspective
    stm           : int        1 = white to move, 0 = black to move
    """
    board = chess.Board(fen)
    w_idx = _indices_for_perspective(board, chess.WHITE)
    b_idx = _indices_for_perspective(board, chess.BLACK)
    stm = 1 if board.turn == chess.WHITE else 0
    return w_idx, b_idx, stm


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Parsing
# ─────────────────────────────────────────────────────────────────────────────


def parse_eval(raw) -> float:
    """
    Convert a raw evaluation value to a float centipawn score.

    Handles
    ───────
    '+56'   →   56.0
    '-26'   →  -26.0
    '0'     →    0.0
    '#3'    → +CLAMP_CP   (forced mate for the winning side)
    '#-3'   → -CLAMP_CP   (forced mate against)
    Anything unparsable → 0.0
    """
    s = str(raw).strip()
    if "#" in s:
        return float(CLAMP_CP) if "-" not in s.replace("#", "", 1) else float(-CLAMP_CP)
    try:
        return float(s.replace("+", ""))
    except ValueError:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# NEW: score parsing for the parquet dataset (numeric cp/mate columns)
# ─────────────────────────────────────────────────────────────────────────────
def parse_score(cp, mate) -> float:
    """
    Combines the 'cp' and 'mate' columns into one centipawn-equivalent score,
    exactly like parse_eval() did for the old CSV's combined string field.

    This dataset stores exactly one of the two per row:
        cp set, mate null   → normal evaluation in centipawns
        cp null, mate set   → forced mate in N moves

    Forced-mate positions are converted to ±CLAMP_CP. The model can't
    meaningfully distinguish "mate in 3" from "mate in 15" through static
    evaluation anyway, and these get dropped by passes_filters() below
    regardless (same as the old dataset's "#3"/"#-7" mate strings did).

    ⚠ Assumes mate>0 means good for whoever 'cp' being positive means good
    for in this dataset (i.e. the SAME convention as cp). Verify this with
    the smoke test before trusting it on the full run.
    """
    if not pd.isna(mate):
        return float(CLAMP_CP) if mate > 0 else float(-CLAMP_CP)
    if pd.isna(cp):
        return 0.0
    return float(cp)


def passes_filters(raw_score: float) -> bool:
    """
    Single source of truth for which positions are excluded from training.
    Used identically by both the validation set and the streaming training
    set so train/val are filtered the same way.

    Drops:
      - exact-zero evals (opening-book/contempt artifacts) — ONLY this now.

    The |eval| > 1500 upper bound has been REMOVED. Real games constantly
    pass through decisive positions — after any blunder, sacrifice, or
    simplification — and a model that's never been trained on a single
    clearly-winning or clearly-losing position has no idea what to do once
    a real game reaches one. This was the most likely cause of the
    catastrophic cutechess results (engine losing ~90% of games, average
    game length dropping to ~33 moves) despite reasonable-looking val
    metrics: the model was only ever being scored on the easy half of the
    distribution it was trained on.

    Magnitude is still controlled — just via clamping in parse_score()
    (then natural sigmoid saturation) instead of outright exclusion.
    """
    if raw_score == 0.0:
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# NEW: validation slice loader + Dataset for the parquet schema
# ─────────────────────────────────────────────────────────────────────────────


def load_validation_slice(parquet_path: str, n_rows: int) -> pd.DataFrame:
    """
    Reads only the first n_rows from a parquet file WITHOUT loading the
    whole file into memory — stops reading row-groups as soon as enough
    rows are collected. Safe even on a 31M-row file when you only want 2M.
    """
    pf = pq.ParquetFile(parquet_path)
    chunks, collected = [], 0
    for batch in pf.iter_batches(batch_size=100_000, columns=["fen", "cp", "mate"]):
        chunks.append(batch.to_pandas())
        collected += len(batch)
        if collected >= n_rows:
            break
    df = pd.concat(chunks, ignore_index=True)
    return df.iloc[:n_rows].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


class HalfKPDataset(Dataset):
    """
    Reads FEN strings and centipawn evaluations.

    HalfKP features are computed on-the-fly inside __getitem__ so that
    DataLoader worker processes can parallelise the (CPU-bound) FEN parsing
    while the GPU handles the previous batch.

    The sigmoid win-probability target is pre-computed once during __init__
    so it is never repeated per epoch.
    """

    def __init__(self, df: pd.DataFrame):
        self.fens = df["FEN"].to_numpy()

        raw_evals = np.array(
            [parse_eval(e) for e in df["Evaluation"]], dtype=np.float32
        )
        raw_evals = np.clip(raw_evals, -CLAMP_CP, CLAMP_CP)
        stm_is_black = (df["FEN"].str.split().str[1] == "b").to_numpy()
        raw_evals[stm_is_black] *= -1.0
        # Pre-compute sigmoid win-probability labels  σ(cp / scale)
        # Stored as float32 once here → never recomputed during training.
        self.targets = (1.0 / (1.0 + np.exp(-raw_evals / SCORE_SCALE))).astype(
            np.float32
        )

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, idx: int):
        w_idx, b_idx, stm = halfkp_features(to_fen_str(self.fens[idx]))
        return (
            torch.tensor(w_idx, dtype=torch.long),  # variable length
            torch.tensor(b_idx, dtype=torch.long),  # variable length
            torch.tensor(stm, dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


def collate_halfkp(batch):
    """
    Custom collate for variable-length HalfKP index lists.

    nn.EmbeddingBag requires:
      indices : 1-D tensor – all active feature indices across the batch,
                             concatenated one sample after another.
      offsets : 1-D tensor – where each sample starts inside `indices`.

    This mirrors exactly what EmbeddingBag.forward() expects and avoids
    any padding overhead.
    """
    w_idx_list, b_idx_list, stm_list, tgt_list = zip(*batch)

    def pack(idx_list):
        offsets = torch.zeros(len(idx_list), dtype=torch.long)
        cursor = 0
        for i, t in enumerate(idx_list):
            offsets[i] = cursor
            cursor += t.numel()
        flat = torch.cat(idx_list)  # 1-D concatenation
        return flat, offsets

    w_flat, w_off = pack(w_idx_list)
    b_flat, b_off = pack(b_idx_list)
    stm = torch.stack(stm_list)
    targets = torch.stack(tgt_list)

    return w_flat, w_off, b_flat, b_off, stm, targets


class ParquetValDataset(Dataset):
    """
    Validation counterpart to HalfKPDataset for the new fen/cp/mate schema.
    Built from a small in-memory slice (see load_validation_slice), so a
    plain Dataset (not streaming) is fine here.
    """

    def __init__(self, df: pd.DataFrame):
        raw_scores = np.array(
            [parse_score(cp, mate) for cp, mate in zip(df["cp"], df["mate"])],
            dtype=np.float32,
        )
        keep_mask = np.array([passes_filters(s) for s in raw_scores])

        fens = df["fen"].to_numpy()[keep_mask]
        fens = np.array([to_fen_str(f) for f in fens])
        raw_scores = raw_scores[keep_mask]

        stm_is_black = np.array([f.split(" ")[1] == "b" for f in fens])
        raw_scores[stm_is_black] *= -1.0

        self.fens = fens
        self.targets = (1.0 / (1.0 + np.exp(-raw_scores / SCORE_SCALE))).astype(
            np.float32
        )

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        w_idx, b_idx, stm = halfkp_features(self.fens[idx].item())
        return (
            torch.tensor(w_idx, dtype=torch.long),
            torch.tensor(b_idx, dtype=torch.long),
            torch.tensor(stm, dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


class ParquetStreamDataset(IterableDataset):
    """
    Streams positions from a list of parquet files without ever holding the
    full dataset in memory — required because 316M rows of FEN strings alone
    would be tens of GB as a Python object array.

    Multi-worker sharding: with DataLoader(num_workers=N), each worker gets
    a disjoint round-robin subset of files, so positions are never duplicated
    and workers never read the same data twice in one epoch.

    Shuffle buffer: parquet rows are read in file/row-group order, which can
    leave nearby rows correlated. A reservoir-style buffer collects
    shuffle_buffer_size samples, shuffles them in memory, then yields them —
    the streaming equivalent of the df.sample(frac=1) shuffle used for CSV.
    """

    def __init__(
        self,
        parquet_paths,
        skip_rows_in_first_file=0,
        shuffle_buffer_size=200_000,
        batch_rows=50_000,
    ):
        super().__init__()
        self.parquet_paths = list(parquet_paths)
        self.skip_rows_in_first_file = skip_rows_in_first_file
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_rows = batch_rows

    def _files_for_this_worker(self):
        info = get_worker_info()
        if info is None:
            return list(enumerate(self.parquet_paths))
        return [
            (i, p)
            for i, p in enumerate(self.parquet_paths)
            if i % info.num_workers == info.id
        ]

    def _raw_samples(self):
        for file_idx, path in self._files_for_this_worker():
            pf = pq.ParquetFile(path)
            rows_seen = 0
            for batch in pf.iter_batches(
                batch_size=self.batch_rows, columns=["fen", "cp", "mate"]
            ):
                df_batch = batch.to_pandas()

                # Validation rows are always reserved from the START of the
                # FIRST file in your parquet_paths list — skip them here.
                if file_idx == 0 and rows_seen < self.skip_rows_in_first_file:
                    skip_n = self.skip_rows_in_first_file - rows_seen
                    df_batch = df_batch.iloc[skip_n:]
                rows_seen += len(batch)
                if df_batch.empty:
                    continue

                for fen, cp, mate in zip(
                    df_batch["fen"], df_batch["cp"], df_batch["mate"]
                ):
                    raw_score = parse_score(cp, mate)
                    if not passes_filters(raw_score):
                        continue

                    fen_str = to_fen_str(fen)  # normalize once, reuse below
                    if fen_str.split(" ")[1] == "b":
                        raw_score = -raw_score

                    target = 1.0 / (1.0 + math.exp(-raw_score / SCORE_SCALE))
                    w_idx, b_idx, stm = halfkp_features(fen_str)

                    yield (
                        torch.tensor(w_idx, dtype=torch.long),
                        torch.tensor(b_idx, dtype=torch.long),
                        torch.tensor(float(stm), dtype=torch.float32),
                        torch.tensor(target, dtype=torch.float32),
                    )

    def __iter__(self):
        buffer = []
        for sample in self._raw_samples():
            buffer.append(sample)
            if len(buffer) >= self.shuffle_buffer_size:
                random.shuffle(buffer)
                yield from buffer
                buffer = []
        if buffer:
            random.shuffle(buffer)
            yield from buffer


# ─────────────────────────────────────────────────────────────────────────────
# NNUE Architecture
# ─────────────────────────────────────────────────────────────────────────────


class NNUE(nn.Module):
    """
    HalfKP NNUE with one shared feature transformer and two hidden layers.

    Forward pass (per sample)
    ─────────────────────────
    1. Feature transformer (FT)
       white_acc = Σ W_ft[i] for i in white_active_features + b_ft   → (256,)
       black_acc = Σ W_ft[i] for i in black_active_features + b_ft   → (256,)
       Both are then clamped to [0, 1]  (Clipped ReLU).

       The same weight matrix W_ft is used for both perspectives — this is
       weight sharing.  It works because the black perspective is already
       coordinate-mirrored in the feature extractor, so the spatial meaning
       of every feature index is consistent across sides.

    2. Accumulator concatenation
       The side-to-move (STM) accumulator is placed first; NSTM second.
       This asymmetry gives the network a single consistent input structure
       regardless of which colour is on move.
       concat  →  (512,)

    3. Hidden layers
       L1: 512 → 32  ClippedReLU
       L2:  32 → 32  ClippedReLU
       out:  32 → 1  (raw score in centipawn-like units)
    """

    def __init__(self, ft_size: int = FT_SIZE):
        super().__init__()

        # ── Feature transformer ───────────────────────────────────────
        # EmbeddingBag(num_embeddings, embedding_dim, mode='sum') computes:
        #   output[i] = embedding_weight[ indices[offsets[i]:offsets[i+1]] ].sum(0)
        # This is exactly the NNUE first-layer accumulation.
        self.ft = nn.EmbeddingBag(HALFKP_SIZE, ft_size, mode="sum", sparse=False)
        self.ft_b = nn.Parameter(torch.zeros(ft_size))

        # ── Hidden layers ─────────────────────────────────────────────
        self.l1 = nn.Linear(2 * ft_size, 32)
        self.l2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 1)

        self._init_weights()

    def _init_weights(self):
        # Small uniform init for the large sparse embedding matrix
        nn.init.uniform_(self.ft.weight, -0.01, 0.01)
        # Kaiming init for the dense layers (accounts for ClippedReLU ≈ ReLU)
        for layer in (self.l1, self.l2, self.out):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    @staticmethod
    def crelu(x: torch.Tensor) -> torch.Tensor:
        """Clipped ReLU: clamp activations to [0, 1].
        The upper bound is required for later int8 quantization."""
        return x.clamp(0.0, 1.0)

    def forward(
        self,
        w_flat: torch.Tensor,  # (total_white_features_in_batch,)  long
        w_off: torch.Tensor,  # (batch_size,)                     long
        b_flat: torch.Tensor,  # (total_black_features_in_batch,)  long
        b_off: torch.Tensor,  # (batch_size,)                     long
        stm: torch.Tensor,  # (batch_size,)                     float  1=white 0=black
    ) -> torch.Tensor:  # (batch_size, 1)

        # ── Step 1 : accumulate both perspectives ────────────────────
        w_acc = self.crelu(self.ft(w_flat, w_off) + self.ft_b)  # (B, 256)
        b_acc = self.crelu(self.ft(b_flat, b_off) + self.ft_b)  # (B, 256)

        # ── Step 2 : side-to-move perspective always goes first ──────
        #   stm=1 (white to move) → [white_acc | black_acc]
        #   stm=0 (black to move) → [black_acc | white_acc]
        #   We implement this without an if-branch so it batches cleanly.
        stm = stm.view(-1, 1)  # (B, 1)
        stm_a = stm * w_acc + (1.0 - stm) * b_acc  # (B, 256)
        nstm_a = stm * b_acc + (1.0 - stm) * w_acc  # (B, 256)
        x = torch.cat([stm_a, nstm_a], dim=1)  # (B, 512)

        # ── Step 3 : hidden layers ───────────────────────────────────
        x = self.crelu(self.l1(x))  # (B, 32)
        x = self.crelu(self.l2(x))  # (B, 32)
        return self.out(x)  # (B, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Loss Function
# ─────────────────────────────────────────────────────────────────────────────


def nnue_loss(
    pred: torch.Tensor,  # (B, 1)  raw network output
    target: torch.Tensor,  # (B,)    pre-computed σ(cp/scale)  ∈ [0,1]
    scale: float = SCORE_SCALE,
) -> torch.Tensor:
    """
    MSE between σ(pred/scale) and the pre-computed win-probability target.

    Both sides of the MSE are in win-probability space [0,1], which keeps
    the gradient well-conditioned regardless of the raw centipawn magnitude.

    If you later add game-result labels (WDL), the loss becomes:
        L = λ · MSE(σ(pred/scale), σ(engine_cp/scale))
          + (1-λ) · MSE(σ(pred/scale), game_result)
    with λ ≈ 0.7.  For now we only have engine evals, so λ = 1.
    """
    pred_prob = torch.sigmoid(pred.squeeze(1) / scale)
    return F.mse_loss(pred_prob, target)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_metrics(model, dataloader, device):
    """
    Returns a dict with:
      val_loss    : mean MSE loss over the validation set
      pearson_r   : Pearson correlation between σ(pred/scale) and target
      sign_acc    : fraction of positions where sign(pred) == sign(target - 0.5)
    """
    model.eval()
    all_pred, all_tgt = [], []

    for w_flat, w_off, b_flat, b_off, stm, targets in dataloader:
        w_flat, w_off = w_flat.to(device), w_off.to(device)
        b_flat, b_off = b_flat.to(device), b_off.to(device)
        stm, targets = stm.to(device), targets.to(device)

        pred = model(w_flat, w_off, b_flat, b_off, stm)
        pred_prob = torch.sigmoid(pred.squeeze(1) / SCORE_SCALE)

        all_pred.append(pred_prob.cpu())
        all_tgt.append(targets.cpu())

    p = torch.cat(all_pred)
    t = torch.cat(all_tgt)

    val_loss = F.mse_loss(p, t).item()

    # Pearson correlation
    p_mean, t_mean = p.mean(), t.mean()
    cov = ((p - p_mean) * (t - t_mean)).mean()
    pearsonr = (cov / (p.std() * t.std() + 1e-8)).item()

    # Sign accuracy: does the model agree on which side is better?
    sign_acc = ((p > 0.5) == (t > 0.5)).float().mean().item()

    return {"val_loss": val_loss, "pearson_r": pearsonr, "sign_acc": sign_acc}


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────


def train(
    csv_path: str,
    epochs: int = 10,
    batch_size: int = 2048,
    lr: float = 1e-3,
    val_split: float = 0.01,
    num_workers: int = 4,
    device_str: str = "cuda",
    log_every: int = 500,  # print progress every N steps
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Load & split ─────────────────────────────────────────────────
    print("Loading CSV…")
    df = pd.read_csv(csv_path)

    # Filtering rows with extreme evals or evals which equal to 0.0 Let's see if this helps the model train better.
    parsed = df["Evaluation"].apply(parse_eval)
    mask = (parsed != 0.0) & (parsed.abs() <= 1500)
    df = df[mask].reset_index(drop=True)
    print(f"Rows after filtering: {len(df):,}  (removed {(~mask).sum():,} positions)")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_val = max(1, int(len(df) * val_split))
    df_val = df.iloc[:n_val].reset_index(drop=True)
    df_trn = df.iloc[n_val:].reset_index(drop=True)
    print(f"Train: {len(df_trn):,}   Val: {len(df_val):,}")

    ds_trn = HalfKPDataset(df_trn)
    ds_val = HalfKPDataset(df_val)

    dl_trn = DataLoader(
        ds_trn,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_halfkp,
        num_workers=num_workers,
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_halfkp,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ── Model / optimiser / scheduler ───────────────────────────────
    model = NNUE().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 100)

    total_params = sum(p.numel() for p in model.parameters())
    ft_params = model.ft.weight.numel()
    print(
        f"Total params : {total_params:,}   (FT: {ft_params:,}  dense: {total_params - ft_params:,})"
    )
    history = {"train_loss": [], "val_loss": [], "pearson_r": [], "sign_acc": []}
    # ── Epoch loop ───────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for step, (w_flat, w_off, b_flat, b_off, stm, targets) in enumerate(dl_trn, 1):
            w_flat = w_flat.to(device, non_blocking=True)
            w_off = w_off.to(device, non_blocking=True)
            b_flat = b_flat.to(device, non_blocking=True)
            b_off = b_off.to(device, non_blocking=True)
            stm = stm.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            opt.zero_grad()
            pred = model(w_flat, w_off, b_flat, b_off, stm)
            loss = nnue_loss(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            running_loss += loss.item()
            if step % log_every == 0:
                avg = running_loss / step
                print(f"  Epoch {epoch} | step {step:>6} | train loss {avg:.6f}")

        sched.step()

        # ── Validation metrics ───────────────────────────────────────
        metrics = compute_metrics(model, dl_val, device)
        trn_loss = running_loss / len(dl_trn)
        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"train {trn_loss:.6f} | "
            f"val {metrics['val_loss']:.6f} | "
            f"pearson_r {metrics['pearson_r']:.4f} | "
            f"sign_acc {metrics['sign_acc']*100:.2f}% | "
            f"lr {sched.get_last_lr()[0]:.2e}"
        )
        history["train_loss"].append(trn_loss)
        history["val_loss"].append(metrics["val_loss"])
        history["pearson_r"].append(metrics["pearson_r"])
        history["sign_acc"].append(metrics["sign_acc"])

        # Save history alongside checkpoint — ADD this after torch.save(...)

        with open("training_log.json", "w") as f:
            json.dump(history, f, indent=2)

        # ── Save checkpoint ──────────────────────────────────────────
        ckpt_path = f"nnue_epoch{epoch:02d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "val_loss": metrics["val_loss"],
            },
            ckpt_path,
        )
        print(f"  → saved {ckpt_path}")

    print("\nTraining complete.")
    return model


def train_on_parquet(
    parquet_paths,
    epochs: int = 2,
    batch_size: int = 2048,
    lr: float = 1e-3,
    val_rows: int = 2_000_000,
    num_workers: int = 4,
    device_str: str = "cuda",
    log_every: int = 500,
    checkpoint_every_steps: int = 10_000,
    shuffle_buffer_size: int = 200_000,
    init_checkpoint: str = "",  # optional: warm-start from a previous .pt
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ── Validation: fixed slice from the FIRST file, small enough for RAM ──
    print(f"Loading {val_rows:,}-row validation slice from {parquet_paths[0]}…")
    df_val = load_validation_slice(parquet_paths[0], val_rows)
    ds_val = ParquetValDataset(df_val)
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_halfkp,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"Validation set: {len(ds_val):,} positions after filtering")

    # ── Training stream: same first file MINUS the validation rows, plus
    #    every other file in full. IterableDataset does NOT support
    #    shuffle=True in DataLoader — shuffling is handled internally by
    #    ParquetStreamDataset's reservoir buffer instead. ──────────────────
    ds_trn = ParquetStreamDataset(
        parquet_paths,
        skip_rows_in_first_file=val_rows,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    dl_trn = DataLoader(
        ds_trn,
        batch_size=batch_size,
        collate_fn=collate_halfkp,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = NNUE().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    start_epoch = 0
    if init_checkpoint:
        print(f"Warm-starting from {init_checkpoint}")
        ckpt = torch.load(init_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])

    # NOTE: no LR scheduler here. With an IterableDataset, total step count
    # per epoch isn't known in advance (it depends on how many rows survive
    # passes_filters()), so a fixed-T_max cosine schedule isn't directly
    # applicable. AdamW's own adaptivity covers most of this. If you want
    # decay later, ReduceLROnPlateau driven by the periodic val_loss checks
    # below is the natural fit — ask if you want that added.
    sched = ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3, threshold=1e-4, min_lr=lr / 100
    )
    history = {"train_loss": [], "val_loss": [], "pearson_r": [], "sign_acc": []}
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        steps_this_epoch = 0

        # data_wait_time accumulates time spent blocked waiting for the
        # DataLoader to hand over a batch (CPU-bound: feature extraction,
        # parquet reads, collation). compute_time accumulates time spent
        # actually on the GPU (forward/backward/optimizer step). Comparing
        # the two tells you definitively which one is the bottleneck —
        # no need to guess from nvidia-smi's noisy, shared snapshot.
        data_wait_time = 0.0
        compute_time = 0.0
        t_prev = time.perf_counter()

        for w_flat, w_off, b_flat, b_off, stm, targets in dl_trn:
            t_got_batch = time.perf_counter()
            data_wait_time += t_got_batch - t_prev

            w_flat, w_off = w_flat.to(device, non_blocking=True), w_off.to(
                device, non_blocking=True
            )
            b_flat, b_off = b_flat.to(device, non_blocking=True), b_off.to(
                device, non_blocking=True
            )
            stm, targets = stm.to(device, non_blocking=True), targets.to(
                device, non_blocking=True
            )

            opt.zero_grad()
            pred = model(w_flat, w_off, b_flat, b_off, stm)
            loss = nnue_loss(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize(
                    device
                )  # required for an honest compute-time measurement

            t_after_compute = time.perf_counter()
            compute_time += t_after_compute - t_got_batch
            t_prev = t_after_compute

            running_loss += loss.item()
            steps_this_epoch += 1
            global_step += 1

            if global_step % log_every == 0:
                avg = running_loss / steps_this_epoch
                wait_pct = (
                    100.0 * data_wait_time / max(1e-9, data_wait_time + compute_time)
                )
                if device.type == "cuda":
                    alloc = torch.cuda.memory_allocated(device) / 1e6
                    peak = torch.cuda.max_memory_allocated(device) / 1e6
                    print(
                        f"  Epoch {epoch} | step {global_step:>8} | train loss {avg:.6f} | "
                        f"GPU mem alloc {alloc:.0f}MB (peak {peak:.0f}MB) | "
                        f"data-wait {wait_pct:.0f}% of time"
                    )
                else:
                    print(
                        f"  Epoch {epoch} | step {global_step:>8} | train loss {avg:.6f}"
                    )

            # Mid-epoch checkpoint — one epoch over 316M rows can take many
            # hours; don't wait that long to find out something's wrong.
            if global_step % checkpoint_every_steps == 0:
                metrics = compute_metrics(model, dl_val, device)
                print(
                    f"    [checkpoint @ step {global_step}] val_loss {metrics['val_loss']:.6f} "
                    f"pearson_r {metrics['pearson_r']:.4f} sign_acc {metrics['sign_acc']*100:.2f}%"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state": model.state_dict(),
                        "opt_state": opt.state_dict(),
                        "val_loss": metrics["val_loss"],
                    },
                    f"nnue_step{global_step}.pt",
                )
                model.train()

        metrics = compute_metrics(model, dl_val, device)
        trn_loss = running_loss / max(1, steps_this_epoch)
        print(
            f"Epoch {epoch:>2}/{epochs} | train {trn_loss:.6f} | val {metrics['val_loss']:.6f} | "
            f"pearson_r {metrics['pearson_r']:.4f} | sign_acc {metrics['sign_acc']*100:.2f}%"
        )

        history["train_loss"].append(trn_loss)
        history["val_loss"].append(metrics["val_loss"])
        history["pearson_r"].append(metrics["pearson_r"])
        history["sign_acc"].append(metrics["sign_acc"])
        with open("training_log_parquet.json", "w") as f:
            json.dump(history, f, indent=2)

        ckpt_path = f"nnue_parquet_epoch{epoch:02d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "val_loss": metrics["val_loss"],
            },
            ckpt_path,
        )
        print(f"  → saved {ckpt_path}")

    print("\nParquet training complete.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Weight Export  (for C++ inference)
# ─────────────────────────────────────────────────────────────────────────────


def export_weights(model: NNUE, path: str = "nnue_weights.bin"):
    """
    Dump all network weights to a flat binary file in the exact order that
    the C++ inference code will read them back.

    Layout (all float32, little-endian):
        ft.weight   [40960, 256]   →  10 485 760 floats
        ft_b        [256]          →        256 floats
        l1.weight   [32,   512]    →     16 384 floats
        l1.bias     [32]           →         32 floats
        l2.weight   [32,    32]    →      1 024 floats
        l2.bias     [32]           →         32 floats
        out.weight  [1,     32]    →         32 floats
        out.bias    [1]            →          1 float
    """
    import struct

    sd = model.state_dict()
    order = [
        "ft.weight",
        "ft_b",
        "l1.weight",
        "l1.bias",
        "l2.weight",
        "l2.bias",
        "out.weight",
        "out.bias",
    ]
    with open(path, "wb") as f:
        for key in order:
            arr = sd[key].cpu().numpy().astype(np.float32)
            f.write(arr.tobytes())
    print(f"Exported weights → {path}  ({sum(sd[k].numel() for k in order):,} floats)")


_PIECE_VALUES = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}


def to_fen_str(x) -> str:
    """
    Safely normalizes a single value pulled out of a numpy object array (or
    a pandas Series) into a real Python str, regardless of whether the
    underlying parquet column was stored as Arrow `string` (→ already str
    at runtime) or `binary` (→ bytes at runtime).

    Blindly calling str(x) is WRONG for the bytes case: str(b'e4') produces
    the literal text "b'e4'" (the repr, including the b-prefix and quotes),
    not a decoded string. This function decodes bytes properly instead.

    Check which case you're actually in with:
        print(type(df['fen'].iloc[0]))
    """
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _material_balance(fen: str) -> int:
    """
    Quick material count from the FEN's piece-placement field — White minus
    Black, in pawn units. NOT used for training; only for sanity-checking
    the dataset's sign convention below. A correctly-signed STM-relative
    score should correlate positively with material advantage for whoever
    is actually to move.
    """
    placement = fen.split(" ")[0]
    balance = 0
    for ch in placement:
        if ch.isalpha():
            v = _PIECE_VALUES.get(ch.upper(), 0)
            balance += v if ch.isupper() else -v
    return balance


def smoke_test(parquet_paths, n_rows: int = 300_000, check_n: int = 20_000):
    """
    Fast, no-GPU-required sanity check on the data pipeline. Run this BEFORE
    committing to the full multi-hour training job — it answers three things
    in under a minute:

      1. Is the 'fen' column str or bytes in this actual file? (answers the
         to_fen_str question directly, with real evidence from your data)
      2. Does filtering behave sensibly (not dropping everything / nothing)?
      3. Is the sign convention correct? This is checked by correlating a
         simple material count (computed independently from the FEN itself,
         with no dependency on cp/mate at all) against the final STM-relative
         training target. A real position where one side is up a queen
         should show up as a clear positive correlation; if the correlation
         comes out strongly NEGATIVE, the sign handling is inverted — the
         same class of bug that broke the very first CSV training run.
    """
    print("=" * 70)
    print("SMOKE TEST — verifying dataset format and sign convention")
    print("=" * 70)

    print(f"\n[1/4] Loading {n_rows:,} rows from {parquet_paths[0]} ...")
    df = load_validation_slice(parquet_paths[0], n_rows)
    print(f"      Loaded {len(df):,} raw rows")

    print(f"\n[2/4] Checking 'fen' column type...")
    sample_fen = df["fen"].iloc[0]
    print(f"      type(df['fen'].iloc[0]) = {type(sample_fen)}")
    if isinstance(sample_fen, bytes):
        print("      → bytes detected — to_fen_str() will decode these correctly.")
    else:
        print("      → already str — to_fen_str() is a safe no-op here.")

    print(f"\n[3/4] Building ParquetValDataset (applies filtering + sign flip)...")
    ds = ParquetValDataset(df)
    dropped = len(df) - len(ds)
    print(f"      {len(ds):,} positions kept, {dropped:,} dropped by passes_filters()")
    if len(ds) == 0:
        print(
            "      ⚠ Nothing survived filtering — check passes_filters()/parse_score()."
        )
        print("=" * 70)
        return None

    print(f"\n[4/4] Checking sign convention against an independent material count...")
    n = min(check_n, len(ds))
    idxs = np.random.choice(len(ds), size=n, replace=False)

    signed_material, targets = [], []
    for i in idxs:
        fen = to_fen_str(ds.fens[i])
        stm_white = fen.split(" ")[1] == "w"
        mat = _material_balance(fen)  # white - black, pawns
        signed_material.append(mat if stm_white else -mat)  # from STM's own POV
        targets.append(ds.targets[i])

    signed_material = np.array(signed_material, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    corr = float(np.corrcoef(signed_material, targets)[0, 1])
    print(f"      Correlation(material-from-STM-perspective, target) = {corr:+.3f}")

    print(
        f"\n      Example positions (largest |material| first, for an eyeball check):"
    )
    order = np.argsort(-np.abs(signed_material))[:5]
    for j in order:
        i = idxs[j]
        fen = to_fen_str(ds.fens[i])
        stm_char = fen.split(" ")[1]
        print(
            f"        stm={stm_char}  material_from_stm={signed_material[j]:+.0f}  "
            f"target={targets[j]:.3f}  fen={fen[:45]}..."
        )

    print("\n" + "=" * 70)
    if corr > 0.3:
        print(
            f"PASS — positive correlation ({corr:+.3f}). Sign convention looks correct."
        )
        print("Safe to proceed with the full training run.")
    elif corr < -0.3:
        print(
            f"FAIL — NEGATIVE correlation ({corr:+.3f}). Sign convention is likely INVERTED."
        )
        print(
            "Do NOT start the full job. Re-check parse_score()'s mate-sign assumption"
        )
        print("and the sign-flip logic before proceeding.")
    else:
        print(
            f"INCONCLUSIVE — weak correlation ({corr:+.3f}). Material alone may not be a"
        )
        print(
            "strong enough signal in this sample. Eyeball the example positions above,"
        )
        print("or rerun with a larger n_rows/check_n, before trusting this either way.")
    print("=" * 70)

    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob

    parquet_paths = sorted(glob.glob(r"C:\Works\chess-engine\dataset\*.parquet"))
    print(f"Found {len(parquet_paths)} parquet files")

    # ── SMOKE TEST FIRST — uncomment this block and run it before the full
    #    job. Takes a few minutes on a few hundred thousand rows and will
    #    immediately reveal a sign-convention bug via sign_acc stuck ~50%,
    #    exactly like the very first broken CSV run did.

    smoke_result = smoke_test(parquet_paths, n_rows=300_000, check_n=20_000)

    proceed = input("\nProceed with the full training run? [y/N] ").strip().lower()
    if proceed != "y":
        print("Stopped after smoke test — re-run with 'y' once you're satisfied.")
        raise SystemExit(0)

    model = train_on_parquet(
        parquet_paths=parquet_paths,
        epochs=2,  # see explanation: ~2.4x your entire
        # previous 10-epoch run in just 1 pass
        batch_size=2048,
        lr=1e-3,
        val_rows=2_000_000,
        num_workers=4,
        device_str="cuda",
        checkpoint_every_steps=10_000,  # roughly every 1-2 hours on an MX450
        # init_checkpoint='nnue_epoch15.pt',   # uncomment to warm-start
    )
    export_weights(model, "nnue_weights_v2.bin")
