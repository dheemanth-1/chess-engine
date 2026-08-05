"""
continue_train.py
─────────────────────────────────────────────────────────────────────────────
Resumes NNUE training from a saved checkpoint (nnue_epoch10.pt) for 5 more
epochs, saving a new checkpoint after every epoch (nnue_epoch11.pt ...
nnue_epoch15.pt) and appending to the SAME training_log.json so the full
loss/metric history across all 15 epochs lives in one continuous file.

Nothing is duplicated from nnue_train.py — this file imports the model
class, dataset, loss function, and metric function directly from it.

Place this file in the same folder as nnue_train.py and run:
    python continue_train.py
"""

import glob
import json
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from nnue_train import (
    NNUE,
    collate_halfkp,
    nnue_loss,
    compute_metrics,
    export_weights,
    load_validation_slice,
    ParquetValDataset,
    ParquetStreamDataset,
)


def continue_train_parquet(
    checkpoint_path: str,
    parquet_paths,
    additional_epochs: int = 2,
    batch_size: int = 2048,
    lr: float = 1e-3,
    val_rows: int = 2_000_000,
    num_workers: int = 4,
    device_str: str = "cuda",
    log_every: int = 500,
    checkpoint_every_steps: int = 10_000,
    shuffle_buffer_size: int = 200_000,
    history_path: str = "training_log_parquet.json",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Load checkpoint ───────────────────────────────────────────────
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    start_epoch = checkpoint["epoch"]  # 2, read directly from the file — not hardcoded
    print(
        f"Resuming from epoch {start_epoch} (val_loss was {checkpoint['val_loss']:.6f})"
    )

    # ── Validation: same fixed slice from the FIRST file as the original run ──
    # Using the SAME val_rows value as before keeps this comparable to the
    # epoch1/epoch2 validation numbers already in training_log_parquet.json.
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

    # ── Training stream: first file minus the reserved val rows, plus every
    #    other file in full — identical setup to the original run. ────────
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

    # ── Rebuild model + optimizer, then load saved state ───────────────
    model = NNUE().to(device)
    model.load_state_dict(checkpoint["model_state"])

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    opt.load_state_dict(
        checkpoint["opt_state"]
    )  # restores AdamW's momentum/variance buffers

    # ReduceLROnPlateau — same settings as train_on_parquet(). Started fresh
    # here (no scheduler state was saved in the checkpoint to resume from);
    # starting back near `lr` gives the model room to keep improving, same
    # "give it room" logic the original continue_train.py used when
    # restarting CosineAnnealingLR for the CSV pipeline's extra 5 epochs.
    sched = ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3, threshold=1e-4, min_lr=lr / 100
    )

    # ── Load existing history so the JSON file stays one continuous record ──
    try:
        with open(history_path) as f:
            history = json.load(f)
        print(f"Loaded existing history ({len(history['train_loss'])} epochs so far)")
    except FileNotFoundError:
        history = {"train_loss": [], "val_loss": [], "pearson_r": [], "sign_acc": []}
        print("No existing history file found — starting a new one")

    # global_step here is LOCAL to this continuation run, used only for log
    # cadence and mid-epoch checkpoint filenames. It intentionally does NOT
    # try to resume the original run's cumulative step count (which wasn't
    # saved in the epoch-end checkpoint) — to avoid any chance of colliding
    # with filenames like nnue_step130000.pt from the original run, this
    # script's mid-epoch checkpoints use a "cont_" prefix instead (see below).
    global_step = 0

    for offset in range(1, additional_epochs + 1):
        epoch = start_epoch + offset  # 3, 4

        model.train()
        running_loss = 0.0
        steps_this_epoch = 0

        for w_flat, w_off, b_flat, b_off, stm, targets in dl_trn:
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

            running_loss += loss.item()
            steps_this_epoch += 1
            global_step += 1

            if global_step % log_every == 0:
                avg = running_loss / steps_this_epoch
                cur_lr = opt.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch} | step {global_step:>8} | train loss {avg:.6f} | lr {cur_lr:.2e}"
                )

            # Mid-epoch checkpoint + ReduceLROnPlateau check — same cadence
            # as the original run, so the scheduler gets enough check-ins
            # within an epoch to actually react (this is what was missing
            # for the regression you saw between step 170000 and epoch end
            # in the original, unscheduled run).
            if global_step % checkpoint_every_steps == 0:
                metrics = compute_metrics(model, dl_val, device)
                sched.step(metrics["val_loss"])
                cur_lr = opt.param_groups[0]["lr"]
                print(
                    f"    [checkpoint @ step {global_step}] val_loss {metrics['val_loss']:.6f} "
                    f"pearson_r {metrics['pearson_r']:.4f} sign_acc {metrics['sign_acc']*100:.2f}% "
                    f"lr {cur_lr:.2e}"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state": model.state_dict(),
                        "opt_state": opt.state_dict(),
                        "val_loss": metrics["val_loss"],
                    },
                    f"nnue_cont_step{global_step}.pt",
                )  # "cont_" prefix avoids name collisions
                model.train()

        # ── End-of-epoch metrics + checkpoint ──────────────────────────
        metrics = compute_metrics(model, dl_val, device)
        sched.step(metrics["val_loss"])
        trn_loss = running_loss / max(1, steps_this_epoch)
        print(
            f"Epoch {epoch:>2} (resumed, +{offset}/{additional_epochs}) | "
            f"train {trn_loss:.6f} | val {metrics['val_loss']:.6f} | "
            f"pearson_r {metrics['pearson_r']:.4f} | sign_acc {metrics['sign_acc']*100:.2f}% | "
            f"lr {opt.param_groups[0]['lr']:.2e}"
        )

        history["train_loss"].append(trn_loss)
        history["val_loss"].append(metrics["val_loss"])
        history["pearson_r"].append(metrics["pearson_r"])
        history["sign_acc"].append(metrics["sign_acc"])
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Correctly spelled this time — your existing checkpoint is
        # "nnue_paraquet_epoch2.pt" (typo), these new ones are "parquet".
        ckpt_path = f"nnue_parquet_epoch{epoch:02d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "val_loss": metrics["val_loss"],
            },
            ckpt_path,
        )
        print(f"  → saved {ckpt_path}")

    print("\nContinued parquet training complete.")
    return model


if __name__ == "__main__":
    parquet_paths = sorted(glob.glob(r"C:\Works\chess-engine\dataset\*.parquet"))
    print(f"Found {len(parquet_paths)} parquet files")

    model = continue_train_parquet(
        checkpoint_path="C:\\Works\\chess-engine\\nnue_parquet_epoch02.pt",
        parquet_paths=parquet_paths,
        additional_epochs=2,  # epoch 2 → epochs 3, 4
        batch_size=2048,
        lr=1e-3,
        val_rows=2_000_000,
        num_workers=4,
        device_str="cuda",
        checkpoint_every_steps=10_000,
    )
    export_weights(model, "nnue_weights_epoch4.bin")
