# ─────────────────────────────────────────────────────────────
# Cell — plot Elo progression across match runs
# Run this any time after accumulating rows in match_summary.csv
# (i.e. after running cutechess + parse_match_results.py multiple
# times, once per engine version you want to compare)
# ─────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("match_summary.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# ── Elo diff with error bars per version ────────────────────────
axes[0].errorbar(
    df["version"],
    df["elo_diff"],
    yerr=df["elo_margin_95"],
    fmt="o-",
    capsize=5,
    color="steelblue",
)
axes[0].axhline(0, color="gray", linestyle="--", label="Equal strength")
axes[0].set_title("NNUE Elo difference vs PST, by version")
axes[0].set_ylabel("Elo (positive = NNUE stronger)")
axes[0].tick_params(axis="x", rotation=30)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ── Win/Loss/Draw stacked bar per version ───────────────────────
axes[1].bar(df["version"], df["wins"], label="NNUE wins", color="seagreen")
axes[1].bar(df["version"], df["draws"], bottom=df["wins"], label="Draws", color="gray")
axes[1].bar(
    df["version"],
    df["losses"],
    bottom=df["wins"] + df["draws"],
    label="NNUE losses",
    color="indianred",
)
axes[1].set_title("Game outcomes by version")
axes[1].set_ylabel("Games")
axes[1].tick_params(axis="x", rotation=30)
axes[1].legend()

plt.tight_layout()
plt.savefig("elo_progression_v2.png", dpi=120)
plt.show()

print(df[["version", "wins", "losses", "draws", "elo_diff", "elo_margin_95"]])
