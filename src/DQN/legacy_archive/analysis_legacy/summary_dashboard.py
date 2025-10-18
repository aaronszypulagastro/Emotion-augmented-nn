# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def min_max(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    xmin, xmax = x.min(), x.max()
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax - xmin < 1e-12:
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    return (x - xmin) / (xmax - xmin)


def main():
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    csv_path = results_dir / "training_log.csv"
    if not csv_path.exists():
        print(f"[ERROR] training_log.csv nicht gefunden: {csv_path}")
        sys.exit(1)

    # Robustes CSV-Parsing: toleriert inkonsistente Zeilenanzahl
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    # Spalten robust finden (unterschiedliche Header möglich)
    lc_map = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in lc_map:
                return lc_map[c]
        return None

    col_episode = pick("episode")
    col_emotion = pick("emotion")
    col_reward  = pick("reward", "return")
    col_td      = pick("td_error", "td-err", "td")
    col_eta     = pick("eta")
    col_sigma   = pick("sigma_mean", "sigma_bar", "sigma")

    # Fallbacks
    if col_episode is None:
        df["episode"] = np.arange(1, len(df) + 1)
        col_episode = "episode"

    # Letzte 100 Episoden
    keep_cols = [c for c in [col_episode, col_emotion, col_reward, col_td, col_eta, col_sigma] if c is not None]
    sub = df[keep_cols].tail(100).rename(columns={
        col_episode: "episode",
        col_emotion or "emotion": "emotion",
        col_reward or "reward": "reward",
        col_td or "td_error": "td_error",
        col_eta or "eta": "eta",
        col_sigma or "sigma_mean": "sigma_mean",
    })

    # Korrelationen (nur vorhandene Spalten)
    corr_cols = [c for c in ["emotion", "reward", "td_error", "eta", "sigma_mean"] if c in sub.columns]
    corr_mat = sub[corr_cols].astype(float).corr()

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax11, ax12 = axes[0]
    ax21, ax22 = axes[1]

    # (1) Emotion ↔ Reward Verlauf
    if "emotion" in sub.columns and "reward" in sub.columns:
        ax11.plot(sub["episode"], sub["emotion"], color="orange", label="Emotion", linewidth=1.8)
        ax11.set_xlabel("Episode")
        ax11.set_ylabel("Emotion")
        ax11_t = ax11.twinx()
        ax11_t.plot(sub["episode"], sub["reward"], color="green", label="Reward", linewidth=1.4, alpha=0.8)
        ax11_t.set_ylabel("Reward")
        ax11.set_title("Emotion ↔ Reward (letzte 100)")
        ax11.grid(True, alpha=0.3)

    # (2) TD-Error ↔ η Verlauf
    if "td_error" in sub.columns and "eta" in sub.columns:
        ax12.plot(sub["episode"], sub["td_error"], color="red", label="TD-Error", linewidth=1.6)
        ax12.set_xlabel("Episode")
        ax12.set_ylabel("TD-Error", color="red")
        ax12_t = ax12.twinx()
        ax12_t.plot(sub["episode"], sub["eta"], color="blue", label="eta", linewidth=1.6, alpha=0.85)
        ax12_t.set_ylabel("eta", color="blue")
        ax12.set_title("TD-Error ↔ eta (letzte 100)")
        ax12.grid(True, alpha=0.3)

    # (3) σ-Aktivität (Plasticity)
    if "sigma_mean" in sub.columns:
        ax21.plot(sub["episode"], sub["sigma_mean"], color="purple", linewidth=1.6)
        ax21.set_xlabel("Episode")
        ax21.set_ylabel("sigma_mean")
        ax21.set_title("σ-Aktivität (letzte 100)")
        ax21.grid(True, alpha=0.3)
    else:
        ax21.text(0.5, 0.5, "sigma_mean nicht im Log", ha="center", va="center")
        ax21.axis("off")

    # (4) Heatmap der Korrelationen
    im = ax22.imshow(corr_mat.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax22.set_xticks(range(len(corr_cols)))
    ax22.set_yticks(range(len(corr_cols)))
    ax22.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax22.set_yticklabels(corr_cols)
    ax22.set_title("Korrelationen (letzte 100)")
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax22.text(j, i, f"{corr_mat.values[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax22, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = results_dir / "summary_dashboard.png"
    plt.savefig(out_path, dpi=200)
    print(f"[Plot] gespeichert unter: {out_path}")


if __name__ == "__main__":
    main()


