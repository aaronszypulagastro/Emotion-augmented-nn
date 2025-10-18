# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def min_max_norm(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    xmin, xmax = x.min(), x.max()
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax - xmin < 1e-12:
        return pd.Series(np.zeros_like(x, dtype=float), index=x.index)
    return (x - xmin) / (xmax - xmin)


def main():
    # Root so setzen, dass wir relative Pfade robust finden
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    results_dir = project_root / "results"
    csv_path = results_dir / "training_log.csv"

    if not csv_path.exists():
        print(f"[ERROR] CSV nicht gefunden: {csv_path}")
        sys.exit(1)

    # CSV laden (robust gegen zusätzliche/spätere Spalten)
    # Robustes CSV-Parsing: toleriert inkonsistente Zeilenanzahl
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    # Erwartete Spaltennamen prüfen und ggf. Alternativen mappen
    # Wir benötigen nur: episode, emotion, td_error, eta
    col_map = {
        "episode": None,
        "emotion": None,
        "td_error": None,
        "eta": None,
    }
    for c in df.columns:
        lc = c.strip().lower()
        if lc in col_map:
            col_map[lc] = c

    missing = [k for k, v in col_map.items() if v is None]
    if missing:
        print(f"[WARN] Fehlende Spalten im CSV: {missing}. Verfügbare: {list(df.columns)}")
        # Versuche Best-Guess (z. B. 'return' vs 'reward', etc.)
        # td_error/eta/emotion sind kritisch; ohne sie macht die Analyse wenig Sinn
        for k in missing:
            if k == "episode" and "episode" not in df.columns:
                # Fallback: Index + 1
                df["episode"] = np.arange(1, len(df) + 1)
                col_map["episode"] = "episode"

    # Reduziere auf benötigte Spalten und tail(100)
    try:
        sub = df[[col_map["episode"], col_map["emotion"], col_map["td_error"], col_map["eta"]]].copy()
    except Exception as e:
        print(f"[ERROR] Benötigte Spalten nicht gefunden: {e}\nSpalten im CSV: {list(df.columns)}")
        sys.exit(1)

    sub.columns = ["episode", "emotion", "td_error", "eta"]
    sub = sub.tail(100).reset_index(drop=True)

    # Korrelationen (Pearson) auf Rohwerten
    corr_em_td = float(sub["emotion"].corr(sub["td_error"]))
    corr_em_eta = float(sub["emotion"].corr(sub["eta"]))
    corr_td_eta = float(sub["td_error"].corr(sub["eta"]))

    print("Korrelationen (letzte 100 Episoden, Pearson):")
    print(f"  Emotion  vs TD-Error: {corr_em_td:+.3f}")
    print(f"  Emotion  vs eta     : {corr_em_eta:+.3f}")
    print(f"  TD-Error vs eta     : {corr_td_eta:+.3f}")

    # Visualisierung: zur besseren Vergleichbarkeit min-max-normalisieren
    plot_df = pd.DataFrame({
        "episode": sub["episode"],
        "emotion_norm": min_max_norm(sub["emotion"]),
        "td_error_norm": min_max_norm(sub["td_error"]),
        "eta_norm": min_max_norm(sub["eta"]),
    })

    plt.figure(figsize=(10, 6))
    plt.plot(plot_df["episode"], plot_df["emotion_norm"], label="Emotion (norm)", color="orange", linewidth=1.8)
    plt.plot(plot_df["episode"], plot_df["td_error_norm"], label="TD-Error (norm)", color="red", linewidth=1.6, alpha=0.85)
    plt.plot(plot_df["episode"], plot_df["eta_norm"], label="eta (norm)", color="blue", linewidth=1.6, alpha=0.85)

    plt.title("Trends (letzte 100 Episoden): Emotion · TD-Error · eta (min-max normalisiert)")
    plt.xlabel("Episode")
    plt.ylabel("Normierter Wert [0..1]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = results_dir / "eta_td_emotion_trends.png"
    plt.savefig(out_path, dpi=200)
    print(f"[Plot] gespeichert unter: {out_path}")

    # Zusätzlich: Emotion vs eta (direkte Kopplung) als Scatter
    plt.figure(figsize=(7, 5))
    plt.scatter(sub["emotion"], sub["eta"], alpha=0.6, color="blue")
    plt.xlabel("Emotion")
    plt.ylabel("eta")
    plt.title("Emotion vs eta (letzte 100 Episoden)")
    plt.grid(True, alpha=0.3)
    out_path2 = results_dir / "emotion_vs_eta.png"
    plt.tight_layout()
    plt.savefig(out_path2, dpi=200)
    print(f"[Plot] gespeichert unter: {out_path2}")


if __name__ == "__main__":
    main()


