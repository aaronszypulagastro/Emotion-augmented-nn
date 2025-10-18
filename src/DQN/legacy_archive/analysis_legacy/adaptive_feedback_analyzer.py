import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr, gaussian_kde
import os

# === Pfad konfigurieren ===
BASE_DIR = os.path.dirname(__file__)
LOG_PATH = os.path.join(BASE_DIR, "results", "training_log.csv")
OUT_DIR  = os.path.join(BASE_DIR, "results", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[Info] Lade Trainingsdaten aus: {LOG_PATH}")
df = pd.read_csv(LOG_PATH, on_bad_lines="skip", engine="python")


# === Daten prüfen ===
required_cols = ["emotion", "reward", "td_error", "eta"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Fehlende Spalten: {missing}")

# === Grundstatistik ===
stats = df[required_cols].describe()
print("\n=== Statistische Übersicht ===")
print(stats)

# === Korrelationen berechnen ===
corr_matrix = df[required_cols].corr(method="pearson")
print("\n=== Korrelationsmatrix ===")
print(corr_matrix)

# === 3D Heatmap: Emotion × TD-Error × η ===
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

x = df["emotion"].values
y = df["td_error"].values
z = df["eta"].values

ax.scatter(x, y, z, c=z, cmap="plasma", alpha=0.7)
ax.set_xlabel("Emotion")
ax.set_ylabel("TD-Error")
ax.set_zlabel("η (adaptive)")
ax.set_title("3D Heatmap: Emotion × TD-Error × η")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "heatmap_3d_emotion_td_eta.png"), dpi=200)
plt.close()
print("[Plot] 3D Heatmap gespeichert.")

# === 2D-Dichtekarte: η ↔ Emotion ===
x = df["emotion"]
y = df["eta"]
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, c=z, s=30, cmap="inferno")
plt.xlabel("Emotion")
plt.ylabel("η (adaptive)")
plt.title("Dichtekarte: Emotion ↔ η")
plt.colorbar(label="Dichte")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "density_emotion_eta.png"), dpi=200)
plt.close()
print("[Plot] Dichtekarte gespeichert.")

# === Adaptive Parameterempfehlung ===
r_em_td, _ = pearsonr(df["emotion"], df["td_error"])
r_em_eta, _ = pearsonr(df["emotion"], df["eta"])
r_eta_td, _ = pearsonr(df["eta"], df["td_error"])

gain_suggestion = 1.05 + (r_em_eta * 0.2)
eta_clip_suggestion = 0.005 + abs(r_eta_td) * 0.004

print("\n=== Adaptive Tuning-Empfehlungen ===")
print(f"Emotion-Gain ≈ {gain_suggestion:.3f}")
print(f"η-Max ≈ {eta_clip_suggestion:.4f}")
print(f"Emotion↔η r = {r_em_eta:.3f}")
print(f"η↔TD-Error r = {r_eta_td:.3f}")

print("\nAnalyse abgeschlossen. Ergebnisse unter:")
print(f"→ {OUT_DIR}")

