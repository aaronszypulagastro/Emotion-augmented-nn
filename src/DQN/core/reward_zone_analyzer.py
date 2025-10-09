import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# === Pfade ===
BASE_DIR = os.path.dirname(__file__)
LOG_PATH = os.path.join(BASE_DIR, "results", "training_log.csv")
OUT_DIR  = os.path.join(BASE_DIR, "results", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[Info] Lade Trainingsdaten aus: {LOG_PATH}")
df = pd.read_csv(LOG_PATH, on_bad_lines="skip", engine="python")

# === Relevante Spalten prüfen ===
required_cols = ["emotion", "reward", "td_error"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Spalte '{c}' fehlt in training_log.csv!")

data = df[required_cols].dropna()

# === Daten normalisieren ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# === K-Means-Clustering (3 emotionale Lernzonen) ===
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
data["zone"] = labels

# === Statistik pro Zone ===
zone_stats = data.groupby("zone").agg({
    "emotion": ["mean", "std"],
    "reward": ["mean", "std"],
    "td_error": ["mean", "std"]
})
zone_stats.columns = ['_'.join(col) for col in zone_stats.columns]
zone_stats = zone_stats.reset_index()

print("\n=== Reward-Zonen Übersicht ===")
for _, row in zone_stats.iterrows():
    z = int(row["zone"])
    emo, rew, err = row["emotion_mean"], row["reward_mean"], row["td_error_mean"]
    print(f"Zone {z}: ⟨Emotion⟩={emo:.3f}, ⟨Reward⟩={rew:.2f}, ⟨TD-Error⟩={err:.2f}")

# === Zonencharakteristik bestimmen ===
def interpret_zone(row):
    if row["reward_mean"] > df["reward"].mean() and row["emotion_mean"] > 0.7:
        return "💎 Konsolidierungszone (stabil)"
    elif row["td_error_mean"] > df["td_error"].mean() and row["emotion_mean"] < 0.5:
        return "⚡ Explorationszone (risikoreich)"
    else:
        return "⚙️ Übergangszone (neutral)"

zone_stats["character"] = zone_stats.apply(interpret_zone, axis=1)

print("\n=== Zonencharakteristik ===")
for _, row in zone_stats.iterrows():
    print(f"Zone {int(row['zone'])}: {row['character']}")

# === 2D Plot: Emotion × Reward ===
plt.figure(figsize=(8,6))
for z in range(3):
    subset = data[data["zone"] == z]
    plt.scatter(subset["emotion"], subset["reward"], s=40, label=f"Zone {z}")
plt.xlabel("Emotion")
plt.ylabel("Reward")
plt.title("Reward-Zonenkarte: Emotion × Reward")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reward_zones_2d.png"), dpi=200)
plt.close()

# === 3D Plot: Emotion × TD-Error × Reward ===
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for z in range(3):
    subset = data[data["zone"] == z]
    ax.scatter(subset["emotion"], subset["td_error"], subset["reward"],
               color=colors[z], label=f"Zone {z}", alpha=0.7)
ax.set_xlabel("Emotion")
ax.set_ylabel("TD-Error")
ax.set_zlabel("Reward")
ax.set_title("3D Reward-Zonenanalyse")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reward_zones_3d.png"), dpi=200)
plt.close()

print("\nAnalyse abgeschlossen ✅")
print(f"→ Ergebnisse gespeichert unter: {OUT_DIR}")
