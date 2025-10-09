import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# === Daten laden ===
df = pd.read_csv(
    "results/training_log.csv",
    sep=",",
    on_bad_lines="skip",   # Zeilen mit Fehlern überspringen
    engine="python"        # nutzt den stabileren Python-Parser
)


# Nur relevante Spalten
df = df[["emotion", "td_error", "eta"]].dropna()

# === Zonen definieren (3 Cluster: stabil, neutral, explorativ) ===
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["zone"] = kmeans.fit_predict(df[["emotion", "td_error"]])

# Cluster-Mittelpunkte berechnen
centers = pd.DataFrame(kmeans.cluster_centers_, columns=["emotion", "td_error"])
centers["zone"] = range(3)

print("\n=== BDH-Zone-Response Übersicht ===")
for _, row in centers.iterrows():
    emo, td = row["emotion"], row["td_error"]
    zone_name = (
        "🟢 Stabilisierung" if emo > 0.7 and td < 1.0 else
        "🟠 Neutral / Übergang" if 0.4 < emo < 0.7 else
        "🔴 Exploration"
    )
    print(f"Zone {int(row['zone'])}: Emotion={emo:.3f}, TD-Error={td:.3f} → {zone_name}")

# === 2D-Plot: Emotion × TD-Error ===
plt.figure(figsize=(8, 6))
colors = ['red', 'orange', 'green']
for i, color in enumerate(colors):
    cluster = df[df["zone"] == i]
    plt.scatter(cluster["emotion"], cluster["td_error"], color=color, label=f"Zone {i}")
plt.xlabel("Emotion")
plt.ylabel("TD-Error")
plt.title("BDH-Zone-Response-Map (2D)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/zone_response_map_2d.png", dpi=200)

# === 3D-Plot: Emotion × TD-Error × η ===
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")
for i, color in enumerate(colors):
    cluster = df[df["zone"] == i]
    ax.scatter(cluster["emotion"], cluster["td_error"], cluster["eta"], color=color, label=f"Zone {i}", s=30)
ax.set_xlabel("Emotion")
ax.set_ylabel("TD-Error")
ax.set_zlabel("η (adaptive)")
ax.set_title("BDH-Zone-Response-Map (3D)")
ax.legend()
plt.tight_layout()
plt.savefig("results/zone_response_map_3d.png", dpi=200)

print("\nAnalyse abgeschlossen ✅")
print("→ Ergebnisse gespeichert unter: results/zone_response_map_2d.png und zone_response_map_3d.png")
