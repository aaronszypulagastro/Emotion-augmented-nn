import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# === Daten laden ===
df = pd.read_csv(
    "results/training_log.csv",
    sep=",",
    on_bad_lines="skip",
    engine="python"
)

# Nur relevante Spalten herausfiltern
cols = [c for c in df.columns if any(k in c.lower() for k in ["emotion", "td_error", "eta", "episode"])]
df = df[cols].dropna()

# Spaltennamen vereinheitlichen
df.columns = [c.lower().replace(" ", "_") for c in df.columns]
if "episode" not in df.columns:
    df["episode"] = np.arange(len(df))

# === Normalisierung ===
df["emotion_n"] = df["emotion"] / df["emotion"].max()
df["td_error_n"] = df["td_error"] / df["td_error"].max()
df["eta_n"] = df["eta"] / df["eta"].max()

# === 3D Surface Plot ===
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

x = df["emotion_n"]
y = df["td_error_n"]
z = df["eta_n"]

# Mesh-Gitter erzeugen
grid_x, grid_y = np.meshgrid(
    np.linspace(x.min(), x.max(), 60),
    np.linspace(y.min(), y.max(), 60)
)
from scipy.interpolate import griddata
grid_z = griddata((x, y), z, (grid_x, grid_y), method="cubic")

surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
ax.set_xlabel("Emotion (normiert)")
ax.set_ylabel("TD-Error (normiert)")
ax.set_zlabel("η (normiert)")
ax.set_title("🧠 Policy Surface – Emotion / TD-Error / η")
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.tight_layout()
plt.savefig("results/policy_surface_3d.png", dpi=200)

# === Heatmap über Episoden ===
plt.figure(figsize=(10, 5))
df["zone_index"] = np.where(df["td_error_n"] > 0.7, 2, np.where(df["emotion_n"] > 0.6, 0, 1))
plt.scatter(df["episode"], df["emotion_n"], c=df["zone_index"], cmap="coolwarm", s=10)
plt.colorbar(label="Zone: 0=stabil, 1=neutral, 2=explorativ")
plt.xlabel("Episode")
plt.ylabel("Emotion (normiert)")
plt.title("🧩 Temporale Zonenverteilung über Episoden")
plt.tight_layout()
plt.savefig("results/policy_zone_heatmap.png", dpi=200)

print("\nAnalyse abgeschlossen ✅")
print("→ Ergebnisse gespeichert unter:")
print("   results/policy_surface_3d.png")
print("   results/policy_zone_heatmap.png")


