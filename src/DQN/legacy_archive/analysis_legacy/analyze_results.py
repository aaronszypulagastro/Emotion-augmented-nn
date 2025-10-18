# -*- coding: utf-8 -*-
"""
Emotion NN – Laufanalyse (Phase 5.5+)
Lädt 'results/training_log.csv' und erstellt:
- Statistische Auswertung (Mittelwerte, Varianzen)
- Korrelationen (Emotion↔Reward, TD-Error↔η)
- Plots zur Stabilität und Dynamik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CSV laden ---
path = "results/training_log.csv"
if not os.path.exists(path):
    raise FileNotFoundError("⚠️ training_log.csv nicht gefunden! Bitte Training zuerst durchführen.")

df = pd.read_csv(path, on_bad_lines="skip", engine="python")

# --- Grundstruktur prüfen ---
print(f"Datensätze: {len(df)} Episoden\nSpalten: {list(df.columns)}")
df = df.fillna(method="ffill").fillna(0)

# --- Grundstatistik ---
stats = df.describe()[["reward", "emotion", "eta", "td_error", "sigma_mean"]]
print("\n📊 Grundstatistik (zentraler Bereich):\n", stats)

# --- Korrelationen ---
corr = df[["reward", "emotion", "eta", "td_error", "sigma_mean"]].corr()
print("\n🔗 Korrelationen:\n", corr)

# --- 1️⃣ Emotion ↔ Reward Verlauf ---
plt.figure(figsize=(8, 5))
plt.plot(df["emotion"], label="Emotion", color="orange")
plt.plot(df["reward"]/df["reward"].abs().max(), label="Reward (normiert)", color="green", alpha=0.6)
plt.title("Emotion ↔ Reward Verlauf")
plt.xlabel("Episode")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/analysis_emotion_reward.png", dpi=200)
print("✅ Plot: Emotion ↔ Reward gespeichert.")

# --- 2️⃣ TD-Error ↔ η Verlauf ---
plt.figure(figsize=(8, 5))
plt.plot(df["td_error"], label="TD-Error", color="red", alpha=0.8)
plt.plot(df["eta"] * 500, label="η (x500 skaliert)", color="blue", alpha=0.7)
plt.title("TD-Error ↔ η Verlauf")
plt.xlabel("Episode")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/analysis_td_eta.png", dpi=200)
print("✅ Plot: TD-Error ↔ η gespeichert.")

# --- 3️⃣ σ-Dynamik ---
plt.figure(figsize=(8, 4))
plt.plot(df["sigma_mean"], color="purple")
plt.title("σ-Aktivität (Plasticity)")
plt.xlabel("Episode")
plt.ylabel("mean(|σ|)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/analysis_sigma_dynamics.png", dpi=200)
print("✅ Plot: σ-Dynamik gespeichert.")

# --- 4️⃣ Heatmap der Beziehungen ---
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korrelationen der Kernvariablen")
plt.tight_layout()
plt.savefig("results/analysis_heatmap.png", dpi=200)
print("✅ Plot: Heatmap gespeichert.")

# --- 5️⃣ Kurze Auswertung ---
corr_em_rew = corr.loc["emotion", "reward"]
corr_td_eta = corr.loc["td_error", "eta"]
eta_var = np.var(df["eta"].tail(100))
td_var = np.var(df["td_error"].tail(100))

print("\n🧠 Zusammenfassung:")
print(f"• Emotion–Reward-Korrelation: {corr_em_rew:+.3f}")
print(f"• TD-Error–η-Korrelation:     {corr_td_eta:+.3f}")
print(f"• Var(η letzte 100):          {eta_var:.6f}")
print(f"• Var(TD-Error letzte 100):   {td_var:.3f}")

# --- Bewertung ---
if corr_em_rew > 0.3 and abs(corr_td_eta) < 0.2 and eta_var < 1e-6:
    print("\n✅ Modell stabil und emotional kohärent.")
elif corr_em_rew < 0:
    print("\n⚠️ Emotion und Reward negativ korreliert → mögliche Fehlanpassung.")
else:
    print("\nℹ️ Teilweise stabile Dynamik, weitere Läufe empfohlen.")

print("\nAlle Analyseplots im Ordner ./results gespeichert.")
