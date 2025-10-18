"""
Finale Option B Analyse mit PSA
================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("╔══════════════════════════════════════════════════════════════╗")
print("║      OPTION A vs B - FINALE VERGLEICHS-ANALYSE               ║")
print("╚══════════════════════════════════════════════════════════════╝\n")

# Lade beide Dateien
df_a = pd.read_csv("results/training_log_option_a_backup.csv", on_bad_lines='skip')
df_b = pd.read_csv("results/training_log.csv", on_bad_lines='skip')

print("📁 Daten geladen:")
print(f"   Option A (ohne PSA): {len(df_a)} Datenpunkte")
print(f"   Option B (mit PSA):  {len(df_b)} Datenpunkte\n")

# Extrahiere Returns
reward_col = 'return' if 'return' in df_a.columns else 'reward'
returns_a = df_a[reward_col].values
returns_b = df_b[reward_col].values

# Metriken berechnen
print("="*60)
print("📊 PERFORMANCE-VERGLEICH")
print("="*60 + "\n")

metrics_a = {
    'mean': np.mean(returns_a),
    'median': np.median(returns_a),
    'std': np.std(returns_a),
    'max': np.max(returns_a),
    'min': np.min(returns_a),
    'cv': np.std(returns_a) / (abs(np.mean(returns_a)) + 1e-8)
}

metrics_b = {
    'mean': np.mean(returns_b),
    'median': np.median(returns_b),
    'std': np.std(returns_b),
    'max': np.max(returns_b),
    'min': np.min(returns_b),
    'cv': np.std(returns_b) / (abs(np.mean(returns_b)) + 1e-8)
}

print("OPTION A (ohne PSA):")
print(f"   Mean:     {metrics_a['mean']:.2f}")
print(f"   Median:   {metrics_a['median']:.2f}")
print(f"   Std Dev:  {metrics_a['std']:.2f}")
print(f"   Max:      {metrics_a['max']:.2f}")
print(f"   CV:       {metrics_a['cv']:.3f}\n")

print("OPTION B (mit PSA):")
print(f"   Mean:     {metrics_b['mean']:.2f}")
print(f"   Median:   {metrics_b['median']:.2f}")
print(f"   Std Dev:  {metrics_b['std']:.2f}")
print(f"   Max:      {metrics_b['max']:.2f}")
print(f"   CV:       {metrics_b['cv']:.3f}\n")

# Verbesserung
improvement_mean = ((metrics_b['mean'] - metrics_a['mean']) / abs(metrics_a['mean'])) * 100
improvement_cv = ((metrics_a['cv'] - metrics_b['cv']) / metrics_a['cv']) * 100

print(f"VERBESSERUNG B vs A:")
print(f"   Mean Return:  {improvement_mean:+.1f}%")
print(f"   CV Reduktion: {improvement_cv:+.1f}%")

if improvement_mean > 10:
    print(f"   Status: ✅ DEUTLICH BESSER")
elif improvement_mean > 0:
    print(f"   Status: ⚠️  LEICHT BESSER")
else:
    print(f"   Status: ❌ SCHLECHTER")

# PSA-Analyse (letzte 5 Spalten)
print("\n" + "="*60)
print("📊 PSA-METRIKEN (Option B)")
print("="*60 + "\n")

# PSA-Spalten sind die letzten 5 Spalten
psa_stability = df_b.iloc[:, -5].values
psa_trend = df_b.iloc[:, -4].values
psa_conf_lower = df_b.iloc[:, -3].values
psa_conf_upper = df_b.iloc[:, -2].values
psa_anomalies = df_b.iloc[:, -1].values

print(f"Stability Score:")
print(f"   Mean:  {np.mean(psa_stability[psa_stability > 0]):.3f}")
print(f"   Min:   {np.min(psa_stability[psa_stability > 0]):.3f}")
print(f"   Max:   {np.max(psa_stability):.3f}\n")

print(f"Trend-Verteilung:")
trends, counts = np.unique(psa_trend, return_counts=True)
for trend, count in zip(trends, counts):
    print(f"   {trend}: {count}x")

print(f"\nTotal Anomalien erkannt: {int(psa_anomalies[-1])}")

# Visualisierung
print("\n" + "="*60)
print("📊 ERSTELLE VISUALISIERUNG")
print("="*60 + "\n")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Option A vs B - Vergleichsanalyse mit PSA', fontsize=16)

# Plot 1: Returns Vergleich
ax = axes[0, 0]
ax.plot(returns_a, 'o-', alpha=0.5, label='Option A (ohne PSA)', markersize=4)
ax.plot(returns_b, 's-', alpha=0.5, label='Option B (mit PSA)', markersize=4)
ax.axhline(y=metrics_a['mean'], color='blue', linestyle='--', label=f"A Mean: {metrics_a['mean']:.1f}")
ax.axhline(y=metrics_b['mean'], color='red', linestyle='--', label=f"B Mean: {metrics_b['mean']:.1f}")
ax.set_title('Returns Vergleich')
ax.set_xlabel('Datenpunkt')
ax.set_ylabel('Return')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Distribution Vergleich
ax = axes[0, 1]
ax.hist(returns_a, bins=30, alpha=0.5, label='Option A', edgecolor='black')
ax.hist(returns_b, bins=30, alpha=0.5, label='Option B', edgecolor='black')
ax.axvline(x=metrics_a['mean'], color='blue', linestyle='--', linewidth=2)
ax.axvline(x=metrics_b['mean'], color='red', linestyle='--', linewidth=2)
ax.set_title('Distribution Vergleich')
ax.set_xlabel('Return')
ax.set_ylabel('Häufigkeit')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Box Plot Vergleich
ax = axes[0, 2]
ax.boxplot([returns_a, returns_b], labels=['Option A', 'Option B'])
ax.set_title('Stabilität Vergleich')
ax.set_ylabel('Return')
ax.grid(True, alpha=0.3)

# Plot 4: PSA Stability Score
ax = axes[1, 0]
valid_stability = psa_stability[psa_stability > 0]
ax.plot(valid_stability, 'o-', color='green', markersize=4)
ax.set_title('PSA Stability Score (Option B)')
ax.set_xlabel('Datenpunkt')
ax.set_ylabel('Stability Score')
ax.axhline(y=0.5, color='red', linestyle='--', label='Target: 0.5')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: PSA Confidence Intervals
ax = axes[1, 1]
valid_idx = psa_conf_lower > 0
x_vals = np.arange(len(psa_conf_lower[valid_idx]))
ax.fill_between(x_vals, psa_conf_lower[valid_idx], psa_conf_upper[valid_idx], alpha=0.3, label='95% CI')
ax.plot(x_vals, returns_b[valid_idx], 'o-', markersize=3, label='Actual Returns')
ax.set_title('Confidence Intervals (PSA)')
ax.set_xlabel('Datenpunkt')
ax.set_ylabel('Return')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Anomaly Count
ax = axes[1, 2]
ax.plot(psa_anomalies, 'o-', color='red', markersize=4)
ax.set_title('Kumulierte Anomalien (PSA)')
ax.set_xlabel('Datenpunkt')
ax.set_ylabel('Anomaly Count')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('option_a_vs_b_final_comparison.png', dpi=150, bbox_inches='tight')
print("📊 Visualisierung gespeichert: option_a_vs_b_final_comparison.png\n")
plt.close()

# Fazit
print("="*60)
print("🎯 FAZIT & BEWERTUNG")
print("="*60 + "\n")

if metrics_b['mean'] > metrics_a['mean']:
    print(f"✅ Option B ist besser (+{improvement_mean:.1f}%)")
else:
    print(f"⚠️  Option B ähnlich/schlechter ({improvement_mean:+.1f}%)")

if metrics_b['cv'] < metrics_a['cv']:
    print(f"✅ Option B ist stabiler (-{improvement_cv:.1f}% CV)")
else:
    print(f"⚠️  Option B nicht stabiler ({improvement_cv:+.1f}% CV)")

print(f"\n📊 PSA-Nutzen:")
print(f"   ✅ {int(psa_anomalies[-1])} Anomalien erkannt")
print(f"   ✅ Trend-Analyse durchgeführt")
print(f"   ✅ Confidence-Intervalle berechnet")
print(f"   ✅ Real-time Monitoring funktioniert")

print("\n" + "="*60)
print("✅ ANALYSE ABGESCHLOSSEN")
print("="*60)

