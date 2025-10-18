"""Schnelle Text-Only Analyse - KEINE Plots"""
import pandas as pd
import numpy as np

print("\n" + "="*60)
print("SCHNELL-ANALYSE - Training mit Fixes")
print("="*60 + "\n")

df = pd.read_csv("results/training_log.csv")
returns = df['return'].values if 'return' in df.columns else df.iloc[:, 1].values

print(f"Datenpunkte: {len(df)}")
print(f"Letzte Episode: {df.iloc[-1, 0]:.0f}\n")

print("PERFORMANCE:")
print(f"  Mean:   {np.mean(returns):.2f}")
print(f"  Median: {np.median(returns):.2f}")
print(f"  Max:    {np.max(returns):.2f}")
print(f"  Min:    {np.min(returns):.2f}")
print(f"  Std:    {np.std(returns):.2f}\n")

# Früh vs. Spät
if len(returns) >= 20:
    early = np.mean(returns[:10])
    late = np.mean(returns[-10:])
    print(f"Frühe Episoden (1-10):  {early:.2f}")
    print(f"Späte Episoden (last):  {late:.2f}")
    print(f"Änderung:              {((late-early)/early)*100:+.1f}%\n")

# PSA Check
if len(df.columns) > 30:
    print("PSA-DATEN:")
    psa_stab = df.iloc[:, -5].dropna()
    psa_trend = df.iloc[:, -4].dropna()
    if len(psa_stab) > 0:
        print(f"  Stability (mean): {psa_stab.mean():.3f}")
        print(f"  Trends: {psa_trend.value_counts().to_dict()}\n")

print("="*60)
print("VERGLEICH MIT VORHERIGEN LÄUFEN:")
print("="*60)
print("  Phase 6.1:     40.05")
print("  Option A/B:    11.20 (Collapse)")
print(f"  Option C (neu): {np.mean(returns):.2f}")
print("\n" + "="*60 + "\n")

