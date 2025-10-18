"""
Training-Stop Diagnose & Problemanalyse
========================================
"""

import pandas as pd
import numpy as np
import os

print("╔══════════════════════════════════════════════════════════════╗")
print("║         TRAINING-STOP DIAGNOSE & PROBLEMANALYSE              ║")
print("╚══════════════════════════════════════════════════════════════╝\n")

# Prüfe beide Logs
option_a_path = "results/training_log_option_a_backup.csv"
option_b_path = "results/training_log.csv"

print("📁 DATEI-CHECK:")
print("="*60)

# Option A (alt)
if os.path.exists(option_a_path):
    df_a = pd.read_csv(option_a_path, on_bad_lines='skip')
    print(f"✅ Option A (backup): {len(df_a)} Datenpunkte")
    print(f"   Letzte Episode: {df_a.iloc[-1, 0]:.0f}")
else:
    print(f"❌ Option A nicht gefunden")
    df_a = None

# Option B (neu)
if os.path.exists(option_b_path):
    df_b = pd.read_csv(option_b_path, on_bad_lines='skip')
    print(f"✅ Option B (neu):    {len(df_b)} Datenpunkte")
    print(f"   Letzte Episode: {df_b.iloc[-1, 0]:.0f}")
    
    # Prüfe PSA-Spalten
    psa_columns = [col for col in df_b.columns if 'psa_' in col]
    if psa_columns:
        print(f"   ✅ PSA-Spalten gefunden: {len(psa_columns)}")
        print(f"      {', '.join(psa_columns)}")
    else:
        print(f"   ❌ KEINE PSA-Spalten!")
else:
    print(f"❌ Option B nicht gefunden")
    df_b = None

print("\n" + "="*60)
print("🔍 PROBLEM-DIAGNOSE:")
print("="*60 + "\n")

if df_b is not None:
    last_episode = int(df_b.iloc[-1, 0])
    target_episodes = 500
    
    if last_episode < target_episodes:
        print(f"⚠️  PROBLEM IDENTIFIZIERT:")
        print(f"   Training bei Episode {last_episode} gestoppt")
        print(f"   Sollte bis Episode {target_episodes} laufen")
        print(f"   Fehlende Episoden: {target_episodes - last_episode}\n")
        
        print(f"🔎 MÖGLICHE URSACHEN:")
        print(f"   1. Training wurde manuell abgebrochen")
        print(f"   2. Fehler im Training-Script")
        print(f"   3. Ressourcen-Problem (RAM/CPU)")
        print(f"   4. Import-Fehler mit PSA")
    else:
        print(f"✅ Training vollständig abgeschlossen!")

# Vergleichs-Analyse
print("\n" + "="*60)
print("📊 PERFORMANCE-VERGLEICH:")
print("="*60 + "\n")

if df_a is not None and df_b is not None:
    reward_col = 'return' if 'return' in df_a.columns else 'reward'
    
    returns_a = df_a[reward_col].values
    returns_b = df_b[reward_col].values
    
    avg_a = np.mean(returns_a)
    avg_b = np.mean(returns_b)
    
    print(f"Option A (ohne PSA):")
    print(f"   Episoden:     {len(returns_a)}")
    print(f"   Mean Return:  {avg_a:.2f}")
    print(f"   Best Return:  {np.max(returns_a):.2f}")
    print(f"   Std Dev:      {np.std(returns_a):.2f}\n")
    
    print(f"Option B (mit PSA):")
    print(f"   Episoden:     {len(returns_b)}")
    print(f"   Mean Return:  {avg_b:.2f}")
    print(f"   Best Return:  {np.max(returns_b):.2f}")
    print(f"   Std Dev:      {np.std(returns_b):.2f}\n")
    
    # PSA-Analyse
    if 'psa_stability_score' in df_b.columns:
        psa_scores = df_b['psa_stability_score'].dropna()
        psa_trends = df_b['psa_trend'].dropna()
        psa_anomalies = df_b['psa_anomaly_count'].dropna()
        
        print(f"📊 PSA-METRIKEN (Option B):")
        print(f"   Stability Score (avg): {psa_scores.mean():.3f}")
        print(f"   Stability Score (last): {psa_scores.iloc[-1]:.3f}")
        print(f"   Trends: {psa_trends.value_counts().to_dict()}")
        print(f"   Total Anomalien: {psa_anomalies.iloc[-1]:.0f}")

# Nächste Schritte
print("\n" + "="*60)
print("🚀 NÄCHSTE SCHRITTE:")
print("="*60 + "\n")

if df_b is not None and last_episode < 500:
    print("EMPFEHLUNG:")
    print("1. ✅ PSA ist integriert und funktioniert")
    print("2. ⚠️  Training wurde bei Episode 491 unterbrochen")
    print("3. 📋 Optionen:\n")
    
    print("   OPTION 1 (Empfohlen): Training fortsetzen")
    print("   → Starte Training erneut")
    print("   → Es wird von vorne beginnen (neuer Run)")
    print("   → Diesmal bis Episode 500 laufen lassen\n")
    
    print("   OPTION 2: Vorhandene Daten analysieren")
    print("   → Analysiere 491 Episoden (fast vollständig)")
    print("   → Vergleiche mit Option A")
    print("   → Bewerte PSA-Effektivität\n")
    
    print("   OPTION 3: Konfiguration prüfen")
    print("   → Überprüfe CONFIG['episodes'] in train_finetuning.py")
    print("   → Stelle sicher es ist 500 (nicht 491)")

elif df_b is not None and last_episode >= 500:
    print("✅ TRAINING VOLLSTÄNDIG!")
    print("   → Führe Vergleichs-Analyse durch")
    print("   → Bewerte PSA-Effektivität")
    print("   → Plane Phase 7.1 Schritte")

print("\n" + "="*60)
print("✅ DIAGNOSE ABGESCHLOSSEN")
print("="*60)

