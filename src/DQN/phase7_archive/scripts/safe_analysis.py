"""
Safe Execution Mode Analysis
=============================
Keine blockierenden Fenster, Timeout-gesichert, nur Text-Output
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Headless matplotlib
import matplotlib
matplotlib.use('Agg')  # Kein interaktives Backend
import matplotlib.pyplot as plt

# Logging Setup
log_file = "results/analysis_log.txt"
os.makedirs("results", exist_ok=True)

def log_print(msg):
    """Print und gleichzeitig in Datei schreiben"""
    print(msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# Start
log_print(f"\n{'='*60}")
log_print(f"SAFE ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"{'='*60}\n")

try:
    # Lade Daten
    log_print("📊 Lade Trainings-Daten...")
    
    option_a_path = "results/training_log_option_a_backup.csv"
    option_b_path = "results/training_log.csv"
    
    df_a = pd.read_csv(option_a_path, on_bad_lines='skip')
    df_b = pd.read_csv(option_b_path, on_bad_lines='skip')
    
    log_print(f"   ✅ Option A: {len(df_a)} Datenpunkte")
    log_print(f"   ✅ Option B: {len(df_b)} Datenpunkte\n")
    
    # Performance-Analyse
    log_print("="*60)
    log_print("PERFORMANCE-VERGLEICH")
    log_print("="*60 + "\n")
    
    # Returns extrahieren
    reward_col = 'return' if 'return' in df_a.columns else 'reward'
    returns_a = df_a[reward_col].values
    returns_b = df_b[reward_col].values
    
    # Option A Metriken
    log_print("OPTION A (ohne PSA):")
    log_print(f"   Mean Return:  {np.mean(returns_a):.2f}")
    log_print(f"   Median:       {np.median(returns_a):.2f}")
    log_print(f"   Std Dev:      {np.std(returns_a):.2f}")
    log_print(f"   Max:          {np.max(returns_a):.2f}")
    log_print(f"   Min:          {np.min(returns_a):.2f}")
    cv_a = np.std(returns_a) / (abs(np.mean(returns_a)) + 1e-8)
    log_print(f"   CV:           {cv_a:.3f}\n")
    
    # Option B Metriken
    log_print("OPTION B (mit PSA):")
    log_print(f"   Mean Return:  {np.mean(returns_b):.2f}")
    log_print(f"   Median:       {np.median(returns_b):.2f}")
    log_print(f"   Std Dev:      {np.std(returns_b):.2f}")
    log_print(f"   Max:          {np.max(returns_b):.2f}")
    log_print(f"   Min:          {np.min(returns_b):.2f}")
    cv_b = np.std(returns_b) / (abs(np.mean(returns_b)) + 1e-8)
    log_print(f"   CV:           {cv_b:.3f}\n")
    
    # PSA-Metriken Check
    log_print("="*60)
    log_print("PSA-METRIKEN (Option B)")
    log_print("="*60 + "\n")
    
    if len(df_b.columns) >= 30:
        # PSA sind die letzten 5 Spalten
        psa_stability = df_b.iloc[:, -5].values
        psa_trend = df_b.iloc[:, -4].values
        psa_anomalies = df_b.iloc[:, -1].values
        
        valid_stability = psa_stability[psa_stability > 0]
        if len(valid_stability) > 0:
            log_print(f"Stability Score:")
            log_print(f"   Mean:  {np.mean(valid_stability):.3f}")
            log_print(f"   Last:  {valid_stability[-1]:.3f}\n")
        
        log_print(f"Trends erkannt:")
        unique_trends, counts = np.unique(psa_trend, return_counts=True)
        for trend, count in zip(unique_trends, counts):
            log_print(f"   {trend}: {count}x")
        
        log_print(f"\nTotal Anomalien: {int(psa_anomalies[-1])}\n")
        log_print("✅ PSA-Daten vorhanden und analysiert!")
    else:
        log_print("⚠️  PSA-Spalten nicht gefunden\n")
    
    # Kritisches Problem-Check
    log_print("="*60)
    log_print("PROBLEM-IDENTIFIKATION")
    log_print("="*60 + "\n")
    
    # Früh vs. Spät
    early = np.mean(returns_b[:10])
    late = np.mean(returns_b[-10:])
    change = ((late - early) / abs(early)) * 100
    
    log_print(f"Frühe Episoden (1-10):   {early:.2f}")
    log_print(f"Späte Episoden (last 10): {late:.2f}")
    log_print(f"Änderung:                {change:+.1f}%\n")
    
    if change < -50:
        log_print("❌ KRITISCH: Massive Performance-Verschlechterung!")
        log_print("   → η-Decay Collapse erkannt")
        log_print("   → Fixes notwendig!\n")
    elif change < 0:
        log_print("⚠️  WARNUNG: Performance-Rückgang")
    else:
        log_print("✅ Performance verbessert sich\n")
    
    # Visualisierung (SAFE - kein show())
    log_print("="*60)
    log_print("VISUALISIERUNG")
    log_print("="*60 + "\n")
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Returns
        axes[0].plot(returns_b, 'o-', markersize=3)
        axes[0].set_title('Option B Returns')
        axes[0].set_xlabel('Datenpunkt')
        axes[0].set_ylabel('Return')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: PSA Stability
        if len(df_b.columns) >= 30:
            psa_stab = df_b.iloc[:, -5].values
            axes[1].plot(psa_stab, 'o-', color='green', markersize=3)
            axes[1].set_title('PSA Stability Score')
            axes[1].set_xlabel('Datenpunkt')
            axes[1].set_ylabel('Stability')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/safe_analysis_plot.png', dpi=100)
        plt.close()  # WICHTIG: Close statt show()
        
        log_print("✅ Plot gespeichert: results/safe_analysis_plot.png\n")
    except Exception as e:
        log_print(f"⚠️  Plot-Fehler (überspringe): {e}\n")
    
    # Fazit
    log_print("="*60)
    log_print("FAZIT")
    log_print("="*60 + "\n")
    
    log_print("✅ Analyse erfolgreich abgeschlossen!")
    log_print(f"📁 Log gespeichert: {log_file}\n")
    
except Exception as e:
    log_print(f"\n❌ FEHLER: {e}")
    import traceback
    log_print(traceback.format_exc())

log_print(f"{'='*60}")
log_print(f"FERTIG - {datetime.now().strftime('%H:%M:%S')}")
log_print(f"{'='*60}\n")

