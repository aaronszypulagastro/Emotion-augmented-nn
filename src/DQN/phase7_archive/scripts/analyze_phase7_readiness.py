"""
Phase 7.0 Readiness Analysis
=============================

Analysiert die aktuellen Trainings-Ergebnisse und bewertet,
ob Phase 7.0 bereit ist für den produktiven Einsatz.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Training Log einlesen
log_path = "results/training_log.csv"

print("╔══════════════════════════════════════════════════════╗")
print("║      Phase 7.0 Readiness Analysis                    ║")
print("╚══════════════════════════════════════════════════════╝\n")

if not os.path.exists(log_path):
    print(f"❌ Training Log nicht gefunden: {log_path}")
    exit(1)

# CSV einlesen (robust mit error handling)
try:
    df = pd.read_csv(log_path, on_bad_lines='skip')
except:
    # Fallback: lese nur die ersten relevanten Spalten
    df = pd.read_csv(log_path, usecols=[0, 1, 2, 3, 4, 5], on_bad_lines='skip')

print(f"📊 Trainings-Daten gefunden:")
print(f"   Anzahl Episoden: {len(df)}")
print(f"   Spalten: {', '.join(df.columns[:6])}...\n")

# Episode column name variation handling
if 'episode' in df.columns:
    episode_col = 'episode'
elif 'Episode' in df.columns:
    episode_col = 'Episode'
else:
    episode_col = df.columns[0]

if 'reward' in df.columns:
    reward_col = 'reward'
elif 'return' in df.columns:
    reward_col = 'return'
else:
    reward_col = df.columns[1]

# Performance-Metriken berechnen
episodes = df[episode_col].values
returns = df[reward_col].values

# avg100 über die Zeit
avg100_history = []
for i in range(len(returns)):
    start_idx = max(0, i - 99)
    avg100 = np.mean(returns[start_idx:i+1])
    avg100_history.append(avg100)

# Aktuelle Performance
current_avg100 = avg100_history[-1]
best_avg100 = max(avg100_history)
worst_avg100 = min(avg100_history)

# Stabilität
recent_100 = returns[-100:] if len(returns) >= 100 else returns
stability_score = 1.0 / (1.0 + np.std(recent_100) / (abs(np.mean(recent_100)) + 1e-8))

# TD-Error
if 'td_error' in df.columns:
    td_errors = df['td_error'].values
    avg_td_error = np.mean(td_errors[-100:]) if len(td_errors) >= 100 else np.mean(td_errors)
else:
    avg_td_error = 0.0

# Emotion
if 'emotion' in df.columns:
    emotions = df['emotion'].values
    avg_emotion = np.mean(emotions[-100:]) if len(emotions) >= 100 else np.mean(emotions)
else:
    avg_emotion = 0.0

print("═" * 54)
print("📈 PERFORMANCE-ANALYSE")
print("═" * 54)
print(f"\n🎯 Aktuelle Metriken (letzte 100 Episoden):")
print(f"   avg100:           {current_avg100:.2f}")
print(f"   Best avg100:      {best_avg100:.2f} (Episode {np.argmax(avg100_history)})")
print(f"   Worst avg100:     {worst_avg100:.2f}")
print(f"   Stability Score:  {stability_score:.3f}")
print(f"   Avg TD Error:     {avg_td_error:.3f}")
print(f"   Avg Emotion:      {avg_emotion:.3f}")

# Phase-Vergleich
print(f"\n📊 Vergleich mit Phase 6.x:")
print(f"   Phase 6.1 (Best):     40.05")
print(f"   Phase 6.3 (Latest):   25.90")
print(f"   Aktuell:              {current_avg100:.2f}")

if current_avg100 > 40.05:
    print(f"   Status: ✅ BESSER als Phase 6.1 (+{current_avg100 - 40.05:.2f})")
elif current_avg100 > 25.90:
    print(f"   Status: ⚠️  Zwischen Phase 6.1 und 6.3")
else:
    print(f"   Status: ❌ Schlechter als Phase 6.3")

# Variabilität
variance = np.var(recent_100)
cv = np.std(recent_100) / (abs(np.mean(recent_100)) + 1e-8)
print(f"\n📉 Variabilität:")
print(f"   Varianz:              {variance:.2f}")
print(f"   Coeff. of Variation:  {cv:.3f}")
print(f"   Target:               < 0.10 (Varianz < 10%)")

if cv < 0.10:
    print(f"   Status: ✅ HOCH STABIL")
elif cv < 0.20:
    print(f"   Status: ⚠️  MITTEL STABIL")
else:
    print(f"   Status: ❌ INSTABIL")

# Trend-Analyse
if len(returns) >= 100:
    early_avg = np.mean(returns[:100])
    late_avg = np.mean(returns[-100:])
    improvement = ((late_avg - early_avg) / (abs(early_avg) + 1e-8)) * 100
    
    print(f"\n📈 Lernfortschritt:")
    print(f"   Start (ep 0-100):     {early_avg:.2f}")
    print(f"   Ende (letzte 100):    {late_avg:.2f}")
    print(f"   Verbesserung:         {improvement:+.1f}%")
    
    if improvement > 50:
        print(f"   Status: ✅ STARKER LERNFORTSCHRITT")
    elif improvement > 0:
        print(f"   Status: ⚠️  MODERATER LERNFORTSCHRITT")
    else:
        print(f"   Status: ❌ KEIN/NEGATIVER LERNFORTSCHRITT")

# Phase 7.0 Readiness Assessment
print("\n" + "═" * 54)
print("🎯 PHASE 7.0 READINESS BEWERTUNG")
print("═" * 54)

readiness_score = 0
max_score = 5

# Kriterium 1: Performance > Phase 6.3
if current_avg100 > 25.90:
    readiness_score += 1
    print("✅ [1/5] Performance besser als Phase 6.3")
else:
    print("❌ [0/5] Performance schlechter als Phase 6.3")

# Kriterium 2: Stabilität
if cv < 0.20:
    readiness_score += 1
    print("✅ [2/5] Akzeptable Stabilität erreicht")
else:
    print("❌ [1/5] Instabile Performance")

# Kriterium 3: Lernfortschritt
if len(returns) >= 100:
    if improvement > 0:
        readiness_score += 1
        print("✅ [3/5] Positiver Lernfortschritt")
    else:
        print("❌ [2/5] Kein Lernfortschritt")
else:
    print("⚠️  [2/5] Nicht genug Daten für Trend-Analyse")

# Kriterium 4: TD-Error unter Kontrolle
if avg_td_error < 10.0:
    readiness_score += 1
    print("✅ [4/5] TD-Error unter Kontrolle")
else:
    print("❌ [3/5] TD-Error zu hoch")

# Kriterium 5: Emotion-System aktiv
if avg_emotion > 0.3:
    readiness_score += 1
    print("✅ [5/5] Emotion-System aktiv")
else:
    print("❌ [4/5] Emotion-System inaktiv")

readiness_percentage = (readiness_score / max_score) * 100

print(f"\n🎯 READINESS SCORE: {readiness_score}/{max_score} ({readiness_percentage:.0f}%)")

if readiness_score >= 4:
    print("\n✅ ERGEBNIS: PHASE 7.0 IST BEREIT FÜR EVALUATION!")
    print("   → Empfehlung: Benchmark-Suite ausführen")
elif readiness_score >= 3:
    print("\n⚠️  ERGEBNIS: PHASE 7.0 IST TEILWEISE BEREIT")
    print("   → Empfehlung: Weitere Optimierung notwendig")
else:
    print("\n❌ ERGEBNIS: PHASE 7.0 BRAUCHT MEHR ENTWICKLUNG")
    print("   → Empfehlung: Baseline-Training und Debugging")

# Nächste Schritte
print("\n" + "═" * 54)
print("🚀 EMPFOHLENE NÄCHSTE SCHRITTE")
print("═" * 54)

if current_avg100 < 25.90:
    print("\n1️⃣  PRIORITÄT HOCH: Performance-Verbesserung")
    print("   → Problem: Performance schlechter als Phase 6.3")
    print("   → Lösung: Phase 7.0 Features aktivieren")
    print("   → Aktion: python train_phase7.py --episodes 200")
    
elif current_avg100 < 40.05:
    print("\n1️⃣  PRIORITÄT MITTEL: Performance-Optimierung")
    print("   → Ziel: Phase 6.1 Performance erreichen (40.05)")
    print("   → Lösung: Hyperparameter-Optimierung")
    print("   → Aktion: Phase 7.0 Bayesian Optimization nutzen")

else:
    print("\n1️⃣  PRIORITÄT: Stabilität erhöhen")
    print("   → Performance ist gut, aber Stabilität verbessern")
    print("   → Lösung: Adaptive Configuration Manager")

if cv > 0.10:
    print("\n2️⃣  STABILITÄT VERBESSERN")
    print("   → Problem: Hohe Variabilität")
    print("   → Lösung: Performance Stability Analyzer nutzen")
    print("   → Aktion: Längeres Training (500+ Episoden)")

print("\n3️⃣  BENCHMARK DURCHFÜHREN")
print("   → Vergleich verschiedener Konfigurationen")
print("   → Aktion: python phase7_benchmark.py")

print("\n4️⃣  DOKUMENTATION & PUBLIKATION")
print("   → Ergebnisse dokumentieren")
print("   → Paper-Draft erstellen")
print("   → GitHub/arXiv veröffentlichen")

# Visualisierung erstellen
if len(returns) >= 10:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 7.0 Readiness Analysis', fontsize=16)
    
    # Plot 1: Returns over time
    ax = axes[0, 0]
    ax.plot(episodes, returns, alpha=0.3, label='Episode Returns')
    ax.plot(episodes, avg100_history, linewidth=2, label='avg100')
    ax.axhline(y=40.05, color='g', linestyle='--', label='Phase 6.1 (Best)')
    ax.axhline(y=25.90, color='r', linestyle='--', label='Phase 6.3')
    ax.set_title('Performance über Zeit')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: TD Error
    if 'td_error' in df.columns:
        ax = axes[0, 1]
        ax.plot(episodes, td_errors, alpha=0.5)
        ax.set_title('TD Error über Zeit')
        ax.set_xlabel('Episode')
        ax.set_ylabel('TD Error')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Emotion
    if 'emotion' in df.columns:
        ax = axes[1, 0]
        ax.plot(episodes, emotions, alpha=0.5)
        ax.set_title('Emotion über Zeit')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Emotion')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Stability (Rolling Std)
    ax = axes[1, 1]
    window = 50
    rolling_std = pd.Series(returns).rolling(window=window).std()
    ax.plot(episodes, rolling_std)
    ax.set_title(f'Stabilität (Rolling Std, window={window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Std Dev')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = 'phase7_readiness_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n📊 Visualisierung gespeichert: {output_path}")
    plt.close()

print("\n" + "═" * 54)
print("✅ Analyse abgeschlossen!")
print("═" * 54 + "\n")

