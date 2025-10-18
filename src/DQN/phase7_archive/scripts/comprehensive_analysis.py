"""
Umfassende Analyse nach Option A Training
==========================================

Detaillierte Auswertung der Trainings-Ergebnisse.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("╔══════════════════════════════════════════════════════════════╗")
print("║        UMFASSENDE TRAININGS-ANALYSE (Option A)               ║")
print("╚══════════════════════════════════════════════════════════════╝\n")

log_path = "results/training_log.csv"

# CSV einlesen
try:
    df = pd.read_csv(log_path, on_bad_lines='skip')
    print(f"✅ Training-Log geladen: {log_path}")
    print(f"   Datenpunkte: {len(df)}")
except Exception as e:
    print(f"❌ Fehler beim Laden: {e}")
    exit(1)

# Spalten identifizieren
episode_col = 'episode' if 'episode' in df.columns else df.columns[0]
reward_col = 'return' if 'return' in df.columns else 'reward'

print(f"   Spalten: {', '.join(df.columns[:8])}...\n")

# Daten extrahieren
episodes = df[episode_col].values
returns = df[reward_col].values

# Metriken berechnen
print("="*60)
print("📊 PERFORMANCE-METRIKEN")
print("="*60 + "\n")

# Gesamtstatistiken
mean_return = np.mean(returns)
median_return = np.median(returns)
std_return = np.std(returns)
min_return = np.min(returns)
max_return = np.max(returns)

print(f"📈 Gesamtstatistik:")
print(f"   Mean:       {mean_return:.2f}")
print(f"   Median:     {median_return:.2f}")
print(f"   Std Dev:    {std_return:.2f}")
print(f"   Min:        {min_return:.2f}")
print(f"   Max:        {max_return:.2f}")

# Quartile
q25 = np.percentile(returns, 25)
q75 = np.percentile(returns, 75)
print(f"   Q25:        {q25:.2f}")
print(f"   Q75:        {q75:.2f}")

# avg100 (oder avg verfügbar)
if len(returns) >= 100:
    avg100 = np.mean(returns[-100:])
    print(f"\n🎯 avg100 (letzte 100): {avg100:.2f}")
elif len(returns) >= 50:
    avg50 = np.mean(returns[-50:])
    print(f"\n🎯 avg50 (letzte 50): {avg50:.2f}")
else:
    avg_all = np.mean(returns)
    print(f"\n🎯 avg (alle): {avg_all:.2f}")

# Stabilität
cv = std_return / (abs(mean_return) + 1e-8)
stability_score = 1.0 / (1.0 + cv)

print(f"\n📉 Stabilität:")
print(f"   Coeff. of Variation: {cv:.3f}")
print(f"   Stability Score:     {stability_score:.3f}")

if cv < 0.1:
    stability_status = "✅ SEHR STABIL"
elif cv < 0.5:
    stability_status = "⚠️  MODERAT STABIL"
elif cv < 1.0:
    stability_status = "⚠️  WENIG STABIL"
else:
    stability_status = "❌ INSTABIL"
print(f"   Status:              {stability_status}")

# Trend-Analyse
if len(returns) >= 20:
    # Erste vs. Letzte 20%
    split_point = int(len(returns) * 0.2)
    early_mean = np.mean(returns[:split_point])
    late_mean = np.mean(returns[-split_point:])
    improvement = ((late_mean - early_mean) / (abs(early_mean) + 1e-8)) * 100
    
    print(f"\n📈 Lernfortschritt:")
    print(f"   Frühe Episoden (20%): {early_mean:.2f}")
    print(f"   Späte Episoden (20%): {late_mean:.2f}")
    print(f"   Verbesserung:         {improvement:+.1f}%")
    
    if improvement > 50:
        trend_status = "✅ STARKER FORTSCHRITT"
    elif improvement > 10:
        trend_status = "✅ GUTER FORTSCHRITT"
    elif improvement > 0:
        trend_status = "⚠️  LEICHTER FORTSCHRITT"
    else:
        trend_status = "❌ KEIN/NEGATIVER FORTSCHRITT"
    print(f"   Status:               {trend_status}")

# TD-Error Analyse
if 'td_error' in df.columns:
    td_errors = df['td_error'].values
    mean_td = np.mean(td_errors)
    late_td = np.mean(td_errors[-20:]) if len(td_errors) >= 20 else mean_td
    
    print(f"\n🔧 TD-Error:")
    print(f"   Mean:       {mean_td:.3f}")
    print(f"   Letzte 20:  {late_td:.3f}")
    
    if late_td < 1.0:
        td_status = "✅ SEHR GUT"
    elif late_td < 5.0:
        td_status = "✅ GUT"
    elif late_td < 10.0:
        td_status = "⚠️  AKZEPTABEL"
    else:
        td_status = "❌ HOCH"
    print(f"   Status:     {td_status}")

# Emotion-Analyse
if 'emotion' in df.columns:
    emotions = df['emotion'].values
    mean_emotion = np.mean(emotions)
    late_emotion = np.mean(emotions[-20:]) if len(emotions) >= 20 else mean_emotion
    
    print(f"\n💓 Emotion:")
    print(f"   Mean:       {mean_emotion:.3f}")
    print(f"   Letzte 20:  {late_emotion:.3f}")
    
    if mean_emotion > 0.3:
        emotion_status = "✅ AKTIV"
    else:
        emotion_status = "❌ INAKTIV"
    print(f"   Status:     {emotion_status}")

# Vergleich mit Phasen
print("\n" + "="*60)
print("📊 VERGLEICH MIT VORHERIGEN PHASEN")
print("="*60 + "\n")

phase61_avg = 40.05
phase63_avg = 25.90
current_avg = mean_return

print(f"Phase 6.1 (Best):     {phase61_avg:.2f}")
print(f"Phase 6.3 (Latest):   {phase63_avg:.2f}")
print(f"Aktuell (Option A):   {current_avg:.2f}\n")

if current_avg > phase61_avg:
    improvement61 = ((current_avg - phase61_avg) / phase61_avg) * 100
    print(f"Status: ✅ BESSER als Phase 6.1 (+{improvement61:.1f}%)")
elif current_avg > phase63_avg:
    print(f"Status: ⚠️  Zwischen Phase 6.1 und 6.3")
else:
    decline = ((phase63_avg - current_avg) / phase63_avg) * 100
    print(f"Status: ❌ Schlechter als Phase 6.3 (-{decline:.1f}%)")

# Readiness-Score
print("\n" + "="*60)
print("🎯 PHASE 7.0 READINESS BEWERTUNG")
print("="*60 + "\n")

readiness = 0
max_readiness = 5

# Kriterium 1: Performance
if current_avg > phase61_avg:
    readiness += 1
    print("✅ [1] Performance übertrifft Phase 6.1")
elif current_avg > phase63_avg:
    readiness += 0.5
    print("⚠️  [0.5] Performance besser als Phase 6.3")
else:
    print("❌ [0] Performance schlechter als Phase 6.3")

# Kriterium 2: Stabilität
if cv < 0.2:
    readiness += 1
    print("✅ [2] Gute Stabilität (CV < 0.2)")
elif cv < 0.5:
    readiness += 0.5
    print("⚠️  [0.5] Moderate Stabilität")
else:
    print("❌ [0] Instabile Performance")

# Kriterium 3: Lernfortschritt
if len(returns) >= 20 and improvement > 10:
    readiness += 1
    print("✅ [3] Positiver Lernfortschritt")
elif len(returns) >= 20 and improvement > 0:
    readiness += 0.5
    print("⚠️  [0.5] Leichter Lernfortschritt")
else:
    print("❌ [0] Kein Lernfortschritt")

# Kriterium 4: TD-Error
if 'td_error' in df.columns and late_td < 5.0:
    readiness += 1
    print("✅ [4] TD-Error unter Kontrolle")
elif 'td_error' in df.columns and late_td < 10.0:
    readiness += 0.5
    print("⚠️  [0.5] TD-Error akzeptabel")
else:
    print("❌ [0] TD-Error zu hoch")

# Kriterium 5: Emotion-System
if 'emotion' in df.columns and mean_emotion > 0.3:
    readiness += 1
    print("✅ [5] Emotion-System aktiv")
else:
    print("❌ [0] Emotion-System inaktiv")

readiness_pct = (readiness / max_readiness) * 100

print(f"\n🎯 READINESS SCORE: {readiness:.1f}/{max_readiness} ({readiness_pct:.0f}%)\n")

if readiness >= 4:
    print("✅ ERGEBNIS: PHASE 7.0 IST BEREIT FÜR NEXT STEPS!")
    print("   → Empfehlung: Option B (PSA Integration) durchführen")
elif readiness >= 3:
    print("⚠️  ERGEBNIS: PHASE 7.0 IST TEILWEISE BEREIT")
    print("   → Empfehlung: Optimierung mit Phase 7 Features")
else:
    print("❌ ERGEBNIS: MEHR ENTWICKLUNG NOTWENDIG")
    print("   → Empfehlung: Baseline verbessern")

# Visualisierung
print("\n" + "="*60)
print("📊 VISUALISIERUNG ERSTELLEN")
print("="*60 + "\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Option A: Training-Ergebnisse (Umfassende Analyse)', fontsize=16)

# Plot 1: Returns over Episodes
ax = axes[0, 0]
ax.plot(episodes, returns, 'o-', alpha=0.6, markersize=3)
ax.axhline(y=phase61_avg, color='g', linestyle='--', linewidth=2, label=f'Phase 6.1 ({phase61_avg:.1f})')
ax.axhline(y=phase63_avg, color='r', linestyle='--', linewidth=2, label=f'Phase 6.3 ({phase63_avg:.1f})')
ax.axhline(y=mean_return, color='b', linestyle='-', linewidth=2, label=f'Mean ({mean_return:.1f})')
ax.set_title('Episode Returns')
ax.set_xlabel('Episode')
ax.set_ylabel('Return')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Distribution
ax = axes[0, 1]
ax.hist(returns, bins=30, alpha=0.7, edgecolor='black')
ax.axvline(x=mean_return, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_return:.1f}')
ax.axvline(x=median_return, color='g', linestyle='--', linewidth=2, label=f'Median: {median_return:.1f}')
ax.set_title('Returns Distribution')
ax.set_xlabel('Return')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: TD Error (if available)
if 'td_error' in df.columns:
    ax = axes[1, 0]
    ax.plot(episodes, td_errors, 'o-', alpha=0.6, markersize=3, color='orange')
    ax.set_title('TD Error over Episodes')
    ax.set_xlabel('Episode')
    ax.set_ylabel('TD Error')
    ax.grid(True, alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, 'TD Error data\nnot available', 
                     ha='center', va='center', fontsize=12)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)

# Plot 4: Emotion (if available)
if 'emotion' in df.columns:
    ax = axes[1, 1]
    ax.plot(episodes, emotions, 'o-', alpha=0.6, markersize=3, color='purple')
    ax.set_title('Emotion over Episodes')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Emotion')
    ax.grid(True, alpha=0.3)
else:
    axes[1, 1].text(0.5, 0.5, 'Emotion data\nnot available', 
                     ha='center', va='center', fontsize=12)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)

plt.tight_layout()

output_path = 'option_a_comprehensive_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"📊 Visualisierung gespeichert: {output_path}\n")
plt.close()

# Nächste Schritte
print("="*60)
print("🚀 NÄCHSTE SCHRITTE")
print("="*60 + "\n")

if readiness >= 4:
    print("✅ OPTION B EMPFOHLEN:")
    print("   Performance Stability Analyzer (PSA) integrieren\n")
    print("   Aktion:")
    print("   1. Öffne training/train_finetuning.py")
    print("   2. Füge PSA-Integration hinzu (siehe PHASE_7_AKTIONSPLAN.md)")
    print("   3. Starte neues Training mit PSA")
elif readiness >= 3:
    print("⚠️  OPTIMIERUNG EMPFOHLEN:")
    print("   Mehr Episoden oder Phase 7 Features aktivieren\n")
    print("   Optionen:")
    print("   A. Längeres Training (1000+ Episoden)")
    print("   B. Phase 7 Features aktivieren (BHO, ACM)")
    print("   C. Hyperparameter manuell optimieren")
else:
    print("❌ BASELINE-VERBESSERUNG NOTWENDIG:")
    print("   System-Debugging erforderlich\n")
    print("   Aktion:")
    print("   1. Überprüfe Emotion-Engine-Parameter")
    print("   2. Validiere Reward-Shaping")
    print("   3. Teste einfachere Konfiguration")

print("\n" + "="*60)
print("✅ ANALYSE ABGESCHLOSSEN")
print("="*60 + "\n")

print(f"📁 Ergebnisse:")
print(f"   - Training Log:    {log_path}")
print(f"   - Visualisierung:  {output_path}")
print(f"   - Aktionsplan:     PHASE_7_AKTIONSPLAN.md\n")

