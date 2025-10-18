"""
Automatische Evaluation nach Training
======================================

Wartet bis Training abgeschlossen ist und führt dann automatisch
die Analyse durch.
"""

import pandas as pd
import numpy as np
import time
import os
import subprocess
from datetime import datetime

log_path = "results/training_log.csv"
target_episodes = 500

print("╔══════════════════════════════════════════════════════╗")
print("║     Automatische Post-Training Evaluation            ║")
print("╚══════════════════════════════════════════════════════╝\n")

print(f"⏳ Warte auf Training-Abschluss ({target_episodes} Episoden)...")
print(f"📊 Monitoring: {log_path}\n")

start_time = time.time()
last_count = 0

while True:
    try:
        if not os.path.exists(log_path):
            time.sleep(5)
            continue
        
        # Lese CSV
        try:
            df = pd.read_csv(log_path, on_bad_lines='skip')
        except:
            time.sleep(5)
            continue
        
        current_episodes = len(df)
        
        # Progress Update
        if current_episodes != last_count:
            last_count = current_episodes
            progress = (current_episodes / target_episodes) * 100
            print(f"📈 Progress: {current_episodes}/{target_episodes} ({progress:.1f}%) - {datetime.now().strftime('%H:%M:%S')}")
        
        # Training abgeschlossen?
        if current_episodes >= target_episodes:
            print(f"\n✅ Training abgeschlossen nach {(time.time() - start_time)/60:.1f} Minuten!")
            break
        
        time.sleep(30)  # Check alle 30 Sekunden
        
    except KeyboardInterrupt:
        print("\n⏸️  Warte-Prozess abgebrochen.")
        exit(0)
    except Exception as e:
        print(f"⚠️  Fehler: {e}")
        time.sleep(5)

# Warte kurz um sicherzugehen dass alles geschrieben wurde
time.sleep(5)

print("\n" + "="*60)
print("🔍 STARTE AUTOMATISCHE ANALYSE")
print("="*60 + "\n")

# Führe Readiness-Analyse aus
print("1️⃣  Phase 7.0 Readiness Analysis...")
try:
    result = subprocess.run(
        ["python", "analyze_phase7_readiness.py"],
        capture_output=True,
        text=True,
        timeout=60
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"⚠️  Fehler bei Analyse: {result.stderr}")
except Exception as e:
    print(f"⚠️  Fehler beim Ausführen der Analyse: {e}")

print("\n" + "="*60)
print("✅ AUTOMATISCHE EVALUATION ABGESCHLOSSEN")
print("="*60)

# Zusammenfassung
try:
    df = pd.read_csv(log_path, on_bad_lines='skip')
    
    reward_col = 'return' if 'return' in df.columns else 'reward'
    returns = df[reward_col].values
    
    avg100 = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
    best = np.max(returns)
    
    print(f"\n📊 FINALE METRIKEN:")
    print(f"   Episoden:   {len(returns)}")
    print(f"   avg100:     {avg100:.2f}")
    print(f"   Best:       {best:.2f}")
    
    # Vergleich
    print(f"\n📈 VERGLEICH MIT VORHERIGEN PHASEN:")
    print(f"   Phase 6.1:  40.05")
    print(f"   Phase 6.3:  25.90")
    print(f"   Aktuell:    {avg100:.2f}")
    
    if avg100 > 40.05:
        improvement = ((avg100 - 40.05) / 40.05) * 100
        print(f"   Status:     ✅ +{improvement:.1f}% besser als Phase 6.1!")
    elif avg100 > 25.90:
        print(f"   Status:     ⚠️  Zwischen Phase 6.1 und 6.3")
    else:
        print(f"   Status:     ❌ Schlechter als Phase 6.3")
    
    print(f"\n📁 Ergebnisse verfügbar:")
    print(f"   - Training Log:    {log_path}")
    print(f"   - Visualisierung:  phase7_readiness_analysis.png")
    print(f"   - Aktionsplan:     PHASE_7_AKTIONSPLAN.md")
    
except Exception as e:
    print(f"\n⚠️  Fehler bei finaler Zusammenfassung: {e}")

print("\n" + "="*60)
print("🚀 NÄCHSTE SCHRITTE:")
print("="*60)
print("1. Überprüfe Visualisierung: phase7_readiness_analysis.png")
print("2. Lese detaillierten Report: PHASE_7_AKTIONSPLAN.md")
print("3. Falls Stabilität OK: Starte Option B (PSA Integration)")
print("4. Falls nicht OK: Wiederhole mit mehr Episoden")
print("\n✨ Fertig!\n")

