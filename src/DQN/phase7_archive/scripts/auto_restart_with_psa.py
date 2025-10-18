"""
Automatischer Neustart mit PSA nach Training-Abschluss
======================================================

Wartet auf Trainings-Ende, sichert Logs, startet neues Training mit PSA.
"""

import pandas as pd
import numpy as np
import time
import os
import shutil
import subprocess
from datetime import datetime

log_path = "results/training_log.csv"
target_episodes = 500

print("╔══════════════════════════════════════════════════════════════╗")
print("║     Auto-Restart: Warte auf Training-Ende, dann PSA-Start   ║")
print("╚══════════════════════════════════════════════════════════════╝\n")

print(f"⏳ Warte bis aktuelles Training abgeschlossen ist...")
print(f"📊 Ziel: {target_episodes} Episoden\n")

last_count = 0
start_time = time.time()

# Phase 1: Warten auf Training-Ende
while True:
    try:
        if not os.path.exists(log_path):
            time.sleep(10)
            continue
        
        # Lese CSV
        try:
            df = pd.read_csv(log_path, on_bad_lines='skip')
        except:
            time.sleep(10)
            continue
        
        if len(df) == 0:
            time.sleep(10)
            continue
        
        # Letzte Episode aus erster Spalte
        current_episode = int(df.iloc[-1, 0])
        
        # Progress Update
        if current_episode != last_count:
            last_count = current_episode
            progress = (current_episode / target_episodes) * 100
            elapsed = (time.time() - start_time) / 60
            print(f"📈 Episode {current_episode}/{target_episodes} ({progress:.1f}%) - "
                  f"Zeit: {elapsed:.1f} min - {datetime.now().strftime('%H:%M:%S')}")
        
        # Training abgeschlossen?
        if current_episode >= target_episodes:
            print(f"\n✅ Altes Training abgeschlossen nach {(time.time() - start_time)/60:.1f} Minuten!")
            break
        
        time.sleep(15)  # Check alle 15 Sekunden
        
    except KeyboardInterrupt:
        print("\n⏸️  Prozess abgebrochen.")
        exit(0)
    except Exception as e:
        print(f"⚠️  Fehler: {e}")
        time.sleep(10)

# Kurz warten damit alles geschrieben wurde
time.sleep(5)

# Phase 2: Logs sichern
print("\n" + "="*60)
print("💾 SICHERE ALTE LOGS (Option A - ohne PSA)")
print("="*60)

backup_name = f"training_log_option_a_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
backup_path = f"results/{backup_name}"

try:
    shutil.copy(log_path, backup_path)
    print(f"✅ Backup erstellt: {backup_path}")
except Exception as e:
    print(f"⚠️  Backup-Fehler: {e}")

# Alte Log-Datei löschen für neues Training
try:
    os.remove(log_path)
    print(f"✅ Alte training_log.csv entfernt")
except Exception as e:
    print(f"⚠️  Lösch-Fehler: {e}")

# Phase 3: Finale Analyse des alten Trainings
print("\n" + "="*60)
print("📊 FINALE ANALYSE - OPTION A (ohne PSA)")
print("="*60 + "\n")

try:
    df = pd.read_csv(backup_path, on_bad_lines='skip')
    reward_col = 'return' if 'return' in df.columns else 'reward'
    returns = df[reward_col].values
    
    avg100 = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
    best = np.max(returns)
    worst = np.min(returns)
    std = np.std(returns)
    
    print(f"📈 Option A Finale Metriken:")
    print(f"   Episoden:    {len(returns)}")
    print(f"   avg100:      {avg100:.2f}")
    print(f"   Best:        {best:.2f}")
    print(f"   Worst:       {worst:.2f}")
    print(f"   Std Dev:     {std:.2f}")
    print(f"   CV:          {std / (abs(avg100) + 1e-8):.3f}")
    
except Exception as e:
    print(f"⚠️  Analyse-Fehler: {e}")

# Phase 4: Neues Training mit PSA starten
print("\n" + "="*60)
print("🚀 STARTE NEUES TRAINING MIT PSA (Option B)")
print("="*60 + "\n")

print("⚙️  Konfiguration:")
print("   - Performance Stability Analyzer: ✅ AKTIV")
print("   - Window Size: 100")
print("   - Anomaly Threshold: 3.0")
print("   - Stability Reports: Alle 50 Episoden")
print("   - CSV erweitert: 5 neue PSA-Spalten\n")

print("🏃 Training startet jetzt...\n")
print("="*60 + "\n")

# Starte Training
try:
    subprocess.run(
        ["python", "training\\train_finetuning.py"],
        check=False
    )
except KeyboardInterrupt:
    print("\n⏸️  Training manuell gestoppt.")
except Exception as e:
    print(f"\n❌ Fehler beim Training-Start: {e}")

print("\n" + "="*60)
print("✅ PROZESS ABGESCHLOSSEN")
print("="*60)
print(f"\n📁 Logs:")
print(f"   Option A (alt):  {backup_path}")
print(f"   Option B (neu):  {log_path}")
print("\n💡 Vergleichs-Analyse:")
print("   python comprehensive_analysis.py\n")

