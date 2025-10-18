"""
Training Progress Monitor
=========================

Überwacht den Trainingsfortschritt in Echtzeit.
"""

import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

log_path = "results/training_log.csv"

print("╔══════════════════════════════════════════════════════╗")
print("║         Training Progress Monitor                    ║")
print("╚══════════════════════════════════════════════════════╝\n")

print(f"📊 Monitoring: {log_path}")
print(f"🕐 Started: {datetime.now().strftime('%H:%M:%S')}\n")

last_episode = 0
start_time = time.time()

while True:
    try:
        if not os.path.exists(log_path):
            print("⏳ Warte auf Training-Start...")
            time.sleep(5)
            continue
        
        # Lese CSV
        try:
            df = pd.read_csv(log_path, on_bad_lines='skip')
        except:
            df = pd.read_csv(log_path, usecols=[0, 1, 2, 3, 4, 5], on_bad_lines='skip')
        
        if len(df) == 0:
            time.sleep(5)
            continue
        
        # Aktueller Episode
        current_episode = len(df) - 1
        
        # Nur Update wenn neue Episoden
        if current_episode > last_episode:
            last_episode = current_episode
            
            # Berechne Metriken
            reward_col = 'return' if 'return' in df.columns else 'reward'
            returns = df[reward_col].values
            
            # avg100
            if len(returns) >= 100:
                avg100 = np.mean(returns[-100:])
            else:
                avg100 = np.mean(returns)
            
            # Aktuelle Episode
            current_return = returns[-1]
            
            # Best so far
            best_return = np.max(returns)
            
            # TD Error
            if 'td_error' in df.columns:
                td_error = df['td_error'].values[-1]
            else:
                td_error = 0.0
            
            # Emotion
            if 'emotion' in df.columns:
                emotion = df['emotion'].values[-1]
            else:
                emotion = 0.0
            
            # Geschätzte verbleibende Zeit
            elapsed = time.time() - start_time
            episodes_per_sec = current_episode / elapsed if elapsed > 0 else 0
            remaining_episodes = 500 - current_episode
            remaining_time = remaining_episodes / episodes_per_sec if episodes_per_sec > 0 else 0
            
            # Progress bar
            progress = (current_episode / 500) * 100
            bar_length = 30
            filled = int(bar_length * current_episode / 500)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            # Clear screen und print status
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("╔══════════════════════════════════════════════════════╗")
            print("║         Training Progress Monitor                    ║")
            print("╚══════════════════════════════════════════════════════╝\n")
            
            print(f"📊 Episode: {current_episode}/500 ({progress:.1f}%)")
            print(f"[{bar}]\n")
            
            print(f"⏱️  Zeit:")
            print(f"   Vergangen:  {elapsed/60:.1f} Minuten")
            print(f"   Verbleibend: {remaining_time/60:.1f} Minuten (geschätzt)")
            print(f"   Speed:      {episodes_per_sec:.2f} ep/s\n")
            
            print(f"🎯 Performance:")
            print(f"   Aktuell:    {current_return:.2f}")
            print(f"   avg100:     {avg100:.2f}")
            print(f"   Best:       {best_return:.2f}")
            print(f"   TD Error:   {td_error:.3f}")
            print(f"   Emotion:    {emotion:.3f}\n")
            
            # Vergleich mit Targets
            if avg100 > 63.86:
                status = "🚀 BESSER als vorher!"
            elif avg100 > 40.05:
                status = "✅ Über Phase 6.1"
            elif avg100 > 25.90:
                status = "⚠️  Über Phase 6.3"
            else:
                status = "❌ Unter Phase 6.3"
            
            print(f"📈 Status: {status}")
            
            if current_episode >= 500:
                print("\n✅ TRAINING ABGESCHLOSSEN!")
                break
        
        time.sleep(10)  # Update alle 10 Sekunden
        
    except KeyboardInterrupt:
        print("\n\n⏸️  Monitor gestoppt.")
        break
    except Exception as e:
        print(f"⚠️  Fehler: {e}")
        time.sleep(5)

print(f"\n🕐 Beendet: {datetime.now().strftime('%H:%M:%S')}")
print("\n💾 Training-Log verfügbar in: results/training_log.csv")
print("📊 Führe Analyse aus: python analyze_phase7_readiness.py\n")

