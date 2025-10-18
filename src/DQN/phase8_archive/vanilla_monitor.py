"""Vanilla DQN Monitor - Gibt Updates alle paar Minuten"""
import time
import os
from datetime import datetime

log_path = "results/vanilla_dqn_training_log.csv"
update_file = "VANILLA_DQN_UPDATES.txt"

print("Monitoring Vanilla DQN Training...")
print(f"Log: {log_path}")
print(f"Updates: {update_file}\n")

start_time = time.time()
last_episode = 0

with open(update_file, 'w') as f:
    f.write(f"VANILLA DQN TRAINING UPDATES\n")
    f.write(f"Started: {datetime.now().strftime('%H:%M:%S')}\n")
    f.write("="*60 + "\n\n")

while True:
    try:
        if not os.path.exists(log_path):
            time.sleep(30)
            continue
        
        # Lese letzte Zeile
        with open(log_path, 'r') as f:
            lines = f.readlines()
            if len(lines) <= 1:  # Nur Header
                continue
            
            last_line = lines[-1].strip().split(',')
            current_episode = int(last_line[0])
            current_return = float(last_line[1])
            current_td = float(last_line[2])
        
        # Neues Update?
        if current_episode != last_episode:
            last_episode = current_episode
            elapsed = (time.time() - start_time) / 60
            
            update_msg = f"[{datetime.now().strftime('%H:%M:%S')}] " \
                        f"Episode {current_episode}/500 ({current_episode/5:.0f}%) | " \
                        f"Return: {current_return:.1f} | TD-Error: {current_td:.2f} | " \
                        f"Zeit: {elapsed:.1f} min\n"
            
            print(update_msg.strip())
            
            with open(update_file, 'a') as f:
                f.write(update_msg)
        
        # Fertig?
        if current_episode >= 500:
            final_msg = f"\n✅ TRAINING ABGESCHLOSSEN nach {elapsed:.1f} Minuten!\n"
            print(final_msg)
            with open(update_file, 'a') as f:
                f.write(final_msg)
            break
        
        time.sleep(60)  # Check jede Minute
        
    except KeyboardInterrupt:
        print("\nMonitoring gestoppt.")
        break
    except Exception as e:
        print(f"Fehler: {e}")
        time.sleep(30)


