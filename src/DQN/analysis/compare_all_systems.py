"""Quick comparison of all systems"""
import pandas as pd
import numpy as np

print("\n" + "="*80)
print("COMPLETE SYSTEM COMPARISON - PHASE 8 OVERVIEW")
print("="*80 + "\n")

systems = [
    ("Vanilla DQN", "results/vanilla_dqn_training_log.csv", "CartPole"),
    ("Competitive Self-Play", "results/competitive_selfplay_log.csv", "CartPole"),
    ("Winner Mindset", "results/acrobot_winner_mindset_log.csv", "Acrobot"),
]

print(f"{'System':<25} {'Environment':<12} {'Avg100':<10} {'Best':<10} {'Emo Std':<10} {'Status'}")
print("-"*80)

for name, file, env in systems:
    try:
        df = pd.read_csv(file)
        
        returns = df['return'].values
        avg100 = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
        best = np.max(returns)
        
        if 'emotion' in df.columns:
            emo_std = df['emotion'].std()
            emo_str = f"{emo_std:.3f}"
        else:
            emo_str = "N/A"
        
        status = "OK" if avg100 > 0 or avg100 > -200 else "POOR"
        
        print(f"{name:<25} {env:<12} {avg100:<10.1f} {best:<10.1f} {emo_str:<10} {status}")
        
    except Exception as e:
        print(f"{name:<25} {env:<12} {'NOT FOUND':<10} {'N/A':<10} {'N/A':<10} ERROR")

print("\n" + "="*80 + "\n")





