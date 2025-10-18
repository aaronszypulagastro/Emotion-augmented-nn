"""Einfacher Status-Check"""
import pandas as pd
import numpy as np

try:
    print("Lade Option B Daten...")
    df = pd.read_csv("results/training_log.csv")
    print(f"Erfolgreich! {len(df)} Zeilen geladen")
    print(f"Spalten: {len(df.columns)}")
    print(f"Letzte Episode: {df.iloc[-1, 0]}")
    
    # PSA Check
    if len(df.columns) > 30:
        print(f"\nPSA-Spalten vorhanden!")
        print(f"Letzte 5 Spalten: {list(df.columns[-5:])}")
    
    # Returns
    returns = df.iloc[:, 1].values
    print(f"\nPerformance:")
    print(f"  Mean: {np.mean(returns):.2f}")
    print(f"  Max: {np.max(returns):.2f}")
    
except Exception as e:
    print(f"FEHLER: {e}")
    import traceback
    traceback.print_exc()

