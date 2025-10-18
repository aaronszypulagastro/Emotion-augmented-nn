"""
Quick Progress Check - Während Training läuft
"""
import pandas as pd
import numpy as np

log_path = "results/training_log.csv"

try:
    df = pd.read_csv(log_path, on_bad_lines='skip')
    
    reward_col = 'return' if 'return' in df.columns else 'reward'
    returns = df[reward_col].values
    
    print("╔══════════════════════════════════════════════════════╗")
    print("║      ZWISCHENANALYSE - Training läuft                ║")
    print("╚══════════════════════════════════════════════════════╝\n")
    
    print(f"📊 Aktueller Stand:")
    print(f"   Episoden aufgezeichnet: {len(df)}")
    print(f"   Letzte Episode:         {df.iloc[-1, 0] if len(df) > 0 else 0:.0f}")
    
    # Letzte Returns
    last_10 = returns[-10:]
    print(f"\n🎯 Letzte 10 Episoden:")
    print(f"   Mean:   {np.mean(last_10):.2f}")
    print(f"   Max:    {np.max(last_10):.2f}")
    print(f"   Min:    {np.min(last_10):.2f}")
    
    # Gesamt
    print(f"\n📈 Gesamt-Performance:")
    print(f"   Mean:   {np.mean(returns):.2f}")
    print(f"   Max:    {np.max(returns):.2f}")
    print(f"   Std:    {np.std(returns):.2f}")
    
    # PSA Check
    if 'psa_stability_score' in df.columns:
        psa_scores = df['psa_stability_score'].dropna()
        if len(psa_scores) > 0:
            print(f"\n📊 PSA Metriken (Option B aktiv!):")
            print(f"   Stability Score: {psa_scores.iloc[-1]:.3f}")
            print(f"   Trend: {df['psa_trend'].iloc[-1] if 'psa_trend' in df.columns else 'N/A'}")
            print(f"   ✅ PSA funktioniert!")
        else:
            print(f"\n⚠️  PSA Daten noch nicht verfügbar (zu früh im Training)")
    else:
        print(f"\n⚠️  PSA Spalten nicht gefunden - Option B nicht aktiv?")
    
    # Vergleich mit Option A
    option_a_avg = 63.86
    current_avg = np.mean(returns)
    
    print(f"\n🔄 Vergleich:")
    print(f"   Option A (ohne PSA): {option_a_avg:.2f}")
    print(f"   Aktuell (mit PSA):   {current_avg:.2f}")
    
    if current_avg > option_a_avg:
        improvement = ((current_avg - option_a_avg) / option_a_avg) * 100
        print(f"   Status: ✅ +{improvement:.1f}% BESSER!")
    else:
        decline = ((option_a_avg - current_avg) / option_a_avg) * 100
        print(f"   Status: ⚠️  -{decline:.1f}%")
    
    print(f"\n💡 Training läuft weiter...")
    print(f"   Geschätzt noch: ~{500 - df.iloc[-1, 0]:.0f} Episoden")
    
except Exception as e:
    print(f"❌ Fehler: {e}")

