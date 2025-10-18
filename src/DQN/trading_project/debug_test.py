"""
Debug Test um das Problem zu identifizieren
"""

print("🔍 Debug Test gestartet...")

try:
    print("1. Importiere numpy...")
    import numpy as np
    print("✅ NumPy erfolgreich importiert")
    
    print("2. Importiere pandas...")
    import pandas as pd
    print("✅ Pandas erfolgreich importiert")
    
    print("3. Teste optimierte Komponenten...")
    from optimized_trading_system import OptimizedTradingEmotionEngine, OptimizedPaperTradingSystem, TradingEmotion
    print("✅ Optimierte Komponenten erfolgreich importiert")
    
    print("4. Erstelle Emotion Engine...")
    emotion_engine = OptimizedTradingEmotionEngine()
    print("✅ Emotion Engine erfolgreich erstellt")
    
    print("5. Erstelle Paper Trading System...")
    paper_system = OptimizedPaperTradingSystem()
    print("✅ Paper Trading System erfolgreich erstellt")
    
    print("6. Teste einfache Operationen...")
    test_array = np.array([1, 2, 3])
    test_df = pd.DataFrame({'test': [1, 2, 3]})
    print("✅ Einfache Operationen erfolgreich")
    
    print("7. Teste Emotion Engine...")
    emotion_engine.update_market_sentiment(0.01, 0.1, 0.02, 0.01)
    emotion_engine.update_performance(0.01, 0.005, 0.02, 0.6)
    print("✅ Emotion Engine Test erfolgreich")
    
    print("8. Teste Paper Trading...")
    result = paper_system.execute_trade('TEST', 'buy', 10.0, 100.0)
    print("✅ Paper Trading Test erfolgreich")
    
    print("\n🎉 ALLE TESTS ERFOLGREICH!")
    print("Das Problem liegt wahrscheinlich in der Multi-Timeframe Implementierung.")
    
except Exception as e:
    print(f"❌ FEHLER GEFUNDEN: {e}")
    print(f"Fehler-Typ: {type(e).__name__}")
    import traceback
    traceback.print_exc()
