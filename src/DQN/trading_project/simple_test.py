"""
Einfacher Test des Trading-Systems ohne externe Dependencies
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import unserer Trading-Komponenten
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_trading_emotion_engine():
    """Teste die Trading Emotion Engine"""
    
    print("🧠 Teste Trading Emotion Engine...")
    
    try:
        from agents.trading_emotion_engine import TradingEmotionEngine, TradingEmotion
        
        # Erstelle Emotion Engine
        emotion_engine = TradingEmotionEngine()
        
        print(f"✅ Initial Emotion: {emotion_engine.current_emotion.value}")
        print(f"✅ Risk Tolerance: {emotion_engine.get_risk_tolerance():.2f}")
        
        # Simuliere verschiedene Szenarien
        scenarios = [
            {"name": "Gute Performance", "return": 0.02, "win_rate": 0.7, "drawdown": 0.01},
            {"name": "Schlechte Performance", "return": -0.015, "win_rate": 0.3, "drawdown": 0.05},
            {"name": "Volatile Märkte", "return": 0.005, "win_rate": 0.5, "drawdown": 0.03}
        ]
        
        for scenario in scenarios:
            print(f"\n--- {scenario['name']} ---")
            
            # Update Market Sentiment
            emotion_engine.update_market_sentiment(
                price_change=scenario['return'],
                volume_change=0.1,
                volatility=0.02,
                trend_strength=scenario['return'] * 2
            )
            
            # Update Performance
            emotion_engine.update_performance(
                portfolio_return=scenario['return'],
                trade_return=scenario['return'],
                drawdown=scenario['drawdown'],
                win_rate=scenario['win_rate']
            )
            
            print(f"Emotion: {emotion_engine.current_emotion.value}")
            print(f"Risk Tolerance: {emotion_engine.get_risk_tolerance():.2f}")
            print(f"Position Sizing: {emotion_engine.get_position_sizing_modifier():.2f}")
        
        print("\n✅ Trading Emotion Engine Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ Fehler bei Trading Emotion Engine Test: {e}")
        return False

def test_paper_trading_system():
    """Teste das Paper Trading System"""
    
    print("\n📊 Teste Paper Trading System...")
    
    try:
        from paper_trading_system import PaperTradingSystem
        
        # Erstelle Paper Trading System
        paper_system = PaperTradingSystem(
            initial_capital=10000.0,
            commission=0.001,
            slippage=0.0005,
            min_trade_size=100.0
        )
        
        print(f"✅ Initial Capital: ${paper_system.initial_capital:,.2f}")
        print(f"✅ Commission: {paper_system.commission*100:.1f}%")
        print(f"✅ Slippage: {paper_system.slippage*100:.2f}%")
        
        # Simuliere Trades
        print("\n--- Simuliere Trades ---")
        
        # Kauf
        result1 = paper_system.execute_trade(
            symbol='AAPL',
            action='buy',
            quantity=10.0,
            price=150.0
        )
        
        if result1['success']:
            print(f"✅ Kauf erfolgreich: {result1['trade_record']['quantity']} AAPL @ ${result1['execution_price']:.2f}")
            print(f"   Commission: ${result1['commission']:.2f}")
            print(f"   Cash nach Trade: ${result1['trade_record']['cash_after']:,.2f}")
        else:
            print(f"❌ Kauf fehlgeschlagen: {result1['reason']}")
        
        # Verkauf
        result2 = paper_system.execute_trade(
            symbol='AAPL',
            action='sell',
            quantity=5.0,
            price=155.0
        )
        
        if result2['success']:
            print(f"✅ Verkauf erfolgreich: {result2['trade_record']['quantity']} AAPL @ ${result2['execution_price']:.2f}")
            print(f"   Commission: ${result2['commission']:.2f}")
            print(f"   Cash nach Trade: ${result2['trade_record']['cash_after']:,.2f}")
        else:
            print(f"❌ Verkauf fehlgeschlagen: {result2['reason']}")
        
        # Performance Metrics
        metrics = paper_system.get_performance_metrics()
        print(f"\n📈 Performance Metrics:")
        print(f"   Total Trades: {metrics['total_trades']}")
        print(f"   Total Commission Paid: ${metrics['total_commission_paid']:.2f}")
        print(f"   Total Slippage Cost: ${metrics['total_slippage_cost']:.2f}")
        print(f"   Final Portfolio Value: ${metrics['final_portfolio_value']:,.2f}")
        
        print("\n✅ Paper Trading System Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ Fehler bei Paper Trading System Test: {e}")
        return False

def test_synthetic_trading_environment():
    """Teste Trading Environment mit synthetischen Daten"""
    
    print("\n🏗️ Teste Trading Environment (Synthetic Data)...")
    
    try:
        from environments.trading_environment import TradingEnvironment
        
        # Erstelle Environment mit synthetischen Daten
        env = TradingEnvironment(
            symbol='TEST',
            timeframe='15m',
            initial_capital=10000.0,
            max_position_size=0.2
        )
        
        print(f"✅ Environment erstellt: {env.symbol}")
        print(f"✅ Data Points: {len(env.data)}")
        print(f"✅ State Size: {env.observation_space.shape[0]}")
        print(f"✅ Action Size: {env.action_space.shape[0]}")
        
        # Teste Environment
        state, info = env.reset()
        print(f"✅ Initial State Shape: {state.shape}")
        print(f"✅ Initial Portfolio: ${info['portfolio_value']:,.2f}")
        
        # Simuliere einige Schritte
        for step in range(5):
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            
            print(f"   Step {step}: Action={action[0]:.3f}, Reward={reward:.4f}, Portfolio=${info['portfolio_value']:.2f}")
            
            if done:
                break
        
        # Performance Metrics
        metrics = env.get_performance_metrics()
        print(f"\n📈 Environment Performance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.4f}")
        
        print("\n✅ Trading Environment Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ Fehler bei Trading Environment Test: {e}")
        return False

def test_emotion_trading_agent():
    """Teste den Emotion Trading Agent"""
    
    print("\n🤖 Teste Emotion Trading Agent...")
    
    try:
        from agents.emotion_trading_agent import EmotionTradingAgent
        
        # Erstelle Agent
        agent = EmotionTradingAgent(
            state_size=7,
            action_size=1,
            learning_rate=1e-4,
            epsilon_decay=0.995
        )
        
        print(f"✅ Agent erstellt")
        print(f"✅ Epsilon: {agent.epsilon:.3f}")
        print(f"✅ Current Emotion: {agent.emotion_engine.current_emotion.value}")
        
        # Teste Action Selection
        test_state = np.random.randn(7).astype(np.float32)
        action = agent.select_action(test_state, training=True)
        
        print(f"✅ Action Selection: {action:.3f}")
        
        # Teste Emotion Update
        agent.update_emotion_engine(
            portfolio_return=0.01,
            trade_return=0.005,
            drawdown=0.02,
            win_rate=0.6,
            price_change=0.01,
            volume_change=0.1,
            volatility=0.02
        )
        
        print(f"✅ Emotion nach Update: {agent.emotion_engine.current_emotion.value}")
        print(f"✅ Risk Tolerance: {agent.emotion_engine.get_risk_tolerance():.2f}")
        
        # Trading Metrics
        metrics = agent.get_trading_metrics()
        print(f"\n📈 Agent Trading Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.4f}")
        
        print("\n✅ Emotion Trading Agent Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ Fehler bei Emotion Trading Agent Test: {e}")
        return False

def run_comprehensive_test():
    """Führe umfassenden Test durch"""
    
    print("🚀 UMFASSENDER TRADING-SYSTEM TEST")
    print("=" * 50)
    
    test_results = {}
    
    # Teste alle Komponenten
    test_results['emotion_engine'] = test_trading_emotion_engine()
    test_results['paper_trading'] = test_paper_trading_system()
    test_results['trading_environment'] = test_synthetic_trading_environment()
    test_results['emotion_agent'] = test_emotion_trading_agent()
    
    # Zusammenfassung
    print("\n📋 TEST ZUSAMMENFASSUNG")
    print("=" * 30)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for component, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{component}: {status}")
    
    print(f"\n🎯 GESAMTERGEBNIS: {passed_tests}/{total_tests} Tests bestanden")
    
    if passed_tests == total_tests:
        print("🎉 ALLE TESTS ERFOLGREICH! Das Trading-System ist bereit!")
    else:
        print("⚠️ Einige Tests fehlgeschlagen. Überprüfe die Fehler.")
    
    return test_results

if __name__ == "__main__":
    run_comprehensive_test()
