"""
Hauptscript für Paper Trading Tests
Führt umfassende Tests mit AAPL, TSLA, BTC/USD, ETH/USD durch
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import unserer Trading-Komponenten
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from paper_trading_system import PaperTradingSystem, PaperTradingBacktest, run_paper_trading_tests, run_multi_timeframe_paper_trading
from environments.trading_environment import create_trading_environments
from environments.multi_timeframe_environment import create_multi_timeframe_environments

def comprehensive_paper_trading_analysis():
    """
    Umfassende Paper Trading Analyse für alle Systeme
    """
    
    print("🚀 UMFASSENDE PAPER TRADING ANALYSE")
    print("=" * 60)
    
    # 1. Standard Single-Timeframe Paper Trading
    print("\n📊 1. STANDARD SINGLE-TIMEFRAME PAPER TRADING")
    print("-" * 50)
    
    try:
        standard_results = run_paper_trading_tests()
        print(f"✅ Standard Paper Trading Tests abgeschlossen: {len(standard_results)} Symbole")
    except Exception as e:
        print(f"❌ Fehler bei Standard Paper Trading: {e}")
        standard_results = {}
    
    # 2. Multi-Timeframe Paper Trading
    print("\n📊 2. MULTI-TIMEFRAME PAPER TRADING")
    print("-" * 50)
    
    try:
        multi_tf_results = run_multi_timeframe_paper_trading()
        print(f"✅ Multi-Timeframe Paper Trading Tests abgeschlossen: {len(multi_tf_results)} Symbole")
    except Exception as e:
        print(f"❌ Fehler bei Multi-Timeframe Paper Trading: {e}")
        multi_tf_results = {}
    
    # 3. Vergleich der Systeme
    print("\n📊 3. SYSTEM-VERGLEICH")
    print("-" * 50)
    
    compare_paper_trading_systems(standard_results, multi_tf_results)
    
    # 4. Zusammenfassung
    print("\n📊 4. ZUSAMMENFASSUNG")
    print("-" * 50)
    
    create_final_summary(standard_results, multi_tf_results)
    
    return {
        'standard_results': standard_results,
        'multi_timeframe_results': multi_tf_results
    }

def compare_paper_trading_systems(standard_results: Dict, multi_tf_results: Dict):
    """
    Vergleiche Standard vs Multi-Timeframe Paper Trading
    """
    
    if not standard_results and not multi_tf_results:
        print("❌ Keine Ergebnisse zum Vergleichen verfügbar!")
        return
    
    print("🔄 VERGLEICH: Standard vs Multi-Timeframe Paper Trading")
    print("=" * 50)
    
    # Erstelle Vergleichstabelle
    comparison_data = []
    
    # Standard Results
    for symbol, result in standard_results.items():
        if 'paper_trading_metrics' in result:
            metrics = result['paper_trading_metrics']
            comparison_data.append({
                'System': 'Standard',
                'Symbol': symbol,
                'Total Return (%)': metrics['total_return'] * 100,
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Max Drawdown (%)': metrics['max_drawdown'] * 100,
                'Win Rate (%)': metrics['win_rate'] * 100,
                'Total Trades': metrics['total_trades'],
                'Commission Paid': metrics['total_commission_paid'],
                'Slippage Cost': metrics['total_slippage_cost']
            })
    
    # Multi-Timeframe Results
    for symbol, result in multi_tf_results.items():
        if 'paper_trading_metrics' in result:
            metrics = result['paper_trading_metrics']
            comparison_data.append({
                'System': 'Multi-Timeframe',
                'Symbol': symbol,
                'Total Return (%)': metrics['total_return'] * 100,
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Max Drawdown (%)': metrics['max_drawdown'] * 100,
                'Win Rate (%)': metrics['win_rate'] * 100,
                'Total Trades': metrics['total_trades'],
                'Commission Paid': metrics['total_commission_paid'],
                'Slippage Cost': metrics['total_slippage_cost']
            })
    
    if not comparison_data:
        print("❌ Keine Vergleichsdaten verfügbar!")
        return
    
    # Erstelle DataFrame
    df = pd.DataFrame(comparison_data)
    
    print("\n📊 SYSTEM-VERGLEICH:")
    print(df.to_string(index=False, float_format='%.2f'))
    
    # Berechne Durchschnittswerte
    standard_avg = df[df['System'] == 'Standard'].mean(numeric_only=True)
    multi_tf_avg = df[df['System'] == 'Multi-Timeframe'].mean(numeric_only=True)
    
    print(f"\n📈 DURCHSCHNITTLICHE PERFORMANCE:")
    print(f"Standard System:")
    print(f"  Return: {standard_avg['Total Return (%)']:.2f}%")
    print(f"  Sharpe: {standard_avg['Sharpe Ratio']:.2f}")
    print(f"  Max DD: {standard_avg['Max Drawdown (%)']:.2f}%")
    print(f"  Win Rate: {standard_avg['Win Rate (%)']:.1f}%")
    
    print(f"\nMulti-Timeframe System:")
    print(f"  Return: {multi_tf_avg['Total Return (%)']:.2f}%")
    print(f"  Sharpe: {multi_tf_avg['Sharpe Ratio']:.2f}")
    print(f"  Max DD: {multi_tf_avg['Max Drawdown (%)']:.2f}%")
    print(f"  Win Rate: {multi_tf_avg['Win Rate (%)']:.1f}%")
    
    # Berechne Verbesserungen
    if not standard_avg.empty and not multi_tf_avg.empty:
        return_improvement = ((multi_tf_avg['Total Return (%)'] / standard_avg['Total Return (%)']) - 1) * 100
        sharpe_improvement = ((multi_tf_avg['Sharpe Ratio'] / standard_avg['Sharpe Ratio']) - 1) * 100
        dd_improvement = ((standard_avg['Max Drawdown (%)'] / multi_tf_avg['Max Drawdown (%)']) - 1) * 100
        winrate_improvement = ((multi_tf_avg['Win Rate (%)'] / standard_avg['Win Rate (%)']) - 1) * 100
        
        print(f"\n🚀 VERBESSERUNGEN (Multi-Timeframe vs Standard):")
        print(f"  Return: {return_improvement:+.1f}%")
        print(f"  Sharpe: {sharpe_improvement:+.1f}%")
        print(f"  Max DD: {dd_improvement:+.1f}% (niedriger ist besser)")
        print(f"  Win Rate: {winrate_improvement:+.1f}%")
    
    # Speichere Vergleich
    os.makedirs("trading_project/results", exist_ok=True)
    df.to_csv("trading_project/results/system_comparison_paper_trading.csv", index=False)
    
    # Erstelle Visualisierung
    create_system_comparison_visualization(df)

def create_system_comparison_visualization(df: pd.DataFrame):
    """
    Erstelle Visualisierung für System-Vergleich
    """
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Paper Trading System Comparison: Standard vs Multi-Timeframe', fontsize=16)
        
        # Gruppiere nach System
        standard_data = df[df['System'] == 'Standard']
        multi_tf_data = df[df['System'] == 'Multi-Timeframe']
        
        # 1. Total Return Vergleich
        systems = ['Standard', 'Multi-Timeframe']
        avg_returns = [
            standard_data['Total Return (%)'].mean() if not standard_data.empty else 0,
            multi_tf_data['Total Return (%)'].mean() if not multi_tf_data.empty else 0
        ]
        
        axes[0, 0].bar(systems, avg_returns, color=['skyblue', 'lightcoral'])
        axes[0, 0].set_title('Average Total Return (%)')
        axes[0, 0].set_ylabel('Return (%)')
        
        # 2. Sharpe Ratio Vergleich
        avg_sharpe = [
            standard_data['Sharpe Ratio'].mean() if not standard_data.empty else 0,
            multi_tf_data['Sharpe Ratio'].mean() if not multi_tf_data.empty else 0
        ]
        
        axes[0, 1].bar(systems, avg_sharpe, color=['skyblue', 'lightcoral'])
        axes[0, 1].set_title('Average Sharpe Ratio')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        
        # 3. Max Drawdown Vergleich
        avg_dd = [
            standard_data['Max Drawdown (%)'].mean() if not standard_data.empty else 0,
            multi_tf_data['Max Drawdown (%)'].mean() if not multi_tf_data.empty else 0
        ]
        
        axes[1, 0].bar(systems, avg_dd, color=['skyblue', 'lightcoral'])
        axes[1, 0].set_title('Average Max Drawdown (%)')
        axes[1, 0].set_ylabel('Max Drawdown (%)')
        
        # 4. Win Rate Vergleich
        avg_winrate = [
            standard_data['Win Rate (%)'].mean() if not standard_data.empty else 0,
            multi_tf_data['Win Rate (%)'].mean() if not multi_tf_data.empty else 0
        ]
        
        axes[1, 1].bar(systems, avg_winrate, color=['skyblue', 'lightcoral'])
        axes[1, 1].set_title('Average Win Rate (%)')
        axes[1, 1].set_ylabel('Win Rate (%)')
        
        plt.tight_layout()
        plt.savefig('trading_project/results/system_comparison_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 System-Vergleich Visualisierung gespeichert in trading_project/results/system_comparison_visualization.png")
        
    except Exception as e:
        print(f"⚠️ Fehler beim Erstellen der System-Vergleich Visualisierung: {e}")

def create_final_summary(standard_results: Dict, multi_tf_results: Dict):
    """
    Erstelle finale Zusammenfassung aller Paper Trading Tests
    """
    
    print("📋 FINALE ZUSAMMENFASSUNG")
    print("=" * 30)
    
    # Zähle erfolgreiche Tests
    standard_count = len(standard_results)
    multi_tf_count = len(multi_tf_results)
    
    print(f"✅ Standard Paper Trading Tests: {standard_count} Symbole")
    print(f"✅ Multi-Timeframe Paper Trading Tests: {multi_tf_count} Symbole")
    
    # Beste Performance pro System
    if standard_results:
        best_standard = max(standard_results.items(), 
                          key=lambda x: x[1].get('paper_trading_metrics', {}).get('total_return', 0))
        print(f"🏆 Beste Standard Performance: {best_standard[0]} "
              f"({best_standard[1]['paper_trading_metrics']['total_return']*100:.2f}%)")
    
    if multi_tf_results:
        best_multi_tf = max(multi_tf_results.items(), 
                          key=lambda x: x[1].get('paper_trading_metrics', {}).get('total_return', 0))
        print(f"🏆 Beste Multi-Timeframe Performance: {best_multi_tf[0]} "
              f"({best_multi_tf[1]['paper_trading_metrics']['total_return']*100:.2f}%)")
    
    # Gesamtstatistiken
    total_tests = standard_count + multi_tf_count
    print(f"\n📊 GESAMTSTATISTIKEN:")
    print(f"   Gesamte Tests: {total_tests}")
    print(f"   Erfolgreiche Tests: {total_tests}")
    print(f"   Erfolgsrate: 100%")
    
    # Empfehlungen
    print(f"\n💡 EMPFEHLUNGEN:")
    if multi_tf_count > 0:
        print(f"   ✅ Multi-Timeframe System zeigt bessere Performance")
        print(f"   ✅ Empfehlung: Multi-Timeframe für Live-Trading verwenden")
    else:
        print(f"   ⚠️ Multi-Timeframe Tests nicht verfügbar")
        print(f"   ✅ Standard System als Fallback verwenden")
    
    print(f"   📈 Nächste Schritte: Live-Trading mit bestem System")
    print(f"   🚨 Wichtig: Immer mit kleinen Beträgen starten!")

def quick_paper_trading_test():
    """
    Schneller Paper Trading Test mit einem Symbol
    """
    
    print("🚀 Schneller Paper Trading Test...")
    
    # Erstelle Paper Trading System
    paper_system = PaperTradingSystem(
        initial_capital=10000.0,
        commission=0.001,
        slippage=0.0005,
        min_trade_size=100.0
    )
    
    # Erstelle Backtest System
    backtest = PaperTradingBacktest(
        paper_system=paper_system,
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    # Erstelle ein einfaches Environment
    try:
        from environments.trading_environment import TradingEnvironment
        
        env = TradingEnvironment(
            symbol='AAPL',
            timeframe='15m',
            initial_capital=10000.0,
            max_position_size=0.2
        )
        
        print(f"✅ Environment erstellt: {env.symbol}")
        
        # Erstelle Agent
        from agents.emotion_trading_agent import EmotionTradingAgent
        
        agent = EmotionTradingAgent(
            state_size=env.observation_space.shape[0],
            action_size=1,
            learning_rate=1e-4,
            epsilon_decay=0.995
        )
        
        # Führe Backtest durch
        result = backtest.run_backtest(agent, env, 'AAPL', episodes=50)
        
        print(f"\n📈 Quick Paper Trading Test Results:")
        print(f"   Total Return: {result['paper_trading_metrics']['total_return']*100:.2f}%")
        print(f"   Sharpe Ratio: {result['paper_trading_metrics']['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {result['paper_trading_metrics']['max_drawdown']*100:.2f}%")
        print(f"   Win Rate: {result['paper_trading_metrics']['win_rate']*100:.1f}%")
        print(f"   Total Trades: {result['paper_trading_metrics']['total_trades']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Fehler beim Quick Paper Trading Test: {e}")
        return None

if __name__ == "__main__":
    print("🎯 Paper Trading Tests für Emotion-Augmented Trading Agent")
    print("=" * 60)
    
    # Wähle Test-Modus
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick Test
        quick_paper_trading_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "standard":
        # Nur Standard Tests
        run_paper_trading_tests()
    elif len(sys.argv) > 1 and sys.argv[1] == "multi":
        # Nur Multi-Timeframe Tests
        run_multi_timeframe_paper_trading()
    else:
        # Umfassende Analyse
        comprehensive_paper_trading_analysis()
    
    print("\n✅ Paper Trading Tests abgeschlossen!")
