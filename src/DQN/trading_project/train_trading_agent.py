"""
Training Script für Emotion-Augmented Trading Agent
Testet verschiedene Märkte und Timeframes
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
from environments.trading_environment import TradingEnvironment, create_trading_environments
from agents.emotion_trading_agent import EmotionTradingAgent, train_emotion_trading_agent

def run_trading_experiments():
    """
    Führe Trading-Experimente mit verschiedenen Märkten durch
    """
    
    print("🚀 Starte Trading-Experimente...")
    
    # Erstelle Environments
    print("\n📊 Erstelle Trading Environments...")
    envs = create_trading_environments()
    
    if not envs:
        print("❌ Keine Environments erstellt!")
        return
    
    print(f"✅ {len(envs)} Environments erstellt")
    
    # Experimente
    results = {}
    
    for env_name, env in envs.items():
        print(f"\n🎯 Teste {env_name}...")
        print(f"   Symbol: {env.symbol}")
        print(f"   Timeframe: {env.timeframe}")
        print(f"   Data Points: {len(env.data)}")
        
        try:
            # Training
            metrics = train_emotion_trading_agent(
                env=env,
                episodes=200,  # Reduziert für schnelleren Test
                save_interval=50,
                model_path=f"trading_project/results/{env_name}_model"
            )
            
            results[env_name] = metrics
            
            print(f"✅ {env_name} Training abgeschlossen")
            print(f"   Final Return: {metrics['total_return']*100:.2f}%")
            print(f"   Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"   Final Emotion: {metrics['current_emotion']}")
            
        except Exception as e:
            print(f"❌ Fehler bei {env_name}: {e}")
            continue
    
    # Analysiere Ergebnisse
    analyze_results(results)
    
    return results

def analyze_results(results: Dict):
    """
    Analysiere Trading-Ergebnisse
    """
    
    print("\n📈 ANALYSE DER TRADING-ERGEBNISSE")
    print("=" * 50)
    
    if not results:
        print("❌ Keine Ergebnisse zu analysieren!")
        return
    
    # Erstelle Vergleichstabelle
    comparison_data = []
    
    for env_name, metrics in results.items():
        comparison_data.append({
            'Environment': env_name,
            'Total Return (%)': metrics['total_return'] * 100,
            'Win Rate (%)': metrics['win_rate'] * 100,
            'Total Trades': metrics['total_trades'],
            'Total Profit': metrics['total_profit'],
            'Avg Profit/Trade': metrics['avg_profit_per_trade'],
            'Final Emotion': metrics['current_emotion'],
            'Risk Tolerance': metrics['risk_tolerance'],
            'Final Portfolio': metrics['final_portfolio_value']
        })
    
    # Erstelle DataFrame
    df = pd.DataFrame(comparison_data)
    
    print("\n📊 PERFORMANCE VERGLEICH:")
    print(df.to_string(index=False, float_format='%.2f'))
    
    # Finde beste Performance
    best_return = df.loc[df['Total Return (%)'].idxmax()]
    best_winrate = df.loc[df['Win Rate (%)'].idxmax()]
    
    print(f"\n🏆 BESTE PERFORMANCE:")
    print(f"   Höchster Return: {best_return['Environment']} ({best_return['Total Return (%)']:.2f}%)")
    print(f"   Beste Win Rate: {best_winrate['Environment']} ({best_winrate['Win Rate (%)']:.1f}%)")
    
    # Emotion-Analyse
    print(f"\n🧠 EMOTION-ANALYSE:")
    emotion_counts = df['Final Emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"   {emotion}: {count} Environments")
    
    # Speichere Ergebnisse
    save_results(results, df)
    
    # Erstelle Visualisierungen
    create_visualizations(results, df)

def save_results(results: Dict, df: pd.DataFrame):
    """
    Speichere Ergebnisse in Dateien
    """
    
    # Erstelle results Ordner falls nicht vorhanden
    os.makedirs("trading_project/results", exist_ok=True)
    
    # Speichere Vergleichstabelle
    df.to_csv("trading_project/results/trading_comparison.csv", index=False)
    
    # Speichere detaillierte Ergebnisse
    detailed_results = {}
    for env_name, metrics in results.items():
        detailed_results[env_name] = {
            'metrics': metrics,
            'episode_rewards': metrics.get('episode_rewards', []),
            'episode_returns': metrics.get('episode_returns', []),
            'episode_emotions': metrics.get('episode_emotions', [])
        }
    
    # Speichere als JSON
    import json
    with open("trading_project/results/detailed_results.json", 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"\n💾 Ergebnisse gespeichert in trading_project/results/")

def create_visualizations(results: Dict, df: pd.DataFrame):
    """
    Erstelle Visualisierungen der Ergebnisse
    """
    
    try:
        # Erstelle Figure mit Subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Emotion-Augmented Trading Agent - Performance Analysis', fontsize=16)
        
        # 1. Total Return Vergleich
        axes[0, 0].bar(df['Environment'], df['Total Return (%)'])
        axes[0, 0].set_title('Total Return (%) by Environment')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Win Rate Vergleich
        axes[0, 1].bar(df['Environment'], df['Win Rate (%)'])
        axes[0, 1].set_title('Win Rate (%) by Environment')
        axes[0, 1].set_ylabel('Win Rate (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Emotion Distribution
        emotion_counts = df['Final Emotion'].value_counts()
        axes[1, 0].pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Final Emotion Distribution')
        
        # 4. Risk vs Return
        axes[1, 1].scatter(df['Risk Tolerance'], df['Total Return (%)'], s=100, alpha=0.7)
        for i, env in enumerate(df['Environment']):
            axes[1, 1].annotate(env, (df['Risk Tolerance'].iloc[i], df['Total Return (%)'].iloc[i]))
        axes[1, 1].set_xlabel('Risk Tolerance')
        axes[1, 1].set_ylabel('Total Return (%)')
        axes[1, 1].set_title('Risk vs Return')
        
        plt.tight_layout()
        plt.savefig('trading_project/results/trading_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualisierungen gespeichert in trading_project/results/trading_analysis.png")
        
    except Exception as e:
        print(f"⚠️ Fehler beim Erstellen der Visualisierungen: {e}")

def quick_test():
    """
    Schneller Test mit einem Environment
    """
    
    print("🚀 Schneller Trading Test...")
    
    # Erstelle ein einfaches Environment
    try:
        env = TradingEnvironment(
            symbol='AAPL',
            timeframe='15m',
            initial_capital=10000.0,
            max_position_size=0.2
        )
        
        print(f"✅ Environment erstellt: {env.symbol}")
        print(f"   Data Points: {len(env.data)}")
        
        # Kurzer Test
        metrics = train_emotion_trading_agent(
            env=env,
            episodes=50,
            save_interval=25,
            model_path="trading_project/results/quick_test_model"
        )
        
        print(f"\n📈 Quick Test Results:")
        print(f"   Final Return: {metrics['total_return']*100:.2f}%")
        print(f"   Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"   Final Emotion: {metrics['current_emotion']}")
        print(f"   Total Trades: {metrics['total_trades']}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Fehler beim Quick Test: {e}")
        return None

if __name__ == "__main__":
    print("🎯 Emotion-Augmented Trading Agent Training")
    print("=" * 50)
    
    # Wähle Test-Modus
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick Test
        quick_test()
    else:
        # Vollständige Experimente
        run_trading_experiments()
    
    print("\n✅ Trading Training abgeschlossen!")
