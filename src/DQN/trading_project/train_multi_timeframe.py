"""
Training Script für Multi-Timeframe Emotion-Augmented Trading Agent
Testet verschiedene Märkte mit Multi-Timeframe Support
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
from environments.multi_timeframe_environment import MultiTimeframeEnvironment, create_multi_timeframe_environments
from agents.multi_timeframe_agent import MultiTimeframeTradingAgent, train_multi_timeframe_agent

def run_multi_timeframe_experiments():
    """
    Führe Multi-Timeframe Trading-Experimente durch
    """
    
    print("🚀 Starte Multi-Timeframe Trading-Experimente...")
    
    # Erstelle Environments
    print("\n📊 Erstelle Multi-Timeframe Trading Environments...")
    envs = create_multi_timeframe_environments()
    
    if not envs:
        print("❌ Keine Environments erstellt!")
        return
    
    print(f"✅ {len(envs)} Multi-Timeframe Environments erstellt")
    
    # Experimente
    results = {}
    
    for env_name, env in envs.items():
        print(f"\n🎯 Teste {env_name}...")
        print(f"   Symbol: {env.symbol}")
        print(f"   Timeframes: {env.timeframes}")
        print(f"   Primary Timeframe: {env.primary_timeframe}")
        print(f"   Data Points: {[len(env.data[tf]) for tf in env.timeframes]}")
        
        try:
            # Training
            metrics = train_multi_timeframe_agent(
                env=env,
                episodes=200,  # Reduziert für schnelleren Test
                save_interval=50,
                model_path=f"trading_project/results/{env_name}_multi_tf_model"
            )
            
            results[env_name] = metrics
            
            print(f"✅ {env_name} Training abgeschlossen")
            print(f"   Final Return: {metrics['total_return']*100:.2f}%")
            print(f"   Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"   Final Emotion: {metrics['current_emotion']}")
            print(f"   Dominant Timeframe: {metrics['timeframe_insights'].get('dominant_timeframe', 0)}")
            print(f"   Timeframe Stability: {metrics['timeframe_insights'].get('timeframe_stability', 0):.3f}")
            
        except Exception as e:
            print(f"❌ Fehler bei {env_name}: {e}")
            continue
    
    # Analysiere Ergebnisse
    analyze_multi_timeframe_results(results)
    
    return results

def analyze_multi_timeframe_results(results: Dict):
    """
    Analysiere Multi-Timeframe Trading-Ergebnisse
    """
    
    print("\n📈 ANALYSE DER MULTI-TIMEFRAME TRADING-ERGEBNISSE")
    print("=" * 60)
    
    if not results:
        print("❌ Keine Ergebnisse zu analysieren!")
        return
    
    # Erstelle Vergleichstabelle
    comparison_data = []
    
    for env_name, metrics in results.items():
        timeframe_insights = metrics.get('timeframe_insights', {})
        
        comparison_data.append({
            'Environment': env_name,
            'Total Return (%)': metrics['total_return'] * 100,
            'Win Rate (%)': metrics['win_rate'] * 100,
            'Total Trades': metrics['total_trades'],
            'Total Profit': metrics['total_profit'],
            'Avg Profit/Trade': metrics['avg_profit_per_trade'],
            'Final Emotion': metrics['current_emotion'],
            'Risk Tolerance': metrics['risk_tolerance'],
            'Dominant Timeframe': timeframe_insights.get('dominant_timeframe', 0),
            'Timeframe Stability': timeframe_insights.get('timeframe_stability', 0),
            'Timeframe Diversity': timeframe_insights.get('timeframe_diversity', 0),
            'Final Portfolio': metrics['final_portfolio_value']
        })
    
    # Erstelle DataFrame
    df = pd.DataFrame(comparison_data)
    
    print("\n📊 MULTI-TIMEFRAME PERFORMANCE VERGLEICH:")
    print(df.to_string(index=False, float_format='%.3f'))
    
    # Finde beste Performance
    best_return = df.loc[df['Total Return (%)'].idxmax()]
    best_winrate = df.loc[df['Win Rate (%)'].idxmax()]
    most_stable = df.loc[df['Timeframe Stability'].idxmin()]  # Niedrigere Werte = stabiler
    
    print(f"\n🏆 BESTE PERFORMANCE:")
    print(f"   Höchster Return: {best_return['Environment']} ({best_return['Total Return (%)']:.2f}%)")
    print(f"   Beste Win Rate: {best_winrate['Environment']} ({best_winrate['Win Rate (%)']:.1f}%)")
    print(f"   Stabilste Timeframes: {most_stable['Environment']} (Stability: {most_stable['Timeframe Stability']:.3f})")
    
    # Timeframe-Analyse
    print(f"\n⏰ TIMEFRAME-ANALYSE:")
    timeframe_counts = df['Dominant Timeframe'].value_counts().sort_index()
    for tf_idx, count in timeframe_counts.items():
        print(f"   Timeframe {tf_idx}: {count} Environments als dominant")
    
    # Emotion-Analyse
    print(f"\n🧠 EMOTION-ANALYSE:")
    emotion_counts = df['Final Emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"   {emotion}: {count} Environments")
    
    # Speichere Ergebnisse
    save_multi_timeframe_results(results, df)
    
    # Erstelle Visualisierungen
    create_multi_timeframe_visualizations(results, df)

def save_multi_timeframe_results(results: Dict, df: pd.DataFrame):
    """
    Speichere Multi-Timeframe Ergebnisse
    """
    
    # Erstelle results Ordner falls nicht vorhanden
    os.makedirs("trading_project/results", exist_ok=True)
    
    # Speichere Vergleichstabelle
    df.to_csv("trading_project/results/multi_timeframe_comparison.csv", index=False)
    
    # Speichere detaillierte Ergebnisse
    detailed_results = {}
    for env_name, metrics in results.items():
        detailed_results[env_name] = {
            'metrics': metrics,
            'episode_rewards': metrics.get('episode_rewards', []),
            'episode_returns': metrics.get('episode_returns', []),
            'episode_emotions': metrics.get('episode_emotions', []),
            'episode_attention_weights': metrics.get('episode_attention_weights', []),
            'timeframe_insights': metrics.get('timeframe_insights', {})
        }
    
    # Speichere als JSON
    import json
    with open("trading_project/results/multi_timeframe_detailed_results.json", 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"\n💾 Multi-Timeframe Ergebnisse gespeichert in trading_project/results/")

def create_multi_timeframe_visualizations(results: Dict, df: pd.DataFrame):
    """
    Erstelle Multi-Timeframe Visualisierungen
    """
    
    try:
        # Erstelle Figure mit Subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Multi-Timeframe Emotion-Augmented Trading Agent - Performance Analysis', fontsize=16)
        
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
        
        # 3. Timeframe Stability
        axes[0, 2].bar(df['Environment'], df['Timeframe Stability'])
        axes[0, 2].set_title('Timeframe Stability (Lower = Better)')
        axes[0, 2].set_ylabel('Stability')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Dominant Timeframe Distribution
        timeframe_counts = df['Dominant Timeframe'].value_counts().sort_index()
        axes[1, 0].pie(timeframe_counts.values, labels=[f'TF {i}' for i in timeframe_counts.index], autopct='%1.1f%%')
        axes[1, 0].set_title('Dominant Timeframe Distribution')
        
        # 5. Timeframe Diversity vs Return
        axes[1, 1].scatter(df['Timeframe Diversity'], df['Total Return (%)'], s=100, alpha=0.7)
        for i, env in enumerate(df['Environment']):
            axes[1, 1].annotate(env, (df['Timeframe Diversity'].iloc[i], df['Total Return (%)'].iloc[i]))
        axes[1, 1].set_xlabel('Timeframe Diversity')
        axes[1, 1].set_ylabel('Total Return (%)')
        axes[1, 1].set_title('Diversity vs Return')
        
        # 6. Emotion Distribution
        emotion_counts = df['Final Emotion'].value_counts()
        axes[1, 2].pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%')
        axes[1, 2].set_title('Final Emotion Distribution')
        
        plt.tight_layout()
        plt.savefig('trading_project/results/multi_timeframe_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Multi-Timeframe Visualisierungen gespeichert in trading_project/results/multi_timeframe_analysis.png")
        
    except Exception as e:
        print(f"⚠️ Fehler beim Erstellen der Multi-Timeframe Visualisierungen: {e}")

def compare_single_vs_multi_timeframe():
    """
    Vergleiche Single-Timeframe vs Multi-Timeframe Performance
    """
    
    print("\n🔄 VERGLEICH: Single vs Multi-Timeframe")
    print("=" * 50)
    
    # Lade Single-Timeframe Ergebnisse
    single_tf_file = "trading_project/results/trading_comparison.csv"
    multi_tf_file = "trading_project/results/multi_timeframe_comparison.csv"
    
    if os.path.exists(single_tf_file) and os.path.exists(multi_tf_file):
        single_df = pd.read_csv(single_tf_file)
        multi_df = pd.read_csv(multi_tf_file)
        
        print("📊 PERFORMANCE VERGLEICH:")
        print(f"Single-Timeframe durchschnittlicher Return: {single_df['Total Return (%)'].mean():.2f}%")
        print(f"Multi-Timeframe durchschnittlicher Return: {multi_df['Total Return (%)'].mean():.2f}%")
        print(f"Verbesserung: {((multi_df['Total Return (%)'].mean() / single_df['Total Return (%)'].mean() - 1) * 100):.1f}%")
        
        print(f"\nSingle-Timeframe durchschnittliche Win Rate: {single_df['Win Rate (%)'].mean():.1f}%")
        print(f"Multi-Timeframe durchschnittliche Win Rate: {multi_df['Win Rate (%)'].mean():.1f}%")
        print(f"Verbesserung: {((multi_df['Win Rate (%)'].mean() / single_df['Win Rate (%)'].mean() - 1) * 100):.1f}%")
        
    else:
        print("⚠️ Vergleich nicht möglich - fehlende Ergebnisdateien")

def quick_multi_timeframe_test():
    """
    Schneller Multi-Timeframe Test
    """
    
    print("🚀 Schneller Multi-Timeframe Trading Test...")
    
    # Erstelle ein einfaches Multi-Timeframe Environment
    try:
        env = MultiTimeframeEnvironment(
            symbol='AAPL',
            timeframes=['5m', '15m', '1h'],
            initial_capital=10000.0,
            max_position_size=0.2
        )
        
        print(f"✅ Multi-Timeframe Environment erstellt: {env.symbol}")
        print(f"   Timeframes: {env.timeframes}")
        print(f"   Primary Timeframe: {env.primary_timeframe}")
        print(f"   Data Points: {[len(env.data[tf]) for tf in env.timeframes]}")
        
        # Kurzer Test
        metrics = train_multi_timeframe_agent(
            env=env,
            episodes=50,
            save_interval=25,
            model_path="trading_project/results/quick_multi_tf_test_model"
        )
        
        print(f"\n📈 Quick Multi-Timeframe Test Results:")
        print(f"   Final Return: {metrics['total_return']*100:.2f}%")
        print(f"   Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"   Final Emotion: {metrics['current_emotion']}")
        print(f"   Total Trades: {metrics['total_trades']}")
        print(f"   Dominant Timeframe: {metrics['timeframe_insights'].get('dominant_timeframe', 0)}")
        print(f"   Timeframe Stability: {metrics['timeframe_insights'].get('timeframe_stability', 0):.3f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Fehler beim Quick Multi-Timeframe Test: {e}")
        return None

if __name__ == "__main__":
    print("🎯 Multi-Timeframe Emotion-Augmented Trading Agent Training")
    print("=" * 60)
    
    # Wähle Test-Modus
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick Test
        quick_multi_timeframe_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        # Vergleich Single vs Multi-Timeframe
        compare_single_vs_multi_timeframe()
    else:
        # Vollständige Multi-Timeframe Experimente
        run_multi_timeframe_experiments()
    
    print("\n✅ Multi-Timeframe Trading Training abgeschlossen!")
