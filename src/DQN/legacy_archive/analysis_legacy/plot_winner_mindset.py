"""
Winner Mindset Visualization
=============================

Visualisiert die Mindset-Dynamics über Training-Episoden

Plots:
------
1. Emotion vs Performance Trends
2. Mindset State Timeline
3. Exploration & Noise Modulation
4. Learning Efficiency Index
5. Performance Stability Heatmap

Author: Phase 8.0
Date: 2025-10-16
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import seaborn as sns

# Matplotlib Backend (headless)
plt.switch_backend('Agg')


def plot_winner_mindset_dashboard(
    mindset_dynamics: Dict,
    save_path: str = "results/winner_mindset_dashboard.png"
):
    """
    Erstelle Dashboard mit allen Mindset-Metriken
    
    Args:
        mindset_dynamics: Output von WinnerMindsetRegulator.log_mindset_dynamics()
        save_path: Speicherpfad
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Extract data
    emotions = mindset_dynamics.get('emotions', [])
    states = mindset_dynamics.get('states', [])
    exploration = mindset_dynamics.get('exploration', [])
    noise = mindset_dynamics.get('noise', [])
    focus = mindset_dynamics.get('focus', [])
    efficiency = mindset_dynamics.get('efficiency', [])
    performance = mindset_dynamics.get('performance', [])
    
    episodes = np.arange(len(emotions))
    
    # 1. Emotion vs Performance
    ax1 = plt.subplot(3, 2, 1)
    ax1_twin = ax1.twinx()
    
    ax1.plot(episodes, emotions, 'b-', label='Emotion', linewidth=2, alpha=0.7)
    ax1_twin.plot(episodes, performance, 'g-', label='Performance', linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Emotion', color='b')
    ax1_twin.set_ylabel('Return', color='g')
    ax1.set_title('Emotion vs Performance Correlation')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # 2. Mindset State Timeline
    ax2 = plt.subplot(3, 2, 2)
    
    # Map states to numbers
    state_map = {
        'frustration': 0,
        'calm': 1,
        'pride': 2,
        'curiosity': 3,
        'focus': 4
    }
    state_values = [state_map.get(s, 1) for s in states]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    ax2.scatter(episodes, state_values, c=[colors[v] for v in state_values], 
                alpha=0.6, s=20)
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(['Frustration', 'Calm', 'Pride', 'Curiosity', 'Focus'])
    ax2.set_xlabel('Episode')
    ax2.set_title('Mindset State Evolution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Exploration Factor
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(episodes, exploration, 'orange', linewidth=2, alpha=0.7)
    ax3.fill_between(episodes, 0, exploration, alpha=0.3, color='orange')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Exploration Factor')
    ax3.set_title('Adaptive Exploration (Epsilon Modulation)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. Noise Scale
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(episodes, noise, 'purple', linewidth=2, alpha=0.7)
    ax4.fill_between(episodes, 0, noise, alpha=0.3, color='purple')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Noise Scale')
    ax4.set_title('Adaptive Noise (BDH-Plasticity Modulation)')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # 5. Focus Intensity
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(episodes, focus, 'darkred', linewidth=2, alpha=0.7)
    ax5.fill_between(episodes, 0, focus, alpha=0.3, color='darkred')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Focus Intensity')
    ax5.set_title('Focus Level (Winner Mindset)')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])
    
    # 6. Learning Efficiency Index
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(episodes, efficiency, 'cyan', linewidth=2, alpha=0.7)
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax6.fill_between(episodes, 0, efficiency, where=np.array(efficiency) > 0,
                     alpha=0.3, color='green', label='Positive')
    ax6.fill_between(episodes, 0, efficiency, where=np.array(efficiency) < 0,
                     alpha=0.3, color='red', label='Negative')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Learning Efficiency')
    ax6.set_title('Learning Efficiency Index (Reward Growth / Episode)')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Winner Mindset Dashboard gespeichert: {save_path}")


def plot_mindset_heatmap(
    mindset_dynamics: Dict,
    save_path: str = "results/mindset_heatmap.png"
):
    """
    Performance Stability Heatmap mit Mindset-States
    
    Zeigt Korrelation zwischen Mindset und Performance-Stabilität
    """
    emotions = mindset_dynamics.get('emotions', [])
    performance = mindset_dynamics.get('performance', [])
    states = mindset_dynamics.get('states', [])
    
    if len(emotions) < 20:
        print("⚠️  Nicht genug Daten für Heatmap")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Emotion-Performance 2D Histogram
    h, xedges, yedges = np.histogram2d(
        emotions, performance,
        bins=20,
        range=[[0, 1], [0, max(performance) if performance else 500]]
    )
    
    im1 = ax1.imshow(h.T, origin='lower', aspect='auto', cmap='viridis',
                     extent=[0, 1, 0, max(performance) if performance else 500])
    ax1.set_xlabel('Emotion')
    ax1.set_ylabel('Performance (Return)')
    ax1.set_title('Emotion-Performance Density')
    plt.colorbar(im1, ax=ax1, label='Count')
    
    # 2. State Distribution Pie Chart
    state_counts = {}
    for state in states:
        state_counts[state] = state_counts.get(state, 0) + 1
    
    colors_map = {
        'frustration': 'red',
        'calm': 'blue',
        'pride': 'green',
        'curiosity': 'orange',
        'focus': 'purple'
    }
    
    colors = [colors_map.get(s, 'gray') for s in state_counts.keys()]
    ax2.pie(state_counts.values(), labels=state_counts.keys(),
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Mindset State Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Mindset Heatmap gespeichert: {save_path}")


def print_mindset_summary(mindset_dynamics: Dict):
    """Drucke Zusammenfassung der Mindset-Dynamics"""
    
    states = mindset_dynamics.get('states', [])
    emotions = mindset_dynamics.get('emotions', [])
    efficiency = mindset_dynamics.get('efficiency', [])
    exploration = mindset_dynamics.get('exploration', [])
    
    if not states:
        print("⚠️  Keine Mindset-Daten vorhanden")
        return
    
    print("\n" + "="*60)
    print("WINNER MINDSET - ZUSAMMENFASSUNG")
    print("="*60)
    
    # State Distribution
    print("\n📊 Mindset State Distribution:")
    state_counts = {}
    for state in states:
        state_counts[state] = state_counts.get(state, 0) + 1
    
    total = len(states)
    for state, count in sorted(state_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        print(f"   {state.capitalize():12s}: {count:4d} ({pct:5.1f}%)")
    
    # Emotion Statistics
    print(f"\n💭 Emotion Statistics:")
    print(f"   Mean: {np.mean(emotions):.3f}")
    print(f"   Std:  {np.std(emotions):.3f}")
    print(f"   Min:  {np.min(emotions):.3f}")
    print(f"   Max:  {np.max(emotions):.3f}")
    
    # Efficiency
    if efficiency and len(efficiency) > 10:
        print(f"\n📈 Learning Efficiency:")
        print(f"   Mean: {np.mean(efficiency):.3f}")
        print(f"   Final: {efficiency[-1]:.3f}")
        positive_pct = 100 * sum(1 for e in efficiency if e > 0) / len(efficiency)
        print(f"   Positive: {positive_pct:.1f}%")
    
    # Exploration
    print(f"\n🔍 Exploration Dynamics:")
    print(f"   Mean: {np.mean(exploration):.3f}")
    print(f"   Range: [{np.min(exploration):.3f}, {np.max(exploration):.3f}]")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Test Visualization
    print("Testing Winner Mindset Visualization...\n")
    
    # Create dummy data
    n = 200
    mindset_data = {
        'emotions': list(0.3 + 0.4 * np.sin(np.linspace(0, 4*np.pi, n)) + 0.1 * np.random.randn(n)),
        'states': ['frustration' if i < 50 else 'calm' if i < 100 else 'pride' if i < 150 else 'curiosity' 
                   for i in range(n)],
        'exploration': list(0.5 + 0.3 * np.cos(np.linspace(0, 3*np.pi, n))),
        'noise': list(0.4 + 0.2 * np.sin(np.linspace(0, 2*np.pi, n))),
        'focus': list(0.5 + 0.3 * (np.arange(n) / n)),
        'efficiency': list(np.tanh((np.arange(n) - 100) / 50)),
        'performance': list(100 + 200 * (np.arange(n) / n) + 50 * np.random.randn(n))
    }
    
    plot_winner_mindset_dashboard(mindset_data, "results/test_mindset_dashboard.png")
    plot_mindset_heatmap(mindset_data, "results/test_mindset_heatmap.png")
    print_mindset_summary(mindset_data)
    
    print("✅ Test-Visualisierungen erstellt!")


