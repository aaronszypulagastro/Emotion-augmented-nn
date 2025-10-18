"""
Visualisierung der Competitive Self-Play Dynamics
==================================================

Erstellt:
1. Performance over Time (mit Competitions markiert)
2. Emotion Dynamics
3. Win/Loss/Draw Distribution
4. Competition Outcome Timeline
5. LR Modulation vs Emotion
6. Mindset State Transitions
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def visualize_competitive_training(log_path="results/competitive_selfplay_log.csv"):
    """Create comprehensive visualization of competitive training"""
    
    if not os.path.exists(log_path):
        print(f"Error: Log file not found: {log_path}")
        return
    
    # Read Data
    try:
        df = pd.read_csv(log_path)
    except:
        print(f"Error: Could not read log file: {log_path}")
        return
    
    if len(df) == 0:
        print("Error: Log file is empty")
        return
    
    print(f"Visualizing {len(df)} episodes...\n")
    
    # Prepare Figure
    fig = plt.figure(figsize=(16, 12))
    
    # ===== 1. PERFORMANCE OVER TIME =====
    ax1 = plt.subplot(3, 3, 1)
    
    episodes = df['episode'].values
    returns = df['return'].values
    
    # Plot returns
    ax1.plot(episodes, returns, alpha=0.3, color='gray', label='Raw Returns')
    
    # Smoothed
    if len(returns) >= 20:
        window = min(20, len(returns))
        smoothed = pd.Series(returns).rolling(window=window, center=True).mean()
        ax1.plot(episodes, smoothed, color='blue', linewidth=2, label=f'MA-{window}')
    
    # Mark Competitions
    competitions = df[df['had_competition'] == True]
    if len(competitions) > 0:
        ax1.scatter(
            competitions['episode'],
            competitions['return'],
            color='red',
            s=50,
            alpha=0.6,
            marker='*',
            label='Competitions',
            zorder=5
        )
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return')
    ax1.set_title('1. Performance over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ===== 2. EMOTION DYNAMICS =====
    ax2 = plt.subplot(3, 3, 2)
    
    emotion = df['emotion'].values
    ax2.plot(episodes, emotion, color='orange', linewidth=2)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Neutral')
    ax2.axhline(y=0.2, color='red', linestyle=':', alpha=0.5, label='Lower Bound')
    ax2.axhline(y=0.8, color='green', linestyle=':', alpha=0.5, label='Upper Bound')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Emotion')
    ax2.set_title('2. Emotion Dynamics')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ===== 3. WIN/LOSS/DRAW OVER TIME =====
    ax3 = plt.subplot(3, 3, 3)
    
    if len(competitions) > 0:
        comp_eps = competitions['episode'].values
        win_rate = competitions['win_rate'].values
        loss_rate = competitions['loss_rate'].values
        draw_rate = competitions['draw_rate'].values
        
        ax3.plot(comp_eps, win_rate, color='green', marker='o', label='Win Rate')
        ax3.plot(comp_eps, loss_rate, color='red', marker='s', label='Loss Rate')
        ax3.plot(comp_eps, draw_rate, color='gray', marker='^', label='Draw Rate')
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Rate')
        ax3.set_title('3. Win/Loss/Draw Rates')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Competitions Yet', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('3. Win/Loss/Draw Rates')
    
    # ===== 4. COMPETITION OUTCOMES =====
    ax4 = plt.subplot(3, 3, 4)
    
    if len(competitions) > 0:
        # Count outcomes
        outcomes = competitions['competition_outcome'].value_counts()
        
        colors_map = {
            'decisive_win': 'darkgreen',
            'win': 'lightgreen',
            'draw': 'gray',
            'loss': 'lightcoral',
            'decisive_loss': 'darkred'
        }
        
        colors = [colors_map.get(o, 'blue') for o in outcomes.index]
        
        ax4.bar(range(len(outcomes)), outcomes.values, color=colors)
        ax4.set_xticks(range(len(outcomes)))
        ax4.set_xticklabels(outcomes.index, rotation=45, ha='right')
        ax4.set_ylabel('Count')
        ax4.set_title('4. Competition Outcome Distribution')
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No Competitions Yet',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('4. Competition Outcome Distribution')
    
    # ===== 5. LR MODULATION VS EMOTION =====
    ax5 = plt.subplot(3, 3, 5)
    
    lr = df['lr_actual'].values
    base_lr = 5e-4
    lr_factor = lr / base_lr
    
    ax5.scatter(emotion, lr_factor, alpha=0.5, c=episodes, cmap='viridis', s=20)
    ax5.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Base LR')
    ax5.set_xlabel('Emotion')
    ax5.set_ylabel('LR Factor (vs Base)')
    ax5.set_title('5. LR Modulation by Emotion')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax5.collections[0], ax=ax5)
    cbar.set_label('Episode')
    
    # ===== 6. EPSILON DECAY =====
    ax6 = plt.subplot(3, 3, 6)
    
    epsilon = df['epsilon'].values
    ax6.plot(episodes, epsilon, color='purple', linewidth=2)
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Epsilon')
    ax6.set_title('6. Exploration (Epsilon) Decay')
    ax6.grid(True, alpha=0.3)
    
    # ===== 7. WIN/LOSS MOMENTUM =====
    ax7 = plt.subplot(3, 3, 7)
    
    if len(competitions) > 0:
        momentum = competitions['win_loss_momentum'].values
        comp_eps_mom = competitions['episode'].values
        
        # Color by sign
        colors_mom = ['green' if m > 0 else 'red' for m in momentum]
        ax7.scatter(comp_eps_mom, momentum, c=colors_mom, s=50, alpha=0.7)
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax7.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Win Streak')
        ax7.axhline(y=-0.3, color='red', linestyle='--', alpha=0.5, label='Loss Streak')
        
        ax7.set_xlabel('Episode')
        ax7.set_ylabel('Momentum')
        ax7.set_title('7. Win/Loss Momentum')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'No Competitions Yet',
                ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('7. Win/Loss Momentum')
    
    # ===== 8. MINDSET STATE OVER TIME =====
    ax8 = plt.subplot(3, 3, 8)
    
    # Encode mindsets as numbers
    mindset_map = {
        'DOMINANT': 5,
        'CONFIDENT': 4,
        'BALANCED': 3,
        'ADAPTIVE': 2,
        'DETERMINED': 1,
        'FRUSTRATED': 0
    }
    
    mindset_values = [mindset_map.get(m.strip(), 2) for m in df['competitive_mindset']]
    
    ax8.plot(episodes, mindset_values, marker='o', markersize=3, linewidth=1, alpha=0.7)
    ax8.set_xlabel('Episode')
    ax8.set_ylabel('Mindset State')
    ax8.set_yticks(range(6))
    ax8.set_yticklabels(['FRUSTRATED', 'DETERMINED', 'ADAPTIVE', 'BALANCED', 'CONFIDENT', 'DOMINANT'])
    ax8.set_title('8. Mindset State Evolution')
    ax8.grid(True, alpha=0.3)
    
    # ===== 9. EMOTION vs PERFORMANCE CORRELATION =====
    ax9 = plt.subplot(3, 3, 9)
    
    # Moving average returns
    if len(returns) >= 10:
        ma_returns = pd.Series(returns).rolling(window=10, center=True).mean().values
        
        ax9.scatter(emotion, ma_returns, alpha=0.5, c=episodes, cmap='coolwarm', s=20)
        ax9.set_xlabel('Emotion')
        ax9.set_ylabel('Performance (MA-10)')
        ax9.set_title('9. Emotion vs Performance')
        ax9.grid(True, alpha=0.3)
        
        # Correlation
        valid_idx = ~np.isnan(ma_returns)
        if valid_idx.sum() > 10:
            corr = np.corrcoef(emotion[valid_idx], ma_returns[valid_idx])[0, 1]
            ax9.text(0.05, 0.95, f'Corr: {corr:.3f}',
                    transform=ax9.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ===== LAYOUT =====
    plt.tight_layout()
    
    # Save
    output_path = "results/competitive_selfplay_analysis.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"[OK] Visualization saved to: {output_path}")
    
    # Show
    # plt.show()  # Uncomment if you want to display
    
    # ===== PRINT SUMMARY STATS =====
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nPerformance:")
    print(f"  Final avg (last 20): {returns[-20:].mean():.1f}")
    if len(returns) >= 100:
        print(f"  Final avg (last 100): {returns[-100:].mean():.1f}")
    print(f"  Best Episode: {returns.max():.1f}")
    print(f"  Worst Episode: {returns.min():.1f}")
    
    print(f"\nEmotion:")
    print(f"  Mean: {emotion.mean():.3f}")
    print(f"  Std: {emotion.std():.3f}")
    print(f"  Range: [{emotion.min():.3f}, {emotion.max():.3f}]")
    
    if len(competitions) > 0:
        print(f"\nCompetitions:")
        print(f"  Total: {len(competitions)}")
        print(f"  Final Win Rate: {df['win_rate'].iloc[-1]:.1%}")
        print(f"  Final Loss Rate: {df['loss_rate'].iloc[-1]:.1%}")
        print(f"  Final Draw Rate: {df['draw_rate'].iloc[-1]:.1%}")
        print(f"  Final Momentum: {df['win_loss_momentum'].iloc[-1]:+.3f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = "results/competitive_selfplay_log.csv"
    
    visualize_competitive_training(log_path)





