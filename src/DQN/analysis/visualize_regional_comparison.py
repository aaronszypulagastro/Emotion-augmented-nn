"""
Regional Infrastructure Comparison Visualization - Phase 8.2
============================================================

Erstellt publikationswürdige Visualisierungen für:
1. Performance Comparison (Region × Metric)
2. Learning Curves (mit Confidence Intervals)
3. Infrastructure Impact Heatmap
4. Emotion Dynamics per Region
5. Robustness Analysis (Variance across Regions)
6. Regional "Sweet Spot" Analysis

Author: Phase 8.2 - Regional Infrastructure Meta-Learning
Date: 2025-10-17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
import sys

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_regional_data(data_dir="results/regional"):
    """
    Lade alle Regional Training Logs
    
    Returns:
        Dict mapping region_name → DataFrame
    """
    regional_data = {}
    
    # Find all CSV files in regional directory
    csv_files = glob(os.path.join(data_dir, "*_training.csv"))
    
    if len(csv_files) == 0:
        print(f"[WARNING] No training logs found in {data_dir}")
        return regional_data
    
    for csv_file in csv_files:
        # Extract region name from filename
        basename = os.path.basename(csv_file)
        region_name = basename.replace("_training.csv", "").title()
        
        try:
            df = pd.read_csv(csv_file)
            regional_data[region_name] = df
            print(f"[LOADED] {region_name}: {len(df)} episodes")
        except Exception as e:
            print(f"[ERROR] Could not load {csv_file}: {e}")
    
    return regional_data

def create_comprehensive_visualization(regional_data, output_path="results/regional_comparison.png"):
    """
    Erstelle umfassende Multi-Region Visualisierung
    
    Layout: 3×3 Grid mit verschiedenen Analysen
    """
    
    if len(regional_data) == 0:
        print("[ERROR] No data to visualize!")
        return
    
    fig = plt.figure(figsize=(18, 14))
    
    regions = list(regional_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(regions)))
    region_colors = {region: colors[i] for i, region in enumerate(regions)}
    
    # ===== 1. PERFORMANCE OVER TIME =====
    ax1 = plt.subplot(3, 3, 1)
    
    for region in regions:
        df = regional_data[region]
        episodes = df['episode'].values
        returns = df['return'].values
        
        # Smoothed
        if len(returns) >= 20:
            smoothed = pd.Series(returns).rolling(window=20, center=True).mean()
            ax1.plot(episodes, smoothed, 
                    color=region_colors[region], 
                    linewidth=2.5, 
                    label=region,
                    alpha=0.9)
    
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Return (MA-20)', fontsize=11)
    ax1.set_title('1. Performance over Time', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ===== 2. FINAL PERFORMANCE COMPARISON =====
    ax2 = plt.subplot(3, 3, 2)
    
    final_avgs = []
    region_names = []
    
    for region in regions:
        df = regional_data[region]
        returns = df['return'].values
        final_100 = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
        final_avgs.append(final_100)
        region_names.append(region)
    
    bars = ax2.bar(range(len(regions)), final_avgs, 
                   color=[region_colors[r] for r in regions],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax2.set_xticks(range(len(regions)))
    ax2.set_xticklabels(region_names, rotation=0)
    ax2.set_ylabel('Final avg100 Return', fontsize=11)
    ax2.set_title('2. Final Performance Comparison', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, final_avgs)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_avgs)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ===== 3. EMOTION DYNAMICS =====
    ax3 = plt.subplot(3, 3, 3)
    
    for region in regions:
        df = regional_data[region]
        episodes = df['episode'].values
        emotion = df['emotion'].values
        
        ax3.plot(episodes, emotion,
                color=region_colors[region],
                linewidth=2,
                label=region,
                alpha=0.8)
    
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('Emotion', fontsize=11)
    ax3.set_title('3. Emotion Dynamics per Region', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ===== 4. INFRASTRUCTURE PARAMETER IMPACT =====
    ax4 = plt.subplot(3, 3, 4)
    
    # Collect infrastructure params and performance
    infra_data = []
    for region in regions:
        df = regional_data[region]
        final_perf = np.mean(df['return'].values[-100:]) if len(df) >= 100 else np.mean(df['return'].values)
        
        infra_data.append({
            'region': region,
            'loop_speed': df['infrastructure_loop_speed'].iloc[0],
            'automation': df['infrastructure_automation'].iloc[0],
            'error_tolerance': df['infrastructure_error_tolerance'].iloc[0],
            'performance': final_perf
        })
    
    infra_df = pd.DataFrame(infra_data)
    
    # Scatter: Loop Speed vs Performance
    ax4.scatter(infra_df['loop_speed'], infra_df['performance'],
               c=[region_colors[r] for r in infra_df['region']],
               s=200, alpha=0.7, edgecolor='black', linewidth=2)
    
    for i, row in infra_df.iterrows():
        ax4.annotate(row['region'], 
                    (row['loop_speed'], row['performance']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    ax4.set_xlabel('Loop Speed (higher = slower feedback)', fontsize=11)
    ax4.set_ylabel('Final Performance', fontsize=11)
    ax4.set_title('4. Loop Speed vs Performance', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # ===== 5. AUTOMATION IMPACT =====
    ax5 = plt.subplot(3, 3, 5)
    
    ax5.scatter(infra_df['automation'], infra_df['performance'],
               c=[region_colors[r] for r in infra_df['region']],
               s=200, alpha=0.7, edgecolor='black', linewidth=2)
    
    for i, row in infra_df.iterrows():
        ax5.annotate(row['region'],
                    (row['automation'], row['performance']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    ax5.set_xlabel('Automation Level', fontsize=11)
    ax5.set_ylabel('Final Performance', fontsize=11)
    ax5.set_title('5. Automation vs Performance', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # ===== 6. LEARNING EFFICIENCY =====
    ax6 = plt.subplot(3, 3, 6)
    
    learning_speeds = []
    
    for region in regions:
        df = regional_data[region]
        returns = df['return'].values
        
        # Measure: Episodes to reach 50% of final performance
        final_perf = np.mean(returns[-50:])
        threshold = final_perf * 0.5
        
        # Find first episode above threshold
        above_threshold = np.where(returns > threshold)[0]
        if len(above_threshold) > 0:
            episodes_to_threshold = above_threshold[0]
        else:
            episodes_to_threshold = len(returns)
        
        learning_speeds.append(episodes_to_threshold)
    
    bars = ax6.bar(range(len(regions)), learning_speeds,
                   color=[region_colors[r] for r in regions],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax6.set_xticks(range(len(regions)))
    ax6.set_xticklabels(regions, rotation=0)
    ax6.set_ylabel('Episodes to 50% Final Perf', fontsize=11)
    ax6.set_title('6. Learning Speed Comparison', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.invert_yaxis()  # Lower is better
    
    # Add labels
    for bar, val in zip(bars, learning_speeds):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() - max(learning_speeds)*0.05,
                f'{val}', ha='center', va='top', fontsize=10, fontweight='bold')
    
    # ===== 7. COMPETITION WIN RATES =====
    ax7 = plt.subplot(3, 3, 7)
    
    win_rates = []
    
    for region in regions:
        df = regional_data[region]
        # Get final win rate
        final_wr = df['win_rate'].iloc[-1] if 'win_rate' in df.columns else 0.0
        win_rates.append(final_wr)
    
    bars = ax7.bar(range(len(regions)), win_rates,
                   color=[region_colors[r] for r in regions],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax7.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='50% Baseline')
    ax7.set_xticks(range(len(regions)))
    ax7.set_xticklabels(regions, rotation=0)
    ax7.set_ylabel('Final Win Rate', fontsize=11)
    ax7.set_title('7. Competition Win Rates', fontsize=12, fontweight='bold')
    ax7.set_ylim([0, 1])
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # ===== 8. EMOTION STABILITY =====
    ax8 = plt.subplot(3, 3, 8)
    
    emotion_stds = []
    
    for region in regions:
        df = regional_data[region]
        emotion = df['emotion'].values
        emotion_std = np.std(emotion)
        emotion_stds.append(emotion_std)
    
    bars = ax8.bar(range(len(regions)), emotion_stds,
                   color=[region_colors[r] for r in regions],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax8.set_xticks(range(len(regions)))
    ax8.set_xticklabels(regions, rotation=0)
    ax8.set_ylabel('Emotion Std (higher = more dynamic)', fontsize=11)
    ax8.set_title('8. Emotion Stability per Region', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # ===== 9. SUMMARY TABLE =====
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Create summary table
    table_data = []
    
    for region in regions:
        df = regional_data[region]
        returns = df['return'].values
        
        final_100 = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
        best = np.max(returns)
        emo_mean = df['emotion'].mean()
        emo_std = df['emotion'].std()
        
        table_data.append([
            region,
            f"{final_100:.1f}",
            f"{best:.1f}",
            f"{emo_mean:.3f}",
            f"{emo_std:.3f}"
        ])
    
    table = ax9.table(
        cellText=table_data,
        colLabels=['Region', 'Avg100', 'Best', 'Emo μ', 'Emo σ'],
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.15, 0.15, 0.15, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color best performance
    best_idx = np.argmax([float(row[1]) for row in table_data])
    for i in range(5):
        table[(best_idx + 1, i)].set_facecolor('#90EE90')
    
    ax9.set_title('9. Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # ===== LAYOUT =====
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Visualization saved: {output_path}")
    
    return fig

def create_infrastructure_heatmap(regional_data, output_path="results/infrastructure_impact_heatmap.png"):
    """
    Erstelle Heatmap: Infrastructure Parameters × Performance
    """
    
    if len(regional_data) == 0:
        print("[ERROR] No data for heatmap!")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Collect data
    regions = list(regional_data.keys())
    
    infra_params = {
        'Loop Speed': [],
        'Automation': [],
        'Error Tolerance': []
    }
    
    performances = []
    
    for region in regions:
        df = regional_data[region]
        
        infra_params['Loop Speed'].append(df['infrastructure_loop_speed'].iloc[0])
        infra_params['Automation'].append(df['infrastructure_automation'].iloc[0])
        infra_params['Error Tolerance'].append(df['infrastructure_error_tolerance'].iloc[0])
        
        final_perf = np.mean(df['return'].values[-100:]) if len(df) >= 100 else np.mean(df['return'].values)
        performances.append(final_perf)
    
    # Create DataFrame for heatmap
    heatmap_data = pd.DataFrame(infra_params, index=regions)
    heatmap_data['Performance'] = performances
    
    # Heatmap 1: Infrastructure Parameters
    sns.heatmap(heatmap_data[['Loop Speed', 'Automation', 'Error Tolerance']].T,
                annot=True, fmt='.2f', cmap='YlOrRd', 
                linewidths=1, linecolor='black',
                cbar_kws={'label': 'Value'},
                ax=ax1)
    ax1.set_title('Infrastructure Parameters by Region', fontsize=12, fontweight='bold')
    ax1.set_xlabel('')
    
    # Heatmap 2: Performance
    perf_data = pd.DataFrame({'Performance': performances}, index=regions).T
    sns.heatmap(perf_data, annot=True, fmt='.1f', cmap='RdYlGn',
                linewidths=1, linecolor='black',
                cbar_kws={'label': 'Return'},
                ax=ax2)
    ax2.set_title('Final Performance by Region', fontsize=12, fontweight='bold')
    ax2.set_xlabel('')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Heatmap saved: {output_path}")
    
    return fig

def print_statistical_analysis(regional_data):
    """
    Drucke statistische Analyse der Regional-Unterschiede
    """
    
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS - REGIONAL COMPARISON")
    print("="*70 + "\n")
    
    regions = list(regional_data.keys())
    
    # Performance Stats
    print("PERFORMANCE STATISTICS:")
    print("-" * 70)
    print(f"{'Region':<12} {'Mean':>10} {'Std':>10} {'Best':>10} {'Worst':>10}")
    print("-" * 70)
    
    for region in regions:
        df = regional_data[region]
        returns = df['return'].values
        
        print(f"{region:<12} {np.mean(returns):>10.2f} {np.std(returns):>10.2f} "
              f"{np.max(returns):>10.2f} {np.min(returns):>10.2f}")
    
    print()
    
    # Emotion Stats
    print("EMOTION STATISTICS:")
    print("-" * 70)
    print(f"{'Region':<12} {'Mean':>10} {'Std':>10} {'Range':>15}")
    print("-" * 70)
    
    for region in regions:
        df = regional_data[region]
        emotion = df['emotion'].values
        
        print(f"{region:<12} {np.mean(emotion):>10.3f} {np.std(emotion):>10.3f} "
              f"[{np.min(emotion):.3f}, {np.max(emotion):.3f}]")
    
    print()
    
    # Learning Efficiency
    print("LEARNING EFFICIENCY:")
    print("-" * 70)
    print(f"{'Region':<12} {'First 50':>10} {'Last 50':>10} {'Improvement':>12}")
    print("-" * 70)
    
    for region in regions:
        df = regional_data[region]
        returns = df['return'].values
        
        first_50 = np.mean(returns[:50]) if len(returns) >= 50 else np.mean(returns)
        last_50 = np.mean(returns[-50:]) if len(returns) >= 50 else np.mean(returns)
        improvement = last_50 - first_50
        
        print(f"{region:<12} {first_50:>10.1f} {last_50:>10.1f} {improvement:>+12.1f}")
    
    print()
    
    # RANKING
    print("REGIONAL RANKING:")
    print("-" * 70)
    
    rankings = []
    for region in regions:
        df = regional_data[region]
        final_perf = np.mean(df['return'].values[-100:]) if len(df) >= 100 else np.mean(df['return'].values)
        rankings.append((region, final_perf))
    
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (region, perf) in enumerate(rankings, 1):
        medal = "GOLD" if rank == 1 else "SILVER" if rank == 2 else "BRONZE" if rank == 3 else ""
        print(f"   {rank}. {region:<12} {perf:>10.1f}  {medal}")
    
    print("\n" + "="*70 + "\n")

def main():
    """Main visualization pipeline"""
    
    print("="*70)
    print("   REGIONAL INFRASTRUCTURE VISUALIZATION")
    print("="*70 + "\n")
    
    # Load Data
    regional_data = load_regional_data("results/regional")
    
    if len(regional_data) == 0:
        print("[ERROR] No regional data found!")
        print("Run train_regional_infrastructure.py first!")
        return
    
    print(f"\n[INFO] Loaded {len(regional_data)} regions")
    print()
    
    # Create Visualizations
    print("[CREATING] Comprehensive visualization...")
    create_comprehensive_visualization(regional_data)
    
    print("[CREATING] Infrastructure heatmap...")
    create_infrastructure_heatmap(regional_data)
    
    # Statistical Analysis
    print_statistical_analysis(regional_data)
    
    print("[DONE] All visualizations created!")
    print("Check: results/regional_comparison.png")
    print("       results/infrastructure_impact_heatmap.png")

if __name__ == "__main__":
    main()





