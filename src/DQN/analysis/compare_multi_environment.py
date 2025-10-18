"""
Multi-Environment × Multi-Region Comparison - Phase 8.2.1
==========================================================

Vergleicht Performance über:
- 3 Environments (CartPole, Acrobot, LunarLander)
- 3-5 Regions (China, Germany, USA, Brazil, India)

Erstellt publication-ready Comparison Matrix

Author: Phase 8.2.1
Date: 2025-10-17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os

def load_all_regional_data():
    """Load all regional training data across environments"""
    
    data = {
        'CartPole': {},
        'Acrobot': {},
        'LunarLander': {}
    }
    
    # CartPole
    cartpole_files = glob("results/regional/*_training.csv")
    for f in cartpole_files:
        region = os.path.basename(f).replace("_training.csv", "").title()
        try:
            data['CartPole'][region] = pd.read_csv(f)
        except:
            pass
    
    # Acrobot
    acrobot_files = glob("results/regional_acrobot/*_acrobot.csv")
    for f in acrobot_files:
        region = os.path.basename(f).replace("_acrobot.csv", "").title()
        try:
            data['Acrobot'][region] = pd.read_csv(f)
        except:
            pass
    
    # LunarLander
    lunar_files = glob("results/regional_lunarlander/*_lunarlander.csv")
    for f in lunar_files:
        region = os.path.basename(f).replace("_lunarlander.csv", "").title()
        try:
            data['LunarLander'][region] = pd.read_csv(f)
        except:
            pass
    
    return data

def create_performance_matrix(data):
    """Create Environment × Region performance matrix"""
    
    print("\n" + "="*80)
    print("MULTI-ENVIRONMENT × MULTI-REGION PERFORMANCE MATRIX")
    print("="*80 + "\n")
    
    # Collect all regions and environments
    all_regions = set()
    for env_data in data.values():
        all_regions.update(env_data.keys())
    
    all_regions = sorted(list(all_regions))
    environments = ['CartPole', 'Acrobot', 'LunarLander']
    
    # Build matrix
    matrix = pd.DataFrame(index=environments, columns=all_regions)
    
    for env in environments:
        for region in all_regions:
            if region in data[env] and len(data[env][region]) > 0:
                df = data[env][region]
                returns = df['return'].values
                avg100 = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
                matrix.loc[env, region] = avg100
            else:
                matrix.loc[env, region] = np.nan
    
    # Convert to numeric
    matrix = matrix.astype(float)
    
    return matrix

def visualize_performance_matrix(matrix, output_path="results/multi_env_region_heatmap.png"):
    """Create heatmap visualization"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                linewidths=2, linecolor='black',
                cbar_kws={'label': 'Avg100 Return'},
                ax=ax, center=0)
    
    ax.set_title('Performance Matrix: Environment × Region\n(Higher is Better)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Region', fontsize=12, fontweight='bold')
    ax.set_ylabel('Environment', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Performance matrix saved: {output_path}")
    
    return fig

def print_summary_table(matrix):
    """Print formatted summary table"""
    
    print("\nPERFORMANCE TABLE:")
    print("-"*80)
    
    # Header
    header = "Environment".ljust(15)
    for region in matrix.columns:
        header += f"{region:>12}"
    print(header)
    print("-"*80)
    
    # Rows
    for env in matrix.index:
        row = env.ljust(15)
        for region in matrix.columns:
            val = matrix.loc[env, region]
            if pd.isna(val):
                row += "N/A".rjust(12)
            else:
                row += f"{val:>12.1f}"
        print(row)
    
    print("-"*80)
    
    # Regional Averages
    print("\nREGIONAL AVERAGES (across environments):")
    print("-"*80)
    
    for region in matrix.columns:
        avg = matrix[region].mean()
        if not pd.isna(avg):
            print(f"  {region:12s}: {avg:>8.1f}")
    
    print()
    
    # Environment Difficulty
    print("\nENVIRONMENT DIFFICULTY (average across regions):")
    print("-"*80)
    
    for env in matrix.index:
        avg = matrix.loc[env].mean()
        if not pd.isna(avg):
            print(f"  {env:15s}: {avg:>8.1f}")
    
    print()

def analyze_regional_consistency(data):
    """Analyze if regions perform consistently across environments"""
    
    print("\nREGIONAL CONSISTENCY ANALYSIS:")
    print("-"*80)
    
    # For each region, compute std across environments
    matrix = create_performance_matrix(data)
    
    for region in matrix.columns:
        values = matrix[region].dropna().values
        if len(values) >= 2:
            std = np.std(values)
            mean = np.mean(values)
            cv = std / abs(mean) if mean != 0 else 0  # Coefficient of Variation
            
            print(f"{region:12s}: Mean={mean:7.1f} | Std={std:6.1f} | CV={cv:.3f}")
            if cv < 0.3:
                print(f"             → CONSISTENT (low variance across tasks)")
            elif cv > 0.5:
                print(f"             → VARIABLE (task-dependent performance)")
    
    print()

def main():
    """Main analysis pipeline"""
    
    print("="*80)
    print("   MULTI-ENVIRONMENT REGIONAL COMPARISON")
    print("="*80)
    
    # Load data
    data = load_all_regional_data()
    
    # Check what data is available
    print("\n[DATA STATUS]")
    for env, regions in data.items():
        if regions:
            print(f"  {env:15s}: {len(regions)} regions ({', '.join(regions.keys())})")
        else:
            print(f"  {env:15s}: No data yet")
    
    # Create matrix
    matrix = create_performance_matrix(data)
    
    # Print table
    print_summary_table(matrix)
    
    # Visualize
    visualize_performance_matrix(matrix)
    
    # Consistency analysis
    analyze_regional_consistency(data)
    
    print("="*80)
    print("[COMPLETE] Multi-Environment Analysis Done!")
    print("="*80)

if __name__ == "__main__":
    main()





