"""
Final Report Generator - Phase 8.2
===================================

Erstellt automatisch einen umfassenden Report mit:
- Regional Performance Comparison
- Statistical Analysis  
- Visualisierungen
- Recommendations

Nutzung nach Training:
python analysis/create_final_report.py
"""

import pandas as pd
import numpy as np
from glob import glob
import os
import subprocess

def generate_final_report():
    """Generate comprehensive final report"""
    
    print("\n" + "="*70)
    print("   FINAL REPORT GENERATION - PHASE 8.2")
    print("="*70 + "\n")
    
    # 1. Load all data
    print("[STEP 1/4] Loading regional data...")
    csv_files = glob("results/regional/*_training.csv")
    
    if len(csv_files) == 0:
        print("[ERROR] No training data found!")
        return
    
    regional_data = {}
    for csv_file in csv_files:
        region = os.path.basename(csv_file).replace("_training.csv", "").title()
        regional_data[region] = pd.read_csv(csv_file)
        print(f"  Loaded: {region} ({len(regional_data[region])} episodes)")
    
    # 2. Generate visualizations
    print("\n[STEP 2/4] Creating visualizations...")
    try:
        subprocess.run(["python", "analysis/visualize_regional_comparison.py"], 
                      check=True, capture_output=True)
        print("  Created: regional_comparison.png")
        print("  Created: infrastructure_impact_heatmap.png")
    except Exception as e:
        print(f"  [WARNING] Visualization failed: {e}")
    
    # 3. Statistical Analysis
    print("\n[STEP 3/4] Statistical analysis...")
    
    rankings = []
    for region, df in regional_data.items():
        returns = df['return'].values
        final_100 = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
        rankings.append((region, final_100))
    
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    print("\n  REGIONAL RANKING (by final avg100):")
    for rank, (region, perf) in enumerate(rankings, 1):
        medal = " [WINNER]" if rank == 1 else " [RUNNER-UP]" if rank == 2 else ""
        print(f"    {rank}. {region:12s}: {perf:7.1f}{medal}")
    
    # 4. Generate text report
    print("\n[STEP 4/4] Generating text report...")
    
    report_path = "results/regional/FINAL_REPORT.md"
    
    with open(report_path, "w") as f:
        f.write("# Regional Infrastructure Meta-Learning - Final Report\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now()}\n\n")
        f.write("---\n\n")
        
        f.write("## REGIONAL RANKING\n\n")
        for rank, (region, perf) in enumerate(rankings, 1):
            f.write(f"{rank}. **{region}**: {perf:.1f}\n")
        
        f.write("\n---\n\n## DETAILED STATISTICS\n\n")
        
        for region, df in regional_data.items():
            returns = df['return'].values
            emotion = df['emotion'].values
            
            f.write(f"### {region}\n\n")
            f.write(f"- **Episodes:** {len(df)}\n")
            f.write(f"- **Final avg100:** {np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns):.1f}\n")
            f.write(f"- **Best Episode:** {np.max(returns):.1f}\n")
            f.write(f"- **Emotion Mean:** {np.mean(emotion):.3f}\n")
            f.write(f"- **Emotion Std:** {np.std(emotion):.3f}\n")
            
            # Infrastructure params
            f.write(f"- **Loop Speed:** {df['infrastructure_loop_speed'].iloc[0]:.2f}\n")
            f.write(f"- **Automation:** {df['infrastructure_automation'].iloc[0]:.2f}\n")
            f.write(f"- **Error Tolerance:** {df['infrastructure_error_tolerance'].iloc[0]:.2f}\n")
            f.write("\n")
        
        f.write("---\n\n## KEY INSIGHTS\n\n")
        
        winner, winner_perf = rankings[0]
        loser, loser_perf = rankings[-1]
        perf_gap = winner_perf - loser_perf
        
        f.write(f"- **Best Region:** {winner} ({winner_perf:.1f})\n")
        f.write(f"- **Worst Region:** {loser} ({loser_perf:.1f})\n")
        f.write(f"- **Performance Gap:** {perf_gap:.1f} ({perf_gap/abs(loser_perf)*100:.1f}%)\n")
        
        f.write("\n---\n\n## VISUALIZATIONS\n\n")
        f.write("See:\n")
        f.write("- `regional_comparison.png` - 9-panel comprehensive view\n")
        f.write("- `infrastructure_impact_heatmap.png` - Parameter impact analysis\n")
    
    print(f"  Created: {report_path}")
    
    print("\n" + "="*70)
    print("REPORT GENERATION COMPLETE!")
    print("="*70)
    print(f"\nCheck: {report_path}")
    print("       results/regional_comparison.png")
    print("       results/infrastructure_impact_heatmap.png")

if __name__ == "__main__":
    generate_final_report()





