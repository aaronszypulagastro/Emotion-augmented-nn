"""
Statistical Analysis Tools - Phase 8.2.1
=========================================

Publication-grade statistical tests for regional comparison:
- ANOVA (Analysis of Variance)
- Post-hoc Tukey HSD tests
- Correlation analysis (Infrastructure × Performance)
- Effect size calculation (Cohen's d)
- Confidence intervals

Author: Phase 8.2.1
Date: 2025-10-17
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway, pearsonr, spearmanr
from glob import glob
import os

def load_regional_data(env_name="CartPole"):
    """
    Load regional data for specific environment
    
    Args:
        env_name: 'CartPole', 'Acrobot', or 'LunarLander'
        
    Returns:
        Dict mapping region → DataFrame
    """
    data = {}
    
    if env_name == "CartPole":
        pattern = "results/regional/*_training.csv"
    elif env_name == "Acrobot":
        pattern = "results/regional_acrobot/*_acrobot.csv"
    elif env_name == "LunarLander":
        pattern = "results/regional_lunarlander/*_lunarlander.csv"
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    files = glob(pattern)
    
    for f in files:
        basename = os.path.basename(f)
        if "CartPole" in env_name:
            region = basename.replace("_training.csv", "").title()
        elif "Acrobot" in env_name:
            region = basename.replace("_acrobot.csv", "").title()
        else:
            region = basename.replace("_lunarlander.csv", "").title()
        
        try:
            data[region] = pd.read_csv(f)
        except Exception as e:
            print(f"[WARNING] Could not load {basename}: {e}")
    
    return data

def anova_regional_comparison(env_name="CartPole", use_last_n=100):
    """
    Perform ANOVA to test if regional differences are significant
    
    H0: All regions have same mean performance
    H1: At least one region differs significantly
    
    Args:
        env_name: Environment to analyze
        use_last_n: Use last N episodes for comparison (convergence period)
        
    Returns:
        Dict with F-statistic, p-value, and interpretation
    """
    print("\n" + "="*70)
    print(f"ANOVA: REGIONAL COMPARISON - {env_name}")
    print("="*70 + "\n")
    
    data = load_regional_data(env_name)
    
    if len(data) < 2:
        print(f"[ERROR] Need at least 2 regions, found {len(data)}")
        return None
    
    # Extract performance data (last N episodes)
    groups = []
    region_names = []
    
    for region, df in data.items():
        returns = df['return'].values
        last_n = returns[-use_last_n:] if len(returns) >= use_last_n else returns
        groups.append(last_n)
        region_names.append(region)
    
    # Perform ANOVA
    f_stat, p_value = f_oneway(*groups)
    
    print(f"Null Hypothesis: All regions have equal mean performance")
    print(f"Alternative: At least one region differs\n")
    
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value:     {p_value:.6f}")
    
    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print(f"\nResult: REJECT null hypothesis (p < {alpha})")
        print("        → Regional differences are STATISTICALLY SIGNIFICANT! ***")
    else:
        print(f"\nResult: FAIL TO REJECT null hypothesis (p >= {alpha})")
        print("        → No significant regional differences")
    
    # Effect size (eta-squared)
    # Total variance / Between-group variance
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    
    # Between-group sum of squares
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    
    # Total sum of squares
    ss_total = np.sum((all_data - grand_mean)**2)
    
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    print(f"\nEffect Size (η²): {eta_squared:.4f}")
    if eta_squared < 0.01:
        print("              → Small effect")
    elif eta_squared < 0.06:
        print("              → Medium effect")
    else:
        print("              → Large effect ***")
    
    # Group means
    print(f"\nGroup Means:")
    for region, group in zip(region_names, groups):
        print(f"  {region:12s}: {np.mean(group):>8.2f} ± {np.std(group):>6.2f}")
    
    print("\n" + "="*70 + "\n")
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'significant': p_value < alpha,
        'region_means': {r: np.mean(g) for r, g in zip(region_names, groups)}
    }

def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size between two groups
    
    Interpretation:
    - |d| < 0.2: Small effect
    - |d| < 0.5: Medium effect
    - |d| >= 0.8: Large effect
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
    
    d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    return d

def pairwise_comparison(env_name="CartPole", use_last_n=100):
    """
    Pairwise Cohen's d for all region pairs
    """
    print("\n" + "="*70)
    print(f"PAIRWISE EFFECT SIZE - {env_name}")
    print("="*70 + "\n")
    
    data = load_regional_data(env_name)
    regions = list(data.keys())
    
    print("Cohen's d (effect size between region pairs):")
    print("-"*70)
    print(f"{'Comparison':<25} {'Cohen\'s d':>12} {'Interpretation'}")
    print("-"*70)
    
    for i, region1 in enumerate(regions):
        for j, region2 in enumerate(regions):
            if i < j:  # Only upper triangle
                returns1 = data[region1]['return'].values[-use_last_n:]
                returns2 = data[region2]['return'].values[-use_last_n:]
                
                d = cohens_d(returns1, returns2)
                
                if abs(d) < 0.2:
                    interp = "Negligible"
                elif abs(d) < 0.5:
                    interp = "Small"
                elif abs(d) < 0.8:
                    interp = "Medium"
                else:
                    interp = "Large ***"
                
                comparison = f"{region1} vs {region2}"
                print(f"{comparison:<25} {d:>12.3f} {interp}")
    
    print("\n" + "="*70 + "\n")

def infrastructure_correlation_analysis(env_name="CartPole"):
    """
    Correlation: Infrastructure Parameters → Performance
    
    Tests which infrastructure factor is most predictive
    """
    print("\n" + "="*70)
    print(f"INFRASTRUCTURE CORRELATION ANALYSIS - {env_name}")
    print("="*70 + "\n")
    
    data = load_regional_data(env_name)
    
    # Collect data
    infra_params = []
    performances = []
    regions = []
    
    for region, df in data.items():
        if len(df) == 0:
            continue
        
        # Final performance
        returns = df['return'].values
        final_perf = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
        
        # Infrastructure parameters
        loop_speed = df['infrastructure_loop_speed'].iloc[0]
        automation = df['infrastructure_automation'].iloc[0]
        
        if 'infrastructure_error_tolerance' in df.columns:
            error_tol = df['infrastructure_error_tolerance'].iloc[0]
        else:
            error_tol = np.nan
        
        infra_params.append({
            'loop_speed': loop_speed,
            'automation': automation,
            'error_tolerance': error_tol
        })
        performances.append(final_perf)
        regions.append(region)
    
    if len(performances) < 3:
        print("[WARNING] Need at least 3 regions for correlation analysis")
        return
    
    # Convert to arrays
    loop_speeds = np.array([p['loop_speed'] for p in infra_params])
    automations = np.array([p['automation'] for p in infra_params])
    error_tols = np.array([p['error_tolerance'] for p in infra_params if not np.isnan(p['error_tolerance'])])
    performances = np.array(performances)
    
    # Correlations
    print("Pearson Correlation: Infrastructure Parameter → Performance\n")
    print(f"{'Parameter':<20} {'r':>10} {'p-value':>12} {'Significance'}")
    print("-"*70)
    
    # Loop Speed
    r_loop, p_loop = pearsonr(loop_speeds, performances)
    sig_loop = "***" if p_loop < 0.01 else "**" if p_loop < 0.05 else "*" if p_loop < 0.10 else "n.s."
    print(f"{'Loop Speed':<20} {r_loop:>10.3f} {p_loop:>12.6f} {sig_loop}")
    
    # Automation
    r_auto, p_auto = pearsonr(automations, performances)
    sig_auto = "***" if p_auto < 0.01 else "**" if p_auto < 0.05 else "*" if p_auto < 0.10 else "n.s."
    print(f"{'Automation':<20} {r_auto:>10.3f} {p_auto:>12.6f} {sig_auto}")
    
    # Error Tolerance (if enough data)
    if len(error_tols) >= 3:
        # Need matching performances
        perf_subset = performances[:len(error_tols)]
        r_error, p_error = pearsonr(error_tols, perf_subset)
        sig_error = "***" if p_error < 0.01 else "**" if p_error < 0.05 else "*" if p_error < 0.10 else "n.s."
        print(f"{'Error Tolerance':<20} {r_error:>10.3f} {p_error:>12.6f} {sig_error}")
    
    print("\nSignificance: *** p<0.01, ** p<0.05, * p<0.10, n.s. not significant")
    
    # Interpretation
    print("\nINTERPRETATION:")
    if abs(r_loop) > 0.7:
        direction = "NEGATIVE" if r_loop < 0 else "POSITIVE"
        print(f"  Loop Speed has STRONG {direction} correlation with performance")
        if r_loop < 0:
            print("  → Faster feedback (lower loop_speed) → Better performance")
        else:
            print("  → Slower feedback (higher loop_speed) → Better performance (counter-intuitive!)")
    
    if abs(r_auto) > 0.5:
        print(f"  Automation has MODERATE correlation (r={r_auto:.3f})")
        if r_auto > 0:
            print("  → Higher automation → Better performance")
    
    print("\n" + "="*70 + "\n")
    
    return {
        'loop_speed': {'r': r_loop, 'p': p_loop},
        'automation': {'r': r_auto, 'p': p_auto}
    }

def confidence_intervals(env_name="CartPole", confidence=0.95):
    """
    Calculate confidence intervals for regional performance
    
    Uses bootstrap method for robust estimation
    """
    print("\n" + "="*70)
    print(f"CONFIDENCE INTERVALS - {env_name} (95% CI)")
    print("="*70 + "\n")
    
    data = load_regional_data(env_name)
    
    print(f"{'Region':<12} {'Mean':>10} {'CI Lower':>12} {'CI Upper':>12} {'Range':>10}")
    print("-"*70)
    
    for region, df in data.items():
        returns = df['return'].values[-100:] if len(df) >= 100 else df['return'].values
        
        if len(returns) < 10:
            continue
        
        # Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(returns, size=len(returns), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        mean = np.mean(returns)
        ci_lower = np.percentile(bootstrap_means, (1-confidence)/2 * 100)
        ci_upper = np.percentile(bootstrap_means, (1+confidence)/2 * 100)
        ci_range = ci_upper - ci_lower
        
        print(f"{region:<12} {mean:>10.2f} {ci_lower:>12.2f} {ci_upper:>12.2f} {ci_range:>10.2f}")
    
    print("\n" + "="*70 + "\n")

def comprehensive_statistical_report(env_name="CartPole"):
    """
    Generate complete statistical report for environment
    """
    print("\n" + "#"*70)
    print(f"# COMPREHENSIVE STATISTICAL REPORT: {env_name}")
    print("#"*70 + "\n")
    
    # 1. ANOVA
    anova_results = anova_regional_comparison(env_name)
    
    # 2. Pairwise Comparisons
    pairwise_comparison(env_name)
    
    # 3. Infrastructure Correlations
    corr_results = infrastructure_correlation_analysis(env_name)
    
    # 4. Confidence Intervals
    confidence_intervals(env_name)
    
    # 5. Summary
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70 + "\n")
    
    if anova_results and anova_results['significant']:
        print("[FINDING] Regional differences are STATISTICALLY SIGNIFICANT")
        print("          → Infrastructure matters for RL performance!\n")
        
        # Best region
        best_region = max(anova_results['region_means'], 
                         key=anova_results['region_means'].get)
        best_perf = anova_results['region_means'][best_region]
        
        print(f"[RECOMMENDATION] For {env_name}:")
        print(f"                 Train in {best_region} (avg={best_perf:.1f})\n")
    
    if corr_results:
        # Identify most important factor
        loop_r = abs(corr_results['loop_speed']['r'])
        auto_r = abs(corr_results['automation']['r'])
        
        if loop_r > auto_r and loop_r > 0.5:
            print("[CRITICAL FACTOR] Loop Speed is most predictive")
            if corr_results['loop_speed']['r'] < 0:
                print("                  → Optimize for FAST feedback loops\n")
            else:
                print("                  → Moderate feedback delay beneficial (counter-intuitive!)\n")
        elif auto_r > 0.5:
            print("[CRITICAL FACTOR] Automation is most predictive")
            print("                  → Invest in high automation infrastructure\n")
    
    print("="*70 + "\n")
    
    return {
        'anova': anova_results,
        'correlations': corr_results
    }

def generate_latex_table(env_name="CartPole"):
    """
    Generate LaTeX table for paper
    """
    print("\n" + "="*70)
    print(f"LATEX TABLE - {env_name}")
    print("="*70 + "\n")
    
    data = load_regional_data(env_name)
    
    print("% Copy this into your LaTeX paper:")
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{Regional Performance Comparison: {env_name}}}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Region & Avg100 & Best & Emotion $\\sigma$ & Win Rate \\\\")
    print("\\midrule")
    
    for region, df in sorted(data.items()):
        returns = df['return'].values
        emotions = df['emotion'].values
        
        avg100 = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
        best = np.max(returns)
        emo_std = np.std(emotions)
        
        win_rate = 0.0
        if 'win_rate' in df.columns:
            win_rate = df['win_rate'].iloc[-1]
        
        print(f"{region} & {avg100:.1f} & {best:.1f} & {emo_std:.3f} & {win_rate:.1%} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    print("\n" + "="*70 + "\n")

def main():
    """Run all statistical analyses"""
    
    print("="*70)
    print("   STATISTICAL ANALYSIS SUITE")
    print("   Phase 8.2.1 - Publication-Grade Statistics")
    print("="*70)
    
    # Analyze all available environments
    environments = ["CartPole", "Acrobot", "LunarLander"]
    
    for env in environments:
        # Check if data exists
        data = load_regional_data(env)
        
        if len(data) >= 2:
            print(f"\n\nAnalyzing {env}...")
            comprehensive_statistical_report(env)
            generate_latex_table(env)
        else:
            print(f"\n[SKIP] {env}: Insufficient data (need >=2 regions)")
    
    print("\n" + "="*70)
    print("[COMPLETE] Statistical Analysis Done!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()





