#!/usr/bin/env python3

"""
RookWorld GRPO Hyperparameter Sweep Results Analyzer

This script analyzes the results from hyperparameter_sweep.sh to identify:
1. Most stable parameter combinations (lowest KL divergence)
2. Success rate by parameter value
3. Parameter interaction effects
4. Optimal parameter recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys

def load_results(csv_path):
    """Load sweep results from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} experimental runs")
        return df
    except FileNotFoundError:
        print(f"❌ Results file not found: {csv_path}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"❌ Results file is empty: {csv_path}")
        sys.exit(1)

def analyze_success_rates(df):
    """Analyze success rates by parameter values."""
    print("\n🎯 SUCCESS RATE ANALYSIS")
    print("=" * 50)
    
    # Overall success rate
    total_runs = len(df)
    successful_runs = len(df[df['status'] == 'SUCCESS'])
    diverged_runs = len(df[df['status'] == 'DIVERGED'])
    timeout_runs = len(df[df['status'] == 'TIMEOUT'])
    oom_runs = len(df[df['status'] == 'OOM'])
    
    print(f"Overall Success Rate: {successful_runs}/{total_runs} ({successful_runs/total_runs*100:.1f}%)")
    print(f"Diverged: {diverged_runs} ({diverged_runs/total_runs*100:.1f}%)")
    print(f"OOM (Memory): {oom_runs} ({oom_runs/total_runs*100:.1f}%)")
    print(f"Timeouts: {timeout_runs} ({timeout_runs/total_runs*100:.1f}%)")
    print()
    
    # Success rate by parameter
    parameters = ['kl_warmup_steps', 'kl_warmup_factor', 'learning_rate', 'kl_coefficient']
    
    for param in parameters:
        print(f"📈 Success Rate by {param}:")
        success_by_param = df.groupby(param)['status'].apply(
            lambda x: (x == 'SUCCESS').sum() / len(x) * 100
        ).round(1)
        for value, rate in success_by_param.items():
            print(f"  {value}: {rate}%")
        print()

def analyze_stability(df):
    """Analyze KL divergence stability for all runs with data."""
    # Include all runs that have KL measurements (SUCCESS, OOM, TIMEOUT with data)
    df_with_kl = df[df['final_kl_mean'] != 'N/A'].copy()
    successful_df = df[df['status'] == 'SUCCESS'].copy()
    
    if len(df_with_kl) == 0:
        print("⚠️  No runs with KL measurements to analyze stability")
        return
    
    print("\n🔬 STABILITY ANALYSIS")
    print("=" * 50)
    
    if len(successful_df) > 0:
        print(f"Successful runs: {len(successful_df)}")
    if len(df_with_kl) > len(successful_df):
        print(f"Runs with partial data: {len(df_with_kl) - len(successful_df)}")
    print()
    
    # Convert final_kl_mean to numeric, handling 'N/A'
    df_with_kl['final_kl_numeric'] = pd.to_numeric(df_with_kl['final_kl_mean'], errors='coerce')
    
    # Remove rows where KL couldn't be parsed
    valid_kl_df = df_with_kl.dropna(subset=['final_kl_numeric'])
    
    if len(valid_kl_df) == 0:
        print("⚠️  No runs with valid KL measurements")
        return
    
    print(f"KL Divergence Statistics ({len(valid_kl_df)} runs):")
    print(f"  Mean: {valid_kl_df['final_kl_numeric'].mean():.2f}")
    print(f"  Median: {valid_kl_df['final_kl_numeric'].median():.2f}")
    print(f"  Std: {valid_kl_df['final_kl_numeric'].std():.2f}")
    print(f"  Min: {valid_kl_df['final_kl_numeric'].min():.2f}")
    print(f"  Max: {valid_kl_df['final_kl_numeric'].max():.2f}")
    print()
    
    # Most stable configurations (lowest KL)
    print("🏆 MOST STABLE CONFIGURATIONS (Lowest KL):")
    top_stable = valid_kl_df.nsmallest(10, 'final_kl_numeric')[
        ['kl_warmup_steps', 'kl_warmup_factor', 'learning_rate', 'kl_coefficient', 'final_kl_numeric', 'status', 'steps_completed', 'training_time']
    ]
    
    for i, (_, row) in enumerate(top_stable.iterrows(), 1):
        status_emoji = "✅" if row['status'] == 'SUCCESS' else "💾" if row['status'] == 'OOM' else "⏱️" if row['status'] == 'TIMEOUT' else "💥"
        print(f"{i:2d}. {status_emoji} KL={row['final_kl_numeric']:5.2f} | "
              f"warmup_steps={row['kl_warmup_steps']:3.0f} | "
              f"warmup_factor={row['kl_warmup_factor']:3.1f} | "
              f"lr={row['learning_rate']:.1e} | "
              f"kl_coef={row['kl_coefficient']:.4f} | "
              f"steps={row['steps_completed']} | "
              f"time={row['training_time']}s")
    print()

def analyze_longest_running(df):
    """Analyze longest-running configurations as a stability metric."""
    print("\n⏱️ LONGEST-RUNNING CONFIGURATIONS")
    print("=" * 50)
    
    # Filter runs with valid step and time data
    df_with_data = df[(df['steps_completed'] != 'N/A') & (df['training_time'] > 0)].copy()
    
    if len(df_with_data) == 0:
        print("⚠️  No runs with step/time data")
        return
    
    # Convert steps to numeric
    df_with_data['steps_numeric'] = pd.to_numeric(df_with_data['steps_completed'], errors='coerce')
    df_with_data = df_with_data.dropna(subset=['steps_numeric'])
    
    print(f"Training Duration Analysis ({len(df_with_data)} runs):")
    print(f"  Mean training time: {df_with_data['training_time'].mean():.1f}s")
    print(f"  Max training time: {df_with_data['training_time'].max():.0f}s")
    print(f"  Mean steps completed: {df_with_data['steps_numeric'].mean():.1f}")
    print(f"  Max steps completed: {df_with_data['steps_numeric'].max():.0f}")
    print()
    
    # Top configurations by training duration
    print("🏃 TOP CONFIGURATIONS BY TRAINING DURATION:")
    longest_duration = df_with_data.nlargest(10, 'training_time')[
        ['kl_warmup_steps', 'kl_warmup_factor', 'learning_rate', 'kl_coefficient', 'status', 'steps_completed', 'training_time', 'final_kl_mean']
    ]
    
    for i, (_, row) in enumerate(longest_duration.iterrows(), 1):
        status_emoji = "✅" if row['status'] == 'SUCCESS' else "💾" if row['status'] == 'OOM' else "⏱️" if row['status'] == 'TIMEOUT' else "💥"
        kl_display = f"{float(row['final_kl_mean']):.1f}" if row['final_kl_mean'] != 'N/A' else "N/A"
        print(f"{i:2d}. {status_emoji} {row['training_time']:3.0f}s | "
              f"steps={row['steps_completed']} | "
              f"KL={kl_display} | "
              f"warmup_steps={row['kl_warmup_steps']:3.0f} | "
              f"warmup_factor={row['kl_warmup_factor']:3.1f} | "
              f"lr={row['learning_rate']:.1e} | "
              f"kl_coef={row['kl_coefficient']:.4f}")
    
    print()
    
    # Top configurations by steps completed
    print("🎯 TOP CONFIGURATIONS BY STEPS COMPLETED:")
    longest_steps = df_with_data.nlargest(10, 'steps_numeric')[
        ['kl_warmup_steps', 'kl_warmup_factor', 'learning_rate', 'kl_coefficient', 'status', 'steps_completed', 'training_time', 'final_kl_mean']
    ]
    
    for i, (_, row) in enumerate(longest_steps.iterrows(), 1):
        status_emoji = "✅" if row['status'] == 'SUCCESS' else "💾" if row['status'] == 'OOM' else "⏱️" if row['status'] == 'TIMEOUT' else "💥"
        kl_display = f"{float(row['final_kl_mean']):.1f}" if row['final_kl_mean'] != 'N/A' else "N/A"
        print(f"{i:2d}. {status_emoji} {row['steps_completed']} steps | "
              f"{row['training_time']:3.0f}s | "
              f"KL={kl_display} | "
              f"warmup_steps={row['kl_warmup_steps']:3.0f} | "
              f"warmup_factor={row['kl_warmup_factor']:3.1f} | "
              f"lr={row['learning_rate']:.1e} | "
              f"kl_coef={row['kl_coefficient']:.4f}")
    
    print()


def analyze_divergence_patterns(df):
    """Analyze when and why training diverges."""
    diverged_df = df[df['status'] == 'DIVERGED'].copy()
    oom_df = df[df['status'] == 'OOM'].copy()
    
    if len(diverged_df) == 0 and len(oom_df) == 0:
        print("\n✅ No failed runs to analyze")
        return
    
    print("\n💥 FAILURE ANALYSIS")
    print("=" * 50)
    
    # Analyze divergence patterns
    if len(diverged_df) > 0:
        # Convert diverged_at_step to numeric
        diverged_df['diverged_step_numeric'] = pd.to_numeric(diverged_df['diverged_at_step'], errors='coerce')
        valid_divergence_df = diverged_df.dropna(subset=['diverged_step_numeric'])
        
        if len(valid_divergence_df) > 0:
            print(f"💥 Divergence Statistics ({len(valid_divergence_df)} runs):")
            print(f"  Mean divergence step: {valid_divergence_df['diverged_step_numeric'].mean():.1f}")
            print(f"  Median divergence step: {valid_divergence_df['diverged_step_numeric'].median():.1f}")
            print(f"  Earliest divergence: {valid_divergence_df['diverged_step_numeric'].min():.0f}")
            print(f"  Latest divergence: {valid_divergence_df['diverged_step_numeric'].max():.0f}")
            print()
    
    # Analyze OOM patterns  
    if len(oom_df) > 0:
        oom_df['steps_numeric'] = pd.to_numeric(oom_df['steps_completed'], errors='coerce')
        valid_oom_df = oom_df.dropna(subset=['steps_numeric'])
        
        if len(valid_oom_df) > 0:
            print(f"💾 OOM Statistics ({len(valid_oom_df)} runs):")
            print(f"  Mean OOM step: {valid_oom_df['steps_numeric'].mean():.1f}")
            print(f"  Median OOM step: {valid_oom_df['steps_numeric'].median():.1f}")
            print(f"  Earliest OOM: {valid_oom_df['steps_numeric'].min():.0f}")
            print(f"  Latest OOM: {valid_oom_df['steps_numeric'].max():.0f}")
            print(f"  Mean time to OOM: {valid_oom_df['training_time'].mean():.1f}s")
            print()
    
    # Most problematic parameter combinations
    print("⚠️  MOST PROBLEMATIC PARAMETERS:")
    parameters = ['kl_warmup_steps', 'kl_warmup_factor', 'learning_rate', 'kl_coefficient']
    
    for param in parameters:
        divergence_rate = df.groupby(param)['status'].apply(
            lambda x: (x == 'DIVERGED').sum() / len(x) * 100
        ).round(1)
        oom_rate = df.groupby(param)['status'].apply(
            lambda x: (x == 'OOM').sum() / len(x) * 100
        ).round(1)
        
        worst_div_param = divergence_rate.idxmax() if divergence_rate.max() > 0 else None
        worst_oom_param = oom_rate.idxmax() if oom_rate.max() > 0 else None
        
        if worst_div_param:
            print(f"  {param} divergence: {worst_div_param} ({divergence_rate.max()}%)")
        if worst_oom_param:
            print(f"  {param} OOM: {worst_oom_param} ({oom_rate.max()}%)")
    print()

def generate_recommendations(df):
    """Generate parameter recommendations based on analysis."""
    print("\n💡 PARAMETER RECOMMENDATIONS")
    print("=" * 50)
    
    successful_df = df[df['status'] == 'SUCCESS']
    df_with_data = df[(df['steps_completed'] != 'N/A') & (df['training_time'] > 0)].copy()
    
    # Strategy 1: Success rate recommendations (if any successful runs)
    if len(successful_df) > 0:
        print("🎯 SUCCESS-BASED RECOMMENDATIONS:")
        parameters = ['kl_warmup_steps', 'kl_warmup_factor', 'learning_rate', 'kl_coefficient']
        
        recommendations = {}
        for param in parameters:
            success_rates = df.groupby(param)['status'].apply(lambda x: (x == 'SUCCESS').sum() / len(x))
            best_value = success_rates.idxmax()
            best_rate = success_rates.max() * 100
            recommendations[param] = (best_value, best_rate)
            print(f"  {param}: {best_value} (Success rate: {best_rate:.1f}%)")
        
        print()
        print("🚀 SUCCESS-BASED TRAIN.SH CONFIGURATION:")
        print(f"KL_WARMUP_STEPS={recommendations['kl_warmup_steps'][0]}")
        print(f"KL_WARMUP_FACTOR={recommendations['kl_warmup_factor'][0]}")
        print(f"LR={recommendations['learning_rate'][0]}")
        print(f"KL_COEF={recommendations['kl_coefficient'][0]}")
        print()
        
        # Best single successful configuration
        successful_df['final_kl_numeric'] = pd.to_numeric(successful_df['final_kl_mean'], errors='coerce')
        best_config = successful_df.loc[successful_df['final_kl_numeric'].idxmin()]
        
        print("🏆 BEST SUCCESSFUL CONFIGURATION (Lowest KL):")
        print(f"KL_WARMUP_STEPS={int(best_config['kl_warmup_steps'])}")
        print(f"KL_WARMUP_FACTOR={best_config['kl_warmup_factor']}")
        print(f"LR={best_config['learning_rate']}")
        print(f"KL_COEF={best_config['kl_coefficient']}")
        print(f"Final KL: {best_config['final_kl_numeric']:.2f}")
        print(f"Final Reward: {best_config['final_reward']}")
        print()
    
    # Strategy 2: Stability-based recommendations (longest running)
    if len(df_with_data) > 0:
        print("⏱️ STABILITY-BASED RECOMMENDATIONS (Longest Running):")
        
        # Convert steps to numeric for analysis
        df_with_data['steps_numeric'] = pd.to_numeric(df_with_data['steps_completed'], errors='coerce')
        df_with_data = df_with_data.dropna(subset=['steps_numeric'])
        
        parameters = ['kl_warmup_steps', 'kl_warmup_factor', 'learning_rate', 'kl_coefficient']
        
        stability_recommendations = {}
        for param in parameters:
            # Find parameter value with highest mean steps completed
            mean_steps_by_param = df_with_data.groupby(param)['steps_numeric'].mean()
            best_value = mean_steps_by_param.idxmax()
            best_steps = mean_steps_by_param.max()
            stability_recommendations[param] = (best_value, best_steps)
            print(f"  {param}: {best_value} (Mean steps: {best_steps:.1f})")
        
        print()
        print("⚡ STABILITY-BASED TRAIN.SH CONFIGURATION:")
        print(f"KL_WARMUP_STEPS={stability_recommendations['kl_warmup_steps'][0]}")
        print(f"KL_WARMUP_FACTOR={stability_recommendations['kl_warmup_factor'][0]}")
        print(f"LR={stability_recommendations['learning_rate'][0]}")
        print(f"KL_COEF={stability_recommendations['kl_coefficient'][0]}")
        print()
        
        # Single most stable configuration (longest running)
        most_stable = df_with_data.loc[df_with_data['steps_numeric'].idxmax()]
        
        print("🥇 MOST STABLE SINGLE CONFIGURATION (Longest Running):")
        print(f"KL_WARMUP_STEPS={int(most_stable['kl_warmup_steps'])}")
        print(f"KL_WARMUP_FACTOR={most_stable['kl_warmup_factor']}")
        print(f"LR={most_stable['learning_rate']}")
        print(f"KL_COEF={most_stable['kl_coefficient']}")
        print(f"Steps Completed: {int(most_stable['steps_numeric'])}")
        print(f"Training Time: {most_stable['training_time']}s")
        print(f"Status: {most_stable['status']}")
        if most_stable['final_kl_mean'] != 'N/A':
            print(f"Final KL: {most_stable['final_kl_mean']}")
    
    if len(successful_df) == 0 and len(df_with_data) == 0:
        print("⚠️  No runs with usable data - cannot generate recommendations")

def create_visualizations(df, output_dir):
    """Create visualization plots if matplotlib is available."""
    try:
        output_path = Path(output_dir)
        
        # Success rate heatmap by key parameters
        if len(df[df['status'] == 'SUCCESS']) > 0:
            plt.figure(figsize=(12, 8))
            
            # Create success rate pivot table
            pivot_data = df.groupby(['kl_warmup_steps', 'learning_rate'])['status'].apply(
                lambda x: (x == 'SUCCESS').sum() / len(x) * 100
            ).unstack(fill_value=0)
            
            sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                       cbar_kws={'label': 'Success Rate (%)'})
            plt.title('Success Rate by KL Warmup Steps vs Learning Rate')
            plt.xlabel('Learning Rate')
            plt.ylabel('KL Warmup Steps')
            plt.tight_layout()
            plt.savefig(output_path / 'success_rate_heatmap.png', dpi=300)
            plt.close()
            
            print(f"📊 Visualization saved: {output_path / 'success_rate_heatmap.png'}")
        
    except ImportError:
        print("📊 Matplotlib not available - skipping visualizations")

def main():
    parser = argparse.ArgumentParser(description='Analyze RookWorld GRPO hyperparameter sweep results')
    parser.add_argument('results_csv', help='Path to the results CSV file')
    parser.add_argument('--output-dir', help='Directory for output files', default='.')
    
    args = parser.parse_args()
    
    print("🔬 RookWorld GRPO Hyperparameter Sweep Analysis")
    print("=" * 60)
    
    # Load and analyze results
    df = load_results(args.results_csv)
    
    analyze_success_rates(df)
    analyze_stability(df)
    analyze_longest_running(df)
    analyze_divergence_patterns(df)
    generate_recommendations(df)
    
    # Create visualizations
    create_visualizations(df, args.output_dir)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()