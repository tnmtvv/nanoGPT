#!/usr/bin/env python3
"""
Script to combine CSV files, filter by seed, select best val loss per batch size,
and create a seaborn plot.

Usage:
    python analyze_results.py --csv-dir csvs --output-dir plots
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_and_combine_csvs(csv_dir):
    """
    Load all CSV files from directory and combine into one DataFrame.
    
    Args:
        csv_dir: Path to directory containing CSV files
        
    Returns:
        Combined DataFrame
    """
    csv_dir = Path(csv_dir)
    csv_files = list(csv_dir.glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {csv_dir}")
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Load and combine all CSVs
    dfs = []
    for csv_file in csv_files:
        print(f"  Loading: {csv_file.name}")
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined dataset shape: {combined_df.shape}")
    print(f"Columns: {combined_df.columns.tolist()}")
    
    return combined_df


def filter_by_seed(df, seed=1):
    """
    Filter DataFrame to keep only rows with specified seed.
    
    Args:
        df: Input DataFrame
        seed: Seed value to filter by
        
    Returns:
        Filtered DataFrame
    """
    if 'seed' not in df.columns:
        print("Warning: 'seed' column not found. Skipping seed filtering.")
        return df
    
    filtered_df = df[df['seed'] == seed].copy()
    print(f"\nFiltered to seed={seed}: {len(filtered_df)} rows")
    
    return filtered_df


def get_best_val_loss_per_batch_size(df):
    """
    For each (batch_size, optimizer_name, seed), keep only the row 
    with the best (minimum) val_loss across iterations.
    """
    required_cols = ['batch_size', 'val_loss', 'optimizer_name', 'seed']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return df
    
    # Group by batch_size, optimizer_name, AND seed
    best_df = df.loc[df.groupby(['batch_size', 'optimizer_name', 'seed'])['val_loss'].idxmin()]
    
    print(f"\nBest results per (batch_size, optimizer_name, seed): {len(best_df)} rows")
    
    return best_df.sort_values(['batch_size', 'optimizer_name', 'seed']).reset_index(drop=True)


import numpy as np


def clean_seeds(df):
    opt_names = np.unique(df['optimizer_name'])
    opt_csvs_correct = []
    
    for opt_name in opt_names:
        opt_df = df.query('optimizer_name == @opt_name').copy()

        # A new run starts whenever iter_num resets to 0
        opt_df["is_new_run"] = opt_df["iter_num"] == 0

        # IMPORTANT: compute run_id per experiment stream, not just per seed
        # (otherwise keeping run_id==0 drops other batch sizes / lrs)
        run_keys = ["seed", "batch_size", "learning_rate"]
        opt_df["run_id"] = opt_df.groupby(run_keys)["is_new_run"].cumsum().astype(int) - 1

        # Keep only the first run within each (seed, batch_size, learning_rate)
        df_single = opt_df[opt_df["run_id"] == 0].copy()
        df_single = df_single.drop(columns=["run_id", "is_new_run"])

        opt_csvs_correct.append(df_single)
    
    return pd.concat(opt_csvs_correct, ignore_index=True)

    


def create_plot(df, output_dir='plots', figsize=(10, 6)):
    """
    Create seaborn plot with batch_size on x-axis, val_loss on y-axis, 
    and optimizer_name as hue.
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save plot
        figsize: Figure size tuple
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create lineplot
    sns.lineplot(
        data=df,
        x='batch_size',
        y='val_loss',
        hue='optimizer_name',
        marker='o',
        errorbar=('se', 2),
        markersize=8,
        linewidth=2.5,
        ax=ax
    )
    
    # Customize plot
    ax.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Best Validation Loss', fontsize=14, fontweight='bold')
    ax.set_title('Validation Loss vs Batch Size by optimizer_name', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Improve legend
    ax.legend(title='optimizer_name', fontsize=11, title_fontsize=12, 
              frameon=True, shadow=True)
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'val_loss_vs_batch_size.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {plot_path}")
    
    # Also save as PDF
    pdf_path = output_dir / 'val_loss_vs_batch_size.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ Plot saved to: {pdf_path}")
    
    # Show plot
    plt.show()
    
    return fig, ax


def save_intermediate_results(combined_df, seed_df, best_df, output_dir='results'):
    """
    Save intermediate DataFrames as CSV files.
    
    Args:
        combined_df: Combined DataFrame
        seed_df: Seed-filtered DataFrame
        best_df: Best results DataFrame
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save combined data
    combined_path = output_dir / 'combined_all.csv'
    combined_df.to_csv(combined_path, index=False)
    print(f"\n✓ Saved combined data: {combined_path}")
    
    # Save seed-filtered data
    seed_path = output_dir / 'filtered_seed1.csv'
    seed_df.to_csv(seed_path, index=False)
    print(f"✓ Saved seed=1 data: {seed_path}")
    
    # Save best results
    best_path = output_dir / 'best_results.csv'
    best_df.to_csv(best_path, index=False)
    print(f"✓ Saved best results: {best_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    print(best_df.groupby('optimizer_name')['val_loss'].describe())


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Analyze experiment results and create plots'
    )
    parser.add_argument(
        '--csv-dir',
        type=str,
        default='csvs',
        help='Directory containing CSV files (default: csvs)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='plots',
        help='Directory to save plots (default: plots)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory to save intermediate CSV files (default: results)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Seed value to filter by (default: 1)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip creating the plot'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Experiment Results Analyzer")
    print("="*60)
    
    # Step 1: Load and combine CSVs
    print("\n[Step 1/4] Loading and combining CSV files...")
    combined_df = load_and_combine_csvs(args.csv_dir)

    new_combined_df = clean_seeds(combined_df)
    print(np.unique(new_combined_df['batch_size']))
    
    # Step 2: Filter by seed
    seeds = [1, 2, 3, 4, 5]
    seed_dfs = []

    # keep only these seeds
    seed_df = combined_df[combined_df["seed"].isin(seeds)].copy()

    # now this will return 1 row per (batch_size, optimizer_name, seed)
    new_df = get_best_val_loss_per_batch_size(seed_df)
    # new_df = pd.concat(seed_dfs, ignore_index=True)
    # To this:
    print("\nSample of concatenated data (new_df):")
    print(new_df[['batch_size', 'optimizer_name', 'val_loss', 'seed']].head(40))
    print(f"\nTotal rows: {len(new_df)}")
    print(f"Expected: {len(seeds)} seeds × {new_df[['batch_size', 'optimizer_name']].drop_duplicates().shape[0]} groups")

    new_df.to_csv("batch_results.csv")
    # # Display best results
    # print("\nBest Results:")
    # print(best_df[['batch_size', 'optimizer_name', 'val_loss', 'seed']].to_string(index=False))
    
    # # Step 4: Create plot
    # if not args.no_plot:
    #     print("\n[Step 4/4] Creating plot...")
    #     create_plot(new_df, output_dir=args.output_dir)
    
    # # Save intermediate results
    # save_intermediate_results(combined_df, combined_df, new_df, args.results_dir)
    
    # print("\n" + "="*60)
    # print("✓ Analysis complete!")
    # print("="*60)


if __name__ == "__main__":
    main()
