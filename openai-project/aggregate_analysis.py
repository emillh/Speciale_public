import os
import glob
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- AESTHETIC REFINEMENTS: Define global styles ---
# 1. Define a consistent color palette for 5, 15, 25 turns
TURN_COLORS = {
    5: "#485690",  # Dark Blue/Purple
    15: "#429590", # Teal
    25: "#90C987"  # Light Green
}

# 2. Set a global font size for better readability
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
# --- END REFINEMENTS ---

def get_turn_count(run_folder_path):
    """Reads the gen_0 log file to determine the number of turns in a run."""
    try:
        log_file = glob.glob(os.path.join(run_folder_path, "logs", "*_gen_0.txt"))[0]
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            return content.count("SalesAgent:")
    except (IndexError, FileNotFoundError):
        # Return a value that indicates an error, e.g., -1 or None
        return -1

def load_and_aggregate_data(results_dir):
    """
    Loads all metric files from all run folders and aggregates them into a single
    long-format DataFrame where each row is one generation per run.
    """
    all_run_data = []
    run_folders = sorted([d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))])

    for run_id in run_folders:
        run_path = os.path.join(results_dir, run_id)
        metrics_path = os.path.join(run_path, "metrics")

        turns = get_turn_count(run_path)
        if turns == -1:
            print(f"Warning: Could not determine turn count for {run_id}. Skipping.")
            continue

        try:
            # --- Load all dataframes for the run ---
            decay_df = pd.read_csv(os.path.join(metrics_path, "information_decay_analysis.csv"))
            fidelity_df = pd.read_csv(os.path.join(metrics_path, "information_fidelity_analysis.csv"))
            judge_df = pd.read_csv(os.path.join(metrics_path, "multi_judge_analysis.csv"))

            # --- Clean and standardize decay_df (per-generation data) ---
            if 'Generation' in decay_df.columns and 'generation' not in decay_df.columns:
                decay_df.rename(columns={'Generation': 'generation'}, inplace=True)
            rename_map = {'Field_Fill_Rate': 'filled_field_fraction'}
            decay_df.rename(columns=rename_map, inplace=True)
            decay_df['generation'] = pd.to_numeric(decay_df['generation'], errors='coerce')
            decay_df.dropna(subset=['generation'], inplace=True)
            decay_df['generation'] = decay_df['generation'].astype(int)
            # Select only the columns we need to avoid duplicates
            decay_df = decay_df[['generation', 'filled_field_fraction']]

            # --- Aggregate per-field data to per-generation data ---
            fidelity_agg = fidelity_df.groupby('generation')[['levenshtein_similarity', 'cosine_similarity']].mean().reset_index()
            judge_agg = judge_df.groupby('generation')[['score_gpt4o', 'score_llama3-70b']].mean().reset_index()

            # --- Merge the aggregated dataframes ---
            # Start with the per-generation decay_df
            run_df = decay_df.copy()
            # Merge the other aggregated metrics
            run_df = pd.merge(run_df, fidelity_agg, on="generation", how="left")
            run_df = pd.merge(run_df, judge_agg, on="generation", how="left")

            # Add the run-specific identifiers
            run_df['run_id'] = run_id
            run_df['turns'] = turns
            
            all_run_data.append(run_df)
        except FileNotFoundError as e:
            print(f"Warning: Missing a metric file in {run_id}. Skipping. Details: {e}")
            continue
            
    if not all_run_data:
        raise ValueError("No data was loaded. Please check your results folders.")

    master_df = pd.concat(all_run_data, ignore_index=True)
    return master_df

def plot_mean_curves(df, output_dir):
    """Plots the mean of specified metrics over generations with SD bands."""
    # Simplified titles and defined y-labels
    metrics_to_plot = {
        'filled_field_fraction': {
            'title': 'Mean Filled Field Fraction Over Generations (±1 SD)',
            'ylabel': 'Filled Field Fraction'
        },
        'cosine_similarity': {
            'title': 'Mean Cosine Similarity Over Generations (±1 SD)',
            'ylabel': 'Cosine Similarity'
        },
        'score_gpt4o': {
            'title': 'Mean Judge Score (GPT-4o) Over Generations (±1 SD)',
            'ylabel': 'Average Judge Score (GPT-4o)'
        },
        'score_llama3-70b': {
            'title': 'Mean Judge Score (Llama3-70b) Over Generations (±1 SD)',
            'ylabel': 'Average Judge Score (Llama3-70b)'
        },
    }

    for metric_col, plot_info in metrics_to_plot.items():
        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(
            data=df,
            x='generation',
            y=metric_col,
            hue='turns',
            palette=TURN_COLORS,
            errorbar='sd',
            legend='full'
        )
        
        # --- AESTHETIC REFINEMENTS ---
        # Set Y-axis limits based on metric type
        if 'score' in metric_col:
            ax.set_ylim(0, 5) # Use full 0-5 scale for judge scores
        elif metric_col in ['filled_field_fraction', 'cosine_similarity']:
            ax.set_ylim(bottom=0) # Start other metrics at 0
        # --- END REFINEMENTS ---

        plt.title(plot_info['title'])
        plt.xlabel('Generation')
        plt.ylabel(plot_info['ylabel']) # Set the explicit, shorter y-label
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(sorted(df['generation'].unique()))
        plt.legend(title='Turn Count')
        
        plot_path = os.path.join(output_dir, f'mean_curve_{metric_col}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved mean curve plot to: {plot_path}")

def analyze_and_plot_summaries(df, output_dir):
    """
    Calculates scalar summaries (final value, AUC) for each run and
    creates boxplots to compare them across turn groups.
    """
    # --- 1. Calculate Scalar Summaries ---
    summary_data = []
    for run_id, group in df.groupby('run_id'):
        turns = group['turns'].iloc[0]
        
        # Final-generation value
        final_gen_row = group.loc[group['generation'].idxmax()]
        
        # --- REFINEMENT: Be explicit about which columns to average for AUC ---
        # This avoids averaging columns like 'generation' or 'turns' by mistake.
        metrics_for_auc = [
            'filled_field_fraction', 'cosine_similarity', 
            'score_gpt4o', 'score_llama3-70b'
        ]
        # Filter for columns that actually exist in the dataframe to avoid errors
        existing_metrics = [m for m in metrics_for_auc if m in group.columns]
        auc_metrics = group[existing_metrics].mean()
        # --- END REFINEMENT ---
        
        summary_data.append({
            'run_id': run_id,
            'turns': turns,
            'final_fill_rate': final_gen_row['filled_field_fraction'],
            'final_cosine_sim': final_gen_row['cosine_similarity'],
            'final_judge_score_gpt4o': final_gen_row['score_gpt4o'],
            'auc_fill_rate': auc_metrics['filled_field_fraction'],
            'auc_cosine_sim': auc_metrics['cosine_similarity'],
            'auc_judge_score_gpt4o': auc_metrics['score_gpt4o']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # --- FIX: Ensure 'turns' column is numeric for correct palette mapping ---
    summary_df['turns'] = pd.to_numeric(summary_df['turns'])
    # --- END FIX ---

    summary_df.to_csv(os.path.join(output_dir, 'run_summary_statistics.csv'), index=False)
    print(f"Saved summary statistics to: {os.path.join(output_dir, 'run_summary_statistics.csv')}")

    # --- 2. Create Boxplots for Summaries ---
    summary_metrics_to_plot = {
        'final_judge_score_gpt4o': 'Final Generation Judge Score (GPT-4o)',
        'auc_judge_score_gpt4o': 'Overall (AUC) Judge Score (GPT-4o)',
        'final_fill_rate': 'Final Generation Fill Rate',
        'auc_fill_rate': 'Overall (AUC) Fill Rate'
    }

    for metric, title_template in summary_metrics_to_plot.items():
        # Create a more descriptive title and y-label
        title = f"Comparison of {title_template} by Turn Count"
        y_label = title_template

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=summary_df, x='turns', y=metric, palette=TURN_COLORS, hue='turns', legend=False)
        sns.swarmplot(data=summary_df, x='turns', y=metric, color='black', alpha=0.7)
        
        plt.title(title)
        plt.xlabel('Turn Count')
        plt.ylabel(y_label)
        
        plot_path = os.path.join(output_dir, f'boxplot_{metric}.png')
        # 3. Save with higher DPI for a crisper image
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved summary boxplot to: {plot_path}")

def main():
    """Main function to run the aggregate analysis."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    
    # Create a directory for the aggregate analysis output
    aggregate_output_dir = os.path.join(script_dir, "aggregate_analysis")
    os.makedirs(aggregate_output_dir, exist_ok=True)
    
    print("--- Step 1: Loading and Aggregating Data ---")
    master_df = load_and_aggregate_data(results_dir)
    master_df.to_csv(os.path.join(aggregate_output_dir, 'master_metrics_table.csv'), index=False)
    print(f"Master data table saved to: {os.path.join(aggregate_output_dir, 'master_metrics_table.csv')}")
    
    # --- REFINEMENT: Add a sanity check to verify run counts per group ---
    print("\n--- Sanity Check: Runs per Turn Group ---")
    print(master_df.groupby('turns')['run_id'].nunique())
    # --- END REFINEMENT ---
    
    print("\n--- Step 2: Plotting Mean Curves with Variability ---")
    plot_mean_curves(master_df, aggregate_output_dir)
    
    print("\n--- Step 3: Analyzing and Plotting Scalar Summaries ---")
    analyze_and_plot_summaries(master_df, aggregate_output_dir)
    
    print("\n--- Aggregate Analysis Complete ---")
    print(f"All outputs saved in: {aggregate_output_dir}")

if __name__ == "__main__":
    main()