import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- AESTHETIC DEFINITIONS ---
TURN_COLORS = {
    5: "#485690",
    15: "#429590",
    25: "#90C987"
}
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
# --- END DEFINITIONS ---

def plot_mean_curves(df, output_dir):
    """Plots the mean of specified metrics over generations with SD bands."""
    metrics_to_plot = {
        'filled_field_fraction': {'title': 'Mean Filled Field Fraction Over Generations (±1 SD)', 'ylabel': 'Filled Field Fraction'},
        'cosine_similarity': {'title': 'Mean Cosine Similarity Over Generations (±1 SD)', 'ylabel': 'Cosine Similarity'},
        'score_gpt4o': {'title': 'Mean Judge Score (GPT-4o) Over Generations (±1 SD)', 'ylabel': 'Average Judge Score (GPT-4o)'},
        'score_llama3-70b': {'title': 'Mean Judge Score (Llama3-70b) Over Generations (±1 SD)', 'ylabel': 'Average Judge Score (Llama3-70b)'},
    }

    for metric_col, plot_info in metrics_to_plot.items():
        if metric_col not in df.columns:
            print(f"Skipping plot for '{metric_col}' as it's not in the data.")
            continue
        
        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(data=df, x='generation', y=metric_col, hue='turns', palette=TURN_COLORS, errorbar='sd', legend='full')
        
        if 'score' in metric_col:
            ax.set_ylim(0, 5)
        else:
            ax.set_ylim(bottom=0)

        plt.title(plot_info['title'])
        plt.xlabel('Generation')
        plt.ylabel(plot_info['ylabel'])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(sorted(df['generation'].unique()))
        plt.legend(title='Turn Count')
        
        plot_path = os.path.join(output_dir, f'mean_curve_{metric_col}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved mean curve plot to: {plot_path}")

def plot_summary_boxplots(df, output_dir):
    """Creates boxplots to compare summary metrics across turn groups."""
    summary_metrics_to_plot = {
        'final_score_gpt4o': 'Final Generation Judge Score (GPT-4o)',
        'auc_score_gpt4o': 'Overall (AUC) Judge Score (GPT-4o)',
        'final_filled_field_fraction': 'Final Generation Fill Rate',
        'auc_filled_field_fraction': 'Overall (AUC) Fill Rate'
    }

    for metric, title_template in summary_metrics_to_plot.items():
        if metric not in df.columns:
            print(f"Skipping boxplot for '{metric}' as it's not in the data.")
            continue

        title = f"Comparison of {title_template} by Turn Count"
        y_label = title_template

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='turns', y=metric, palette=TURN_COLORS, hue='turns', legend=False)
        sns.swarmplot(data=df, x='turns', y=metric, color='black', alpha=0.7)
        
        plt.title(title)
        plt.xlabel('Turn Count')
        plt.ylabel(y_label)
        
        plot_path = os.path.join(output_dir, f'boxplot_{metric}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved summary boxplot to: {plot_path}")

def main():
    """Main function to generate all plots from pre-aggregated data."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    aggregate_output_dir = os.path.join(script_dir, "aggregate_analysis")
    
    # Define paths to the data files
    master_table_path = os.path.join(aggregate_output_dir, 'master_metrics_table.csv')
    summary_table_path = os.path.join(aggregate_output_dir, 'run_summary_statistics.csv')

    if not os.path.exists(master_table_path) or not os.path.exists(summary_table_path):
        print("Error: Data files not found. Please run '1_create_aggregate_data.py' first.")
        return

    print("--- Loading Pre-aggregated Data ---")
    master_df = pd.read_csv(master_table_path)
    summary_df = pd.read_csv(summary_table_path)

    print("\n--- Generating Mean Curve Plots ---")
    plot_mean_curves(master_df, aggregate_output_dir)
    
    print("\n--- Generating Summary Boxplots ---")
    plot_summary_boxplots(summary_df, aggregate_output_dir)
    
    print("\n--- Plotting Complete ---")

if __name__ == "__main__":
    main()