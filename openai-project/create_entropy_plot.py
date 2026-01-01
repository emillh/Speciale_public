import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- AESTHETIC DEFINITIONS (from your other scripts) ---
TURN_COLORS = {
    5: "#485690",
    15: "#429590",
    25: "#90C987"
}
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
# --- END DEFINITIONS ---

def get_turn_count(run_folder_path):
    """Reads the gen_0 log file to determine the number of turns in a run."""
    try:
        log_file = glob.glob(os.path.join(run_folder_path, "logs", "*_gen_0.txt"))[0]
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            return content.count("SalesAgent:")
    except (IndexError, FileNotFoundError):
        return -1

def main():
    """
    Finds all Shannon Entropy data, aggregates it, and creates a single plot.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    output_dir = os.path.join(script_dir, "aggregate_analysis")
    os.makedirs(output_dir, exist_ok=True)

    print("--- Finding and Aggregating Shannon Entropy Data ---")
    
    all_entropy_data = []
    metric_files = glob.glob(os.path.join(results_dir, "run_*", "metrics", "information_decay_analysis.csv"))

    for metric_file in metric_files:
        run_path = os.path.dirname(os.path.dirname(metric_file))
        run_id = os.path.basename(run_path)
        
        turns = get_turn_count(run_path)
        if turns == -1:
            print(f"Warning: Could not determine turn count for {run_id}. Skipping.")
            continue

        try:
            df = pd.read_csv(metric_file)
            # Filter out the non-generational row
            df = df[df['Generation'] != 'Ground Truth'].copy()
            
            # Standardize column names
            df.rename(columns={'Generation': 'generation', 'Shannon_Entropy': 'shannon_entropy'}, inplace=True)
            
            # Ensure correct data types
            df['generation'] = pd.to_numeric(df['generation'])
            df['shannon_entropy'] = pd.to_numeric(df['shannon_entropy'])
            
            df['turns'] = turns
            all_entropy_data.append(df[['generation', 'shannon_entropy', 'turns']])
        except (FileNotFoundError, KeyError) as e:
            print(f"Warning: Could not process {metric_file}. Skipping. Error: {e}")
            continue
            
    if not all_entropy_data:
        print("Error: No Shannon Entropy data found. Exiting.")
        return

    master_df = pd.concat(all_entropy_data, ignore_index=True)
    print(f"Successfully aggregated entropy data from {len(metric_files)} runs.")

    print("\n--- Generating Shannon Entropy Plot ---")
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=master_df,
        x='generation',
        y='shannon_entropy',
        hue='turns',
        palette=TURN_COLORS,
        errorbar='sd',
        legend='full'
    )

    plt.title('Mean Shannon Entropy Over Generations (±1 SD)')
    plt.xlabel('Generation')
    plt.ylabel('Shannon Entropy (bits)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(sorted(master_df['generation'].unique()))
    plt.legend(title='Turn Count')
    
    plot_path = os.path.join(output_dir, 'mean_curve_shannon_entropy.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved successfully to: {plot_path}")

if __name__ == "__main__":
    main()