import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# --- AESTHETIC DEFINITIONS ---
TURN_COLORS = {
    5: "#485690",
    15: "#429590",
    25: "#90C987"
}
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
# -----------------------------

def get_turn_count(run_folder_path):
    """Reads the gen_0 log file to determine the number of turns in a run."""
    try:
        log_files = glob.glob(os.path.join(run_folder_path, "logs", "*_gen_0.txt"))
        if not log_files: return -1
        with open(log_files[0], 'r', encoding='utf-8') as f:
            content = f.read()
            count = content.count("SalesAgent:")
            if count <= 10: return 5
            elif count <= 20: return 15
            else: return 25
    except Exception:
        return -1

def find_judge_names(df):
    """Dynamically finds judge names from the DataFrame columns."""
    score_cols = [col for col in df.columns if col.startswith('score_')]
    judges = [re.sub(r'^score_', '', col) for col in score_cols]
    return judges

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    output_dir = os.path.join(script_dir, "aggregate_analysis")
    os.makedirs(output_dir, exist_ok=True)

    print("--- Aggregating Filtered Judge Scores (Excluding Double Omissions) ---")
    
    all_data = []
    run_folders = glob.glob(os.path.join(results_dir, "run_*"))

    for run_folder in run_folders:
        turns = get_turn_count(run_folder)
        if turns == -1: continue

        judge_file = os.path.join(run_folder, "metrics", "multi_judge_analysis.csv")
        if not os.path.exists(judge_file): continue

        try:
            df = pd.read_csv(judge_file)
            judges = find_judge_names(df)
            if len(judges) < 2: continue

            j1, j2 = judges[0], judges[1]
            s1, s2 = f'score_{j1}', f'score_{j2}'
            e1, e2 = f'error_type_{j1}', f'error_type_{j2}'

            # Ensure columns exist
            if not all(col in df.columns for col in [s1, s2, e1, e2]): continue

            # Convert scores to numeric
            df[s1] = pd.to_numeric(df[s1], errors='coerce').fillna(0)
            df[s2] = pd.to_numeric(df[s2], errors='coerce').fillna(0)

            # --- FILTERING LOGIC (Same as IRR script) ---
            # Condition: Both judges agree it is omitted (Score 0 AND Error 'omitted')
            is_omitted_j1 = (df[s1] == 0) & (df[e1] == 'omitted')
            is_omitted_j2 = (df[s2] == 0) & (df[e2] == 'omitted')
            double_omission_mask = is_omitted_j1 & is_omitted_j2
            
            # Keep only valid rows
            filtered_df = df[~double_omission_mask].copy()
            # --------------------------------------------

            if filtered_df.empty: continue

            # Calculate mean score per generation for this run
            # We group by generation and take the mean of the scores
            run_summary = filtered_df.groupby('generation')[[s1, s2]].mean().reset_index()
            run_summary['turns'] = turns
            
            all_data.append(run_summary)

        except Exception as e:
            print(f"Error processing {os.path.basename(run_folder)}: {e}")

    if not all_data:
        print("No data found.")
        return

    master_df = pd.concat(all_data, ignore_index=True)
    print(f"Successfully aggregated filtered data from {len(all_data)} runs.")

    # --- PLOTTING ---
    # We need to know the judge names again to plot them
    # We assume the first entry has the correct columns
    sample_cols = master_df.columns
    judge_cols = [c for c in sample_cols if c.startswith('score_')]

    for score_col in judge_cols:
        judge_name = score_col.replace('score_', '')
        
        plt.figure(figsize=(10, 6))
        
        sns.lineplot(
            data=master_df,
            x='generation',
            y=score_col,
            hue='turns',
            palette=TURN_COLORS,
            errorbar='sd',
            legend='full'
        )

        plt.title(f'Mean Judge Score ({judge_name}) Over Generations (±1 SD)\n(Excluding Double Omissions)')
        plt.xlabel('Generation')
        plt.ylabel(f'Average Judge Score ({judge_name})')
        plt.ylim(0, 5.2) # Scores are 0-5
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(sorted(master_df['generation'].unique()))
        plt.legend(title='Turn Count')
        
        filename = f"mean_curve_score_{judge_name}_filtered.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to: {save_path}")

if __name__ == "__main__":
    main()