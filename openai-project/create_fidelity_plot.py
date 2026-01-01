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
        if not log_files:
            return -1
            
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

    print("--- Aggregating Information Fidelity Data (Strict Filter: No Double Omissions) ---")
    
    all_data = []
    run_folders = glob.glob(os.path.join(results_dir, "run_*"))

    for run_folder in run_folders:
        turns = get_turn_count(run_folder)
        if turns == -1: continue

        # We need both fidelity scores and judge analysis
        fidelity_file = os.path.join(run_folder, "metrics", "information_fidelity_analysis.csv")
        judge_file = os.path.join(run_folder, "metrics", "multi_judge_analysis.csv")

        if not os.path.exists(fidelity_file) or not os.path.exists(judge_file):
            continue

        try:
            # 1. Load Data
            df_fidelity = pd.read_csv(fidelity_file)
            df_judge = pd.read_csv(judge_file)

            # 2. Apply Strict Filtering Logic
            judges = find_judge_names(df_judge)
            if len(judges) < 2:
                print(f"Skipping {os.path.basename(run_folder)}: Need 2 judges for filtering logic.")
                continue

            j1, j2 = judges[0], judges[1]
            s1, s2 = f'score_{j1}', f'score_{j2}'
            e1, e2 = f'error_type_{j1}', f'error_type_{j2}'

            # Ensure columns exist
            if not all(col in df_judge.columns for col in [s1, s2, e1, e2]):
                print(f"Skipping {os.path.basename(run_folder)}: Missing required judge columns.")
                continue

            # Convert scores to numeric
            df_judge[s1] = pd.to_numeric(df_judge[s1], errors='coerce').fillna(0)
            df_judge[s2] = pd.to_numeric(df_judge[s2], errors='coerce').fillna(0)

            # --- THE FILTERING LOGIC ---
            # Condition: Judge 1 says (Score 0 AND Omitted)
            is_omitted_j1 = (df_judge[s1] == 0) & (df_judge[e1] == 'omitted')
            # Condition: Judge 2 says (Score 0 AND Omitted)
            is_omitted_j2 = (df_judge[s2] == 0) & (df_judge[e2] == 'omitted')
            
            # Combine: Both must agree it is omitted
            double_omission_mask = is_omitted_j1 & is_omitted_j2
            
            # We keep the rows that are NOT double omissions
            valid_fields_df = df_judge[~double_omission_mask][['generation', 'field_name']]
            # ---------------------------

            if valid_fields_df.empty:
                print(f"Warning: All fields filtered out in {os.path.basename(run_folder)}")
                continue

            # 3. Merge to Filter Fidelity Scores
            # Inner join keeps only the fidelity scores for the valid fields
            df_filtered = pd.merge(df_fidelity, valid_fields_df, on=['generation', 'field_name'], how='inner')
            
            # 4. Aggregate
            run_summary = df_filtered.groupby('generation')[['levenshtein_similarity', 'cosine_similarity']].mean().reset_index()
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
    metrics_to_plot = {
        'cosine_similarity': {
            'title': 'Mean Semantic Fidelity (Cosine) Over Generations (±1 SD)\n(Excluding Double Omissions)',
            'ylabel': 'Cosine Similarity'
        },
        'levenshtein_similarity': {
            'title': 'Mean Syntactic Fidelity (Levenshtein) Over Generations (±1 SD)\n(Excluding Double Omissions)',
            'ylabel': 'Levenshtein Similarity'
        }
    }

    for metric, info in metrics_to_plot.items():
        plt.figure(figsize=(10, 6))
        
        sns.lineplot(
            data=master_df,
            x='generation',
            y=metric,
            hue='turns',
            palette=TURN_COLORS,
            errorbar='sd',
            legend='full'
        )

        plt.title(info['title'])
        plt.xlabel('Generation')
        plt.ylabel(info['ylabel'])
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(sorted(master_df['generation'].unique()))
        plt.legend(title='Turn Count')
        
        filename = f"aggregate_{metric}_filtered.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to: {save_path}")

if __name__ == "__main__":
    main()