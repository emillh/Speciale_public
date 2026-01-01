import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re

# --- AESTHETIC DEFINITIONS ---
ERROR_COLORS = {
    'faithful': '#1f77b4',          # Blue
    'mostly_faithful': '#aec7e8',   # Light Blue
    'partially_faithful': '#9467bd',# Purple
    'vague': '#6a3d9a',             # Dark Purple
    'overly_general': '#ffbb78',    # Light Orange
    'weakly_related': '#8c564b',    # Brown
    'contradictory': '#98df8a',     # Light Green
    'fabricated': '#2ca02c',        # Green
    'omitted': '#d62728',           # Red
    'no_information': '#ff9896',    # Light Red
    'judgement_error': '#7f7f7f'    # Gray
}

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

    print("--- Aggregating Filtered Error Distributions (Excluding Double Omissions) ---")
    
    all_data = []
    run_folders = glob.glob(os.path.join(results_dir, "run_*"))

    # 1. Load and Filter Data
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

            if not all(col in df.columns for col in [s1, s2, e1, e2]): continue

            # Numeric conversion
            df[s1] = pd.to_numeric(df[s1], errors='coerce').fillna(0)
            df[s2] = pd.to_numeric(df[s2], errors='coerce').fillna(0)

            # --- FILTERING LOGIC ---
            # Identify rows where BOTH judges say it's omitted (Score 0 + Error 'omitted')
            is_omitted_j1 = (df[s1] == 0) & (df[e1] == 'omitted')
            is_omitted_j2 = (df[s2] == 0) & (df[e2] == 'omitted')
            double_omission_mask = is_omitted_j1 & is_omitted_j2
            
            # Keep only the rows that are NOT double omissions
            filtered_df = df[~double_omission_mask].copy()
            # -----------------------

            if filtered_df.empty: continue

            filtered_df['turns'] = turns
            all_data.append(filtered_df)

        except Exception as e:
            print(f"Error reading {os.path.basename(run_folder)}: {e}")

    if not all_data:
        print("No data found.")
        return

    master_df = pd.concat(all_data, ignore_index=True)
    judges = find_judge_names(master_df)
    
    print(f"Aggregated {len(master_df)} valid fields (excluding double omissions).")

    # 2. Generate Plots
    turn_conditions = sorted(master_df['turns'].unique())

    for judge in judges:
        error_col = f'error_type_{judge}'
        if error_col not in master_df.columns: continue

        for turns in turn_conditions:
            print(f"Generating filtered plot for Judge: {judge}, Turns: {turns}...")
            
            subset_df = master_df[master_df['turns'] == turns].copy()
            
            # Group by Generation and Error Type
            error_counts = subset_df.groupby(['generation', error_col]).size().unstack(fill_value=0)
            
            if error_counts.empty:
                print(f"  - No data for {judge} at {turns} turns.")
                continue

            # Normalize to percentages
            error_percentages = error_counts.divide(error_counts.sum(axis=1), axis=0) * 100
            
            # Reorder columns for consistent colors
            existing_errors = [e for e in ERROR_COLORS.keys() if e in error_percentages.columns]
            others = [e for e in error_percentages.columns if e not in ERROR_COLORS]
            ordered_cols = existing_errors + others
            error_percentages = error_percentages[ordered_cols]

            plot_colors = [ERROR_COLORS.get(e, '#333333') for e in ordered_cols]

            # Plot
            plt.figure(figsize=(12, 7))
            ax = error_percentages.plot(
                kind='bar', 
                stacked=True, 
                figsize=(12, 7), 
                color=plot_colors,
                width=0.85
            )

            plt.title(f'Error Type Distribution: {judge} ({turns} Turns)\n(Excluding Double Omissions)')
            plt.xlabel('Generation')
            plt.ylabel('Percentage of Remaining Fields (%)')
            plt.xticks(rotation=0)
            plt.ylim(0, 100)
            
            plt.legend(title='Error Type', bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout()

            # Save with _filtered suffix
            filename = f"error_dist_{judge}_{turns}turns_filtered.png"
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  - Saved to: {output_path}")

if __name__ == "__main__":
    main()