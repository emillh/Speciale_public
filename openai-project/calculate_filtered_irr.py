import os
import glob
import pandas as pd
import numpy as np
import argparse
import re
from sklearn.metrics import cohen_kappa_score

def find_judge_names(df):
    """Dynamically finds judge names from the DataFrame columns."""
    score_cols = [col for col in df.columns if col.startswith('score_')]
    judges = [re.sub(r'^score_', '', col) for col in score_cols]
    return judges

def calculate_and_save_filtered_irr(input_csv_path, output_csv_path):
    """
    Loads data, filters out 'double omitted' rows, calculates reliability, and saves.
    """
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"  - Error: Input file not found at {input_csv_path}")
        return

    judges = find_judge_names(df)
    
    if len(judges) < 2:
        print(f"  - Warning: Less than two judges found. Cannot calculate reliability.")
        return

    judge1, judge2 = judges[0], judges[1]
    
    # Define column names
    s1_col, s2_col = f'score_{judge1}', f'score_{judge2}'
    e1_col, e2_col = f'error_type_{judge1}', f'error_type_{judge2}'
    
    # Ensure scores are numeric
    df[s1_col] = pd.to_numeric(df[s1_col], errors='coerce').fillna(0)
    df[s2_col] = pd.to_numeric(df[s2_col], errors='coerce').fillna(0)
    
    # --- FILTERING LOGIC ---
    # Check if error columns exist
    if e1_col in df.columns and e2_col in df.columns:
        # Condition: Both scores are 0 AND Both error types are 'omitted'
        is_omitted_j1 = (df[s1_col] == 0) & (df[e1_col] == 'omitted')
        is_omitted_j2 = (df[s2_col] == 0) & (df[e2_col] == 'omitted')
        
        # Identify rows where BOTH judges omitted the field
        double_omission_mask = is_omitted_j1 & is_omitted_j2
        
        # Keep rows that are NOT double omissions
        filtered_df = df[~double_omission_mask].copy()
        
        dropped_count = double_omission_mask.sum()
        print(f"  - Filtering: Dropped {dropped_count} rows where both judges omitted the field.")
    else:
        print("  - Warning: Error type columns not found. Using unfiltered data.")
        filtered_df = df.copy()
    # --- END FILTERING LOGIC ---

    if filtered_df.empty:
        print(f"  - Warning: No data left after filtering in {input_csv_path}.")
        return

    # 1. Agreement Rate
    agreement = np.sum(filtered_df[s1_col] == filtered_df[s2_col]) / len(filtered_df)

    # 2. Cohen's Kappa
    # Kappa can fail if there is only one class present (e.g., all scores are 5)
    try:
        kappa = cohen_kappa_score(filtered_df[s1_col], filtered_df[s2_col])
    except ValueError:
        kappa = np.nan 

    # 3. Score Correlation (Pearson)
    # Correlation requires variance; if all scores are identical, it returns NaN
    correlation = filtered_df[s1_col].corr(filtered_df[s2_col], method='pearson')

    # Create DataFrame
    reliability_data = {
        'metric': ["Agreement Rate", "Cohen's Kappa", "Pearson Correlation"],
        'judge_1': [judge1, judge1, judge1],
        'judge_2': [judge2, judge2, judge2],
        'value': [agreement, kappa, correlation]
    }
    reliability_df = pd.DataFrame(reliability_data)
    
    # Save
    reliability_df.to_csv(output_csv_path, index=False)
    print(f"  - Success: Saved filtered stats to '{output_csv_path}'")

def main():
    parser = argparse.ArgumentParser(description="Calculate Filtered Inter-Rater Reliability.")
    parser.add_argument('--run_folder', type=str, help="Path to a specific run folder.")
    args = parser.parse_args()

    # Robust path handling
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.run_folder:
        if not os.path.isabs(args.run_folder):
            target_folders = [os.path.join(script_dir, args.run_folder)]
        else:
            target_folders = [args.run_folder]
    else:
        search_path = os.path.join(script_dir, "results", "run_*")
        target_folders = glob.glob(search_path)
        print(f"Processing all {len(target_folders)} runs in 'results/'...")

    if not target_folders:
        print("No run folders found.")
        return

    for run_folder in target_folders:
        print(f"\nProcessing run: {os.path.basename(run_folder)}")
        metrics_dir = os.path.join(run_folder, "metrics")
        input_file = os.path.join(metrics_dir, "multi_judge_analysis.csv")
        
        # Save to a NEW filename to distinguish from the raw version
        output_file = os.path.join(metrics_dir, "inter_rater_reliability_filtered.csv")
        
        calculate_and_save_filtered_irr(input_file, output_file)

if __name__ == "__main__":
    main()