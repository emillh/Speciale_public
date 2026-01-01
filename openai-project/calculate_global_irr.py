import os
import glob
import pandas as pd
import numpy as np
import re
from sklearn.metrics import cohen_kappa_score

# --- CONFIGURATION ---
TURN_COLORS = {5: "#485690", 15: "#429590", 25: "#90C987"}
# ---------------------

def get_turn_count(run_folder_path):
    """Reads the gen_0 log file to determine the number of turns in a run."""
    try:
        log_file = glob.glob(os.path.join(run_folder_path, "logs", "*_gen_0.txt"))[0]
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            return content.count("SalesAgent:")
    except (IndexError, FileNotFoundError):
        return -1

def find_judge_names(df):
    score_cols = [col for col in df.columns if col.startswith('score_')]
    judges = [re.sub(r'^score_', '', col) for col in score_cols]
    return judges

def calculate_metrics(df, judge1, judge2):
    """Calculates Agreement, Kappa, and Correlation for a given DataFrame."""
    s1_col, s2_col = f'score_{judge1}', f'score_{judge2}'
    
    # 1. Agreement Rate
    agreement = np.sum(df[s1_col] == df[s2_col]) / len(df)

    # 2. Cohen's Kappa
    try:
        kappa = cohen_kappa_score(df[s1_col], df[s2_col])
    except ValueError:
        kappa = np.nan

    # 3. Pearson Correlation
    correlation = df[s1_col].corr(df[s2_col], method='pearson')
    
    return agreement, kappa, correlation

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    output_dir = os.path.join(script_dir, "aggregate_analysis")
    os.makedirs(output_dir, exist_ok=True)

    print("--- Loading and Pooling All Judge Data ---")
    
    all_data = []
    metric_files = glob.glob(os.path.join(results_dir, "run_*", "metrics", "multi_judge_analysis.csv"))

    for metric_file in metric_files:
        run_path = os.path.dirname(os.path.dirname(metric_file))
        turns = get_turn_count(run_path)
        if turns == -1: continue

        try:
            df = pd.read_csv(metric_file)
            df['turns'] = turns
            all_data.append(df)
        except Exception:
            continue

    if not all_data:
        print("No data found.")
        return

    # Combine into one massive DataFrame
    master_df = pd.concat(all_data, ignore_index=True)
    
    # --- PREPARE DATA (Filter Omitted) ---
    judges = find_judge_names(master_df)
    if len(judges) < 2: return
    j1, j2 = judges[0], judges[1]
    s1, s2 = f'score_{j1}', f'score_{j2}'
    e1, e2 = f'error_type_{j1}', f'error_type_{j2}'

    # Numeric conversion
    master_df[s1] = pd.to_numeric(master_df[s1], errors='coerce').fillna(0)
    master_df[s2] = pd.to_numeric(master_df[s2], errors='coerce').fillna(0)

    # Filter Logic (Strict Double Omission)
    if e1 in master_df.columns and e2 in master_df.columns:
        is_omitted_j1 = (master_df[s1] == 0) & (master_df[e1] == 'omitted')
        is_omitted_j2 = (master_df[s2] == 0) & (master_df[e2] == 'omitted')
        double_omission_mask = is_omitted_j1 & is_omitted_j2
        
        filtered_df = master_df[~double_omission_mask].copy()
        print(f"Total Data Points: {len(master_df)}")
        print(f"Filtered Data Points (excluding double omissions): {len(filtered_df)}")
    else:
        filtered_df = master_df.copy()

    # --- CALCULATE METRICS ---
    results = []

    # 1. GLOBAL (All data combined)
    agg, kap, corr = calculate_metrics(filtered_df, j1, j2)
    results.append({
        'Group': 'Global (All Runs)', 
        'N_Samples': len(filtered_df),
        'Agreement': agg, 
        'Kappa': kap, 
        'Correlation': corr
    })

    # 2. BY TURN COUNT
    for turns in sorted(filtered_df['turns'].unique()):
        group_df = filtered_df[filtered_df['turns'] == turns]
        agg, kap, corr = calculate_metrics(group_df, j1, j2)
        results.append({
            'Group': f'{turns} Turns', 
            'N_Samples': len(group_df),
            'Agreement': agg, 
            'Kappa': kap, 
            'Correlation': corr
        })

    # --- OUTPUT ---
    results_df = pd.DataFrame(results)
    
    # Print to console for immediate viewing
    print("\n--- POOLED INTER-RATER RELIABILITY RESULTS ---")
    print(results_df.to_string(index=False))
    
    # Save to CSV
    output_path = os.path.join(output_dir, "pooled_inter_rater_reliability.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved pooled results to: {output_path}")

if __name__ == "__main__":
    main()