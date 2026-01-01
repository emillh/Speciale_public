import os
import glob
import pandas as pd
import re

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

    print("--- Creating Master Judgement Dataset ---")
    print("1. Aggregating all runs...")
    print("2. Relabeling (Score 0 + 'omitted') -> 'null'...")
    print("   (This distinguishes 'Empty Field' from 'Content judged as Omitted')")
    
    all_data = []
    run_folders = glob.glob(os.path.join(results_dir, "run_*"))

    total_rows = 0
    relabelled_count = 0

    for run_folder in run_folders:
        turns = get_turn_count(run_folder)
        if turns == -1: continue

        judge_file = os.path.join(run_folder, "metrics", "multi_judge_analysis.csv")
        if not os.path.exists(judge_file): continue

        try:
            df = pd.read_csv(judge_file)
            judges = find_judge_names(df)
            if len(judges) < 2: continue

            # Add Metadata
            df['run_id'] = os.path.basename(run_folder)
            df['turns'] = turns

            total_rows += len(df)

            # --- RELABELING LOGIC ---
            j1, j2 = judges[0], judges[1]
            s1, s2 = f'score_{j1}', f'score_{j2}'
            e1, e2 = f'error_type_{j1}', f'error_type_{j2}'

            # Ensure numeric
            df[s1] = pd.to_numeric(df[s1], errors='coerce').fillna(0)
            df[s2] = pd.to_numeric(df[s2], errors='coerce').fillna(0)

            # Logic for Judge 1: If Score is 0 AND Error is 'omitted', rename to 'null'
            mask_j1 = (df[s1] == 0) & (df[e1] == 'omitted')
            df.loc[mask_j1, e1] = 'null'
            
            # Logic for Judge 2: If Score is 0 AND Error is 'omitted', rename to 'null'
            mask_j2 = (df[s2] == 0) & (df[e2] == 'omitted')
            df.loc[mask_j2, e2] = 'null'

            relabelled_count += mask_j1.sum() + mask_j2.sum()
            
            all_data.append(df)

        except Exception as e:
            print(f"Error reading {os.path.basename(run_folder)}: {e}")

    if not all_data:
        print("No data found.")
        return

    master_df = pd.concat(all_data, ignore_index=True)
    
    output_path = os.path.join(output_dir, "all_judgements_relabelled.csv")
    master_df.to_csv(output_path, index=False)

    print("-" * 30)
    print(f"Total Judgements Processed: {total_rows}")
    print(f"Total Labels Changed to 'null': {relabelled_count}")
    print(f"Saved to: {output_path}")
    print("-" * 30)

if __name__ == "__main__":
    main()