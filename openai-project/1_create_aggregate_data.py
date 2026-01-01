import os
import glob
import pandas as pd

def get_turn_count(run_folder_path):
    """Reads the gen_0 log file to determine the number of turns in a run."""
    try:
        log_file = glob.glob(os.path.join(run_folder_path, "logs", "*_gen_0.txt"))[0]
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            return content.count("SalesAgent:")
    except (IndexError, FileNotFoundError):
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
            decay_df = pd.read_csv(os.path.join(metrics_path, "information_decay_analysis.csv"))
            fidelity_df = pd.read_csv(os.path.join(metrics_path, "information_fidelity_analysis.csv"))
            judge_df = pd.read_csv(os.path.join(metrics_path, "multi_judge_analysis.csv"))

            if 'Generation' in decay_df.columns and 'generation' not in decay_df.columns:
                decay_df.rename(columns={'Generation': 'generation'}, inplace=True)
            rename_map = {'Field_Fill_Rate': 'filled_field_fraction'}
            decay_df.rename(columns=rename_map, inplace=True)
            decay_df['generation'] = pd.to_numeric(decay_df['generation'], errors='coerce')
            decay_df.dropna(subset=['generation'], inplace=True)
            decay_df['generation'] = decay_df['generation'].astype(int)
            decay_df = decay_df[['generation', 'filled_field_fraction']]

            fidelity_agg = fidelity_df.groupby('generation')[['levenshtein_similarity', 'cosine_similarity']].mean().reset_index()
            judge_agg = judge_df.groupby('generation')[['score_gpt4o', 'score_llama3-70b']].mean().reset_index()

            run_df = pd.merge(decay_df, fidelity_agg, on="generation", how="left")
            run_df = pd.merge(run_df, judge_agg, on="generation", how="left")

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

def create_run_summaries(df, output_dir):
    """
    Calculates scalar summaries (final value, AUC) for each run and saves them.
    """
    summary_data = []
    for run_id, group in df.groupby('run_id'):
        turns = group['turns'].iloc[0]
        final_gen_row = group.loc[group['generation'].idxmax()]
        
        metrics_for_auc = [
            'filled_field_fraction', 'cosine_similarity', 
            'score_gpt4o', 'score_llama3-70b'
        ]
        existing_metrics = [m for m in metrics_for_auc if m in group.columns]
        auc_metrics = group[existing_metrics].mean()
        
        run_summary = {'run_id': run_id, 'turns': turns}
        for metric in existing_metrics:
            run_summary[f'final_{metric}'] = final_gen_row[metric]
            run_summary[f'auc_{metric}'] = auc_metrics[metric]
        summary_data.append(run_summary)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df['turns'] = pd.to_numeric(summary_df['turns'])
    
    path = os.path.join(output_dir, 'run_summary_statistics.csv')
    summary_df.to_csv(path, index=False)
    print(f"Saved run summary statistics to: {path}")

def main():
    """Main function to run the data aggregation."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    
    aggregate_output_dir = os.path.join(script_dir, "aggregate_analysis")
    os.makedirs(aggregate_output_dir, exist_ok=True)
    
    print("--- Step 1: Loading and Creating Master Data Table ---")
    master_df = load_and_aggregate_data(results_dir)
    master_path = os.path.join(aggregate_output_dir, 'master_metrics_table.csv')
    master_df.to_csv(master_path, index=False)
    print(f"Master data table saved to: {master_path}")
    
    print("\n--- Sanity Check: Runs per Turn Group ---")
    print(master_df.groupby('turns')['run_id'].nunique())
    
    print("\n--- Step 2: Creating Run Summary Statistics ---")
    create_run_summaries(master_df, aggregate_output_dir)
    
    print("\n--- Data Processing Complete ---")

if __name__ == "__main__":
    main()