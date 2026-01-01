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
    # Extract names like 'gpt4o' from 'score_gpt4o'
    judges = [re.sub(r'^score_', '', col) for col in score_cols]
    return judges

def calculate_and_save_irr(input_csv_path, output_csv_path):
    """
    Loads a multi_judge_analysis.csv file, calculates inter-rater reliability,
    and saves the results to a new CSV file.
    """
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"  - Error: Input file not found at {input_csv_path}")
        return

    judges = find_judge_names(df)
    
    if len(judges) < 2:
        print(f"  - Warning: Less than two judges found in {input_csv_path}. Cannot calculate reliability.")
        return

    # Use the first two judges found for comparison
    judge1, judge2 = judges[0], judges[1]
    score1_col, score2_col = f'score_{judge1}', f'score_{judge2}'
    
    # Ensure scores are numeric for calculation
    df[score1_col] = pd.to_numeric(df[score1_col], errors='coerce')
    df[score2_col] = pd.to_numeric(df[score2_col], errors='coerce')
    
    # Drop rows where either judge failed to provide a score
    comparison_df = df.dropna(subset=[score1_col, score2_col]).copy()
    
    if comparison_df.empty:
        print(f"  - Warning: No overlapping data to compare judges in {input_csv_path}.")
        return

    # 1. Agreement Rate
    agreement = np.sum(comparison_df[score1_col] == comparison_df[score2_col]) / len(comparison_df)

    # 2. Cohen's Kappa
    kappa = cohen_kappa_score(comparison_df[score1_col], comparison_df[score2_col])
    
    # 3. Score Correlation (Pearson)
    correlation = comparison_df[score1_col].corr(comparison_df[score2_col], method='pearson')

    # Create a DataFrame to save the metrics
    reliability_data = {
        'metric': ["Agreement Rate", "Cohen's Kappa", "Pearson Correlation"],
        'judge_1': [judge1, judge1, judge1],
        'judge_2': [judge2, judge2, judge2],
        'value': [agreement, kappa, correlation]
    }
    reliability_df = pd.DataFrame(reliability_data)
    
    # Save the results
    reliability_df.to_csv(output_csv_path, index=False)
    print(f"  - Success: Saved inter-rater reliability stats to '{output_csv_path}'")

def main():
    """
    Main function to find and process multi_judge_analysis.csv files.
    """
    parser = argparse.ArgumentParser(description="Calculate Inter-Rater Reliability from existing judge analysis files.")
    parser.add_argument(
        '--run_folder', 
        type=str, 
        help="Path to a specific run folder. If not provided, all runs in the 'results' directory will be processed."
    )
    args = parser.parse_args()

    if args.run_folder:
        # Process a single specified folder
        target_folders = [args.run_folder]
    else:
        # Process all folders in the 'results' directory
        target_folders = glob.glob("results/run_*")
        print(f"No specific run folder provided. Processing all {len(target_folders)} runs in 'results/'...")

    if not target_folders:
        print("No run folders found to process.")
        return

    for run_folder in target_folders:
        print(f"\nProcessing run: {run_folder}")
        metrics_dir = os.path.join(run_folder, "metrics")
        input_file = os.path.join(metrics_dir, "multi_judge_analysis.csv")
        output_file = os.path.join(metrics_dir, "inter_rater_reliability.csv")
        
        calculate_and_save_irr(input_file, output_file)

if __name__ == "__main__":
    main()