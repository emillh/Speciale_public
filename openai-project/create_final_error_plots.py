import os
import pandas as pd
import matplotlib.pyplot as plt

# --- AESTHETIC DEFINITIONS ---
ERROR_COLORS = {
    'faithful': '#1f77b4',          # Blue
    'mostly_faithful': '#aec7e8',   # Light Blue
    'partially_faithful': '#9467bd',# Purple
    'vague': '#6a3d9a',             # Dark Purple
    'overly_general': '#ffbb78',    # Light Orange
    'weakly_related': '#8c564b',    # Brown
    'weakly related': '#8c564b',    # Brown (Catching variations)
    'contradictory': '#98df8a',     # Light Green
    'fabricated': '#2ca02c',        # Green
    
    'null': '#d62728',              # Red (The empty fields)
    'omitted': '#b01e1e',           # Darker Red (Content exists but judged as omitted)
    'no_information': '#ff9896',    # Light Red (Model says "I don't know")
    
    'judgement_error': '#7f7f7f'    # Gray
}

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "aggregate_analysis", "all_judgements_relabelled.csv")
    output_dir = os.path.join(script_dir, "aggregate_analysis")
    
    if not os.path.exists(input_file):
        print(f"Error: File not found at {input_file}")
        return

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # --- CRITICAL FIX ---
    # Pandas reads the string "null" as NaN (Missing Data).
    # We must convert these NaNs back to the string "null" for the error columns.
    error_cols = [c for c in df.columns if c.startswith('error_type_')]
    for col in error_cols:
        df[col] = df[col].fillna('null')
    # --------------------
    
    judges = ['gpt4o', 'llama3-70b']
    turn_counts = sorted(df['turns'].unique())
    
    print("--- Generating Error Distribution Plots ---")

    for judge in judges:
        error_col = f'error_type_{judge}'
        if error_col not in df.columns:
            print(f"Warning: Column {error_col} not found. Skipping.")
            continue

        for turns in turn_counts:
            subset_df = df[df['turns'] == turns].copy()
            if subset_df.empty: continue

            # --- VARIATION 1: WITH NULLS (Total Distribution) ---
            # We plot everything. 'null' will show up as Red.
            plot_distribution(
                subset_df, 
                judge, 
                turns, 
                error_col, 
                include_nulls=True, 
                output_dir=output_dir
            )

            # --- VARIATION 2: WITHOUT NULLS (Mutation Distribution) ---
            # We ONLY filter out 'null'. We keep 'omitted' and 'no_information'.
            plot_distribution(
                subset_df, 
                judge, 
                turns, 
                error_col, 
                include_nulls=False, 
                output_dir=output_dir
            )

def plot_distribution(df, judge, turns, error_col, include_nulls, output_dir):
    """Helper function to generate and save a single stacked bar chart."""
    
    # --- FILTERING LOGIC ---
    if not include_nulls:
        # STRICT FILTER: Only remove 'null'. Keep everything else.
        df = df[df[error_col] != 'null']
        
        suffix = "no_nulls"
        title_suffix = "(Excluding Nulls)"
        y_label = "Percentage of Non-Null Fields (%)"
    else:
        suffix = "with_nulls"
        title_suffix = "(Including Nulls)"
        y_label = "Percentage of All Fields (%)"

    if df.empty:
        print(f"  Skipping {judge} {turns} turns ({suffix}) - dataset empty after filtering.")
        return

    # Group by Generation and Error Type
    counts = df.groupby(['generation', error_col]).size().unstack(fill_value=0)
    
    # Normalize to percentages (0-100%)
    percentages = counts.divide(counts.sum(axis=1), axis=0) * 100
    
    # Reorder columns to match the color map
    present_cols = [c for c in ERROR_COLORS.keys() if c in percentages.columns]
    extra_cols = [c for c in percentages.columns if c not in present_cols]
    ordered_cols = present_cols + extra_cols
    
    percentages = percentages[ordered_cols]
    colors = [ERROR_COLORS.get(c, '#333333') for c in ordered_cols]

    # Plotting
    plt.figure(figsize=(12, 7))
    ax = percentages.plot(
        kind='bar', 
        stacked=True, 
        color=colors, 
        width=0.85, 
        figsize=(12, 7)
    )
    
    plt.title(f"Error Distribution: {judge} ({turns} Turns)\n{title_suffix}")
    plt.xlabel("Generation")
    plt.ylabel(y_label)
    plt.ylim(0, 100)
    plt.xticks(rotation=0)
    
    plt.legend(title='Error Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    
    filename = f"dist_{judge}_{turns}turns_{suffix}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

if __name__ == "__main__":
    main()