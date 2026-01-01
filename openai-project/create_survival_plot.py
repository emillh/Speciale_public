import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

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

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    output_dir = os.path.join(script_dir, "aggregate_analysis")
    os.makedirs(output_dir, exist_ok=True)

    print("--- Aggregating Survival Analysis Data ---")
    
    all_data = []
    run_folders = glob.glob(os.path.join(results_dir, "run_*"))

    for run_folder in run_folders:
        turns = get_turn_count(run_folder)
        if turns == -1: continue

        survival_file = os.path.join(run_folder, "metrics", "field_survival_analysis.csv")
        if not os.path.exists(survival_file):
            continue

        try:
            # The survival file has comments at the top, so we skip them
            df = pd.read_csv(survival_file, comment='#')
            df['turns'] = turns
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {survival_file}: {e}")

    if not all_data:
        print("No data found.")
        return

    master_df = pd.concat(all_data, ignore_index=True)
    print(f"Successfully aggregated survival data from {len(all_data)} runs.")

    # --- PLOTTING ---
    plt.figure(figsize=(10, 7))
    kmf = KaplanMeierFitter()

    # We iterate through the sorted turn counts to ensure the legend is ordered
    for turns in sorted(master_df['turns'].unique()):
        group_df = master_df[master_df['turns'] == turns]
        
        # Fit the model for this specific group
        kmf.fit(
            durations=group_df['duration'], 
            event_observed=group_df['event'], 
            label=f"{turns} Turns"
        )
        
        # Plot with the specific color
        kmf.plot_survival_function(
            color=TURN_COLORS.get(turns, 'black'),
            ci_show=True # Show Confidence Intervals (shaded area)
        )

    plt.title("Kaplan-Meier Survival Curve of Information Fields")
    plt.xlabel("Generation")
    plt.ylabel("Survival Probability (Fraction of Fields Retained)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.05)
    plt.xlim(0, 9) # Assuming 9 generations max
    
    output_path = os.path.join(output_dir, "aggregate_kaplan_meier_survival.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Survival plot saved to: {output_path}")

if __name__ == "__main__":
    main()