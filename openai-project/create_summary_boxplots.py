import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- AESTHETIC DEFINITIONS ---
TURN_COLORS = {
    5: "#485690",
    15: "#429590",
    25: "#90C987"
}
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
# -----------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    aggregate_dir = os.path.join(script_dir, "aggregate_analysis")
    input_file = os.path.join(aggregate_dir, "run_summary_statistics.csv")

    if not os.path.exists(input_file):
        print(f"Error: Could not find input file at {input_file}")
        return

    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)

    # Define the metrics we want to plot
    metrics_to_plot = {
        'auc_score_gpt4o': 'Overall (AUC) Judge Score (GPT-4o)',
        'final_score_gpt4o': 'Final Generation Judge Score (GPT-4o)',
        'auc_score_llama3-70b': 'Overall (AUC) Judge Score (Llama3-70b)',
        'final_score_llama3-70b': 'Final Generation Judge Score (Llama3-70b)',
        'auc_filled_field_fraction': 'Overall (AUC) Fill Rate',
        'final_filled_field_fraction': 'Final Generation Fill Rate'
    }

    print("--- Generating Summary Boxplots ---")

    for metric, title_label in metrics_to_plot.items():
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in CSV. Skipping.")
            continue

        plt.figure(figsize=(8, 6))
        
        # Create Boxplot
        sns.boxplot(
            data=df, 
            x='turns', 
            y=metric, 
            palette=TURN_COLORS, 
            hue='turns', 
            legend=False
        )
        
        # Overlay Swarmplot to show individual data points
        sns.swarmplot(
            data=df, 
            x='turns', 
            y=metric, 
            color='black', 
            alpha=0.7,
            size=6
        )

        plt.title(f"Comparison of {title_label}\nby Turn Count")
        plt.xlabel('Turn Count')
        plt.ylabel(title_label)
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)

        # Set Y-limits based on metric type
        if 'score' in metric:
            plt.ylim(0, 5.2) # Judge scores are 0-5
        else:
            plt.ylim(0, 1.05) # Fractions are 0-1

        # Save the plot
        safe_name = metric.replace('_', '-')
        output_path = os.path.join(aggregate_dir, f"boxplot_{safe_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()