import os
import glob
import json
import re
import pandas as pd
import shutil
from datetime import datetime
from src.report_schema import LeadReport
import matplotlib.pyplot as plt
import numpy as np
import argparse

# --- New Imports for Advanced Metrics ---
from lifelines import KaplanMeierFitter
import Levenshtein
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy

# --- Helper Functions (Parsing & Calculation) ---
def parse_report_from_log(file_path):
    """Reads a log file and extracts the final JSON report string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            report_marker = "--- Final Extracted Report (Input for Next Generation's Persona) ---"
            report_start_index = content.find(report_marker)
            if report_start_index == -1:
                print(f"Warning: Report marker not found in {file_path}")
                return None
            json_string = content[report_start_index + len(report_marker):].strip()
            return json.loads(json_string)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def calculate_shannon_entropy(text):
    """Calculates the Shannon entropy for a given string."""
    if not text:
        return 0
    counts = pd.Series(list(text)).value_counts()
    probabilities = counts / len(text)
    return entropy(probabilities, base=2)

# --- Analysis Functions ---
def analyze_decay_and_survival(all_reports, num_generations):
    """Calculates Field Fill Rate, Info Volume, Shannon Entropy, and data for Survival Analysis."""
    # Correctly access model_fields from the class, not the instance
    total_fields = len(LeadReport.model_fields)
    
    field_fill_rate = [sum(1 for v in r.values() if v is not None) / total_fields for r in all_reports]
    info_volume = [sum(len(str(v)) for v in r.values() if v is not None) for r in all_reports]
    report_texts = ["".join(str(v) for v in r.values() if v is not None) for r in all_reports]
    shannon_entropy = [calculate_shannon_entropy(text) for text in report_texts]

    decay_df = pd.DataFrame({
        'Generation': ['Ground Truth'] + list(range(num_generations)),
        'Field_Fill_Rate': field_fill_rate,
        'Information_Volume_Chars': info_volume,
        'Shannon_Entropy': shannon_entropy,
    })

    survival_data = {}
    # Correctly access model_fields from the class
    for field_name in LeadReport.model_fields:
        first_null_gen = -1
        for i, report in enumerate(all_reports[1:]):
            if report.get(field_name) is None:
                first_null_gen = i
                break
        duration = first_null_gen if first_null_gen != -1 else num_generations
        event_observed = 1 if first_null_gen != -1 else 0
        survival_data[field_name] = {'duration': duration, 'event': event_observed}

    survival_df = pd.DataFrame.from_dict(survival_data, orient='index').reset_index().rename(columns={'index': 'field_name'})
    return decay_df, survival_df

def analyze_fidelity(all_reports, ground_truth_dict):
    """Calculates Levenshtein and Cosine Similarity against ground truth."""
    print("Initializing sentence embedding model (may download on first run)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    fidelity_records = []
    field_names = list(ground_truth_dict.keys())

    gt_texts = [str(ground_truth_dict.get(f, '')) for f in field_names]
    gt_embeddings = model.encode(gt_texts)

    for gen_idx, report in enumerate(all_reports):
        if gen_idx == 0: continue

        gen_texts = [str(report.get(f, '')) for f in field_names]
        gen_embeddings = model.encode(gen_texts)

        for field_idx, field_name in enumerate(field_names):
            gt_text, gen_text = gt_texts[field_idx], gen_texts[field_idx]
            
            max_len = max(len(gt_text), len(gen_text))
            lev_sim = 1 - (Levenshtein.distance(gt_text, gen_text) / max_len) if max_len > 0 else 1.0

            gt_emb, gen_emb = gt_embeddings[field_idx].reshape(1, -1), gen_embeddings[field_idx].reshape(1, -1)
            cos_sim = cosine_similarity(gt_emb, gen_emb)[0, 0]

            fidelity_records.append({
                'generation': gen_idx - 1,
                'field_name': field_name,
                'levenshtein_similarity': lev_sim,
                'cosine_similarity': float(cos_sim)
            })
            
    return pd.DataFrame(fidelity_records)

# --- Plotting Functions ---
def plot_and_save_survival_curve(survival_df, output_path):
    """Generates and saves a Kaplan-Meier survival curve plot."""
    kmf = KaplanMeierFitter()
    kmf.fit(durations=survival_df['duration'], event_observed=survival_df['event'])
    plt.figure(figsize=(12, 7))
    kmf.plot(label="Kaplan-Meier Estimate of Field Survival")
    plt.title("Survival Curve of Information Fields")
    plt.xlabel("Generation")
    plt.ylabel("Survival Probability (Fraction of Fields Retained)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1)
    plt.xticks(range(int(max(survival_df['duration'])) + 2))
    plt.savefig(output_path)
    plt.close()
    print(f"Survival curve plot saved to '{output_path}'")

def plot_and_save_fidelity_curves(fidelity_df, output_path):
    """Plots the average Levenshtein and Cosine similarity over generations."""
    avg_fidelity = fidelity_df.groupby('generation')[['levenshtein_similarity', 'cosine_similarity']].mean().reset_index()
    plt.figure(figsize=(12, 7))
    plt.plot(avg_fidelity['generation'], avg_fidelity['levenshtein_similarity'], marker='o', label='Avg. Levenshtein Similarity (Syntactic)')
    plt.plot(avg_fidelity['generation'], avg_fidelity['cosine_similarity'], marker='s', linestyle='--', label='Avg. Cosine Similarity (Semantic)')
    plt.title('Information Fidelity Decay Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Average Similarity to Ground Truth')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(0, 1)
    plt.xticks(range(int(max(avg_fidelity['generation'])) + 1))
    plt.savefig(output_path)
    plt.close()
    print(f"Fidelity decay plot saved to '{output_path}'")

def plot_and_save_entropy_curve(decay_df, output_path):
    """Plots the Shannon Entropy over generations."""
    plt.figure(figsize=(12, 7))
    # Correctly slice the 'Generation' column to only include numbers for plotting
    numeric_generations = decay_df['Generation'][1:]
    entropy_values = decay_df['Shannon_Entropy'][1:]
    
    plt.plot(numeric_generations, entropy_values, marker='d', linestyle='-', color='green', label='Shannon Entropy')
    plt.title('Information Richness (Shannon Entropy) Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Shannon Entropy (bits)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    # Correctly calculate the max tick from the numeric part of the column
    if not numeric_generations.empty:
        plt.xticks(range(int(max(numeric_generations)) + 1))
    plt.savefig(output_path)
    plt.close()
    print(f"Entropy decay plot saved to '{output_path}'")

# --- Main Execution ---
def main():
    """Main function to run all analyses, archive logs, and save results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, default=None, help="Path to an existing run folder (with logs/). If omitted, uses conversation_logs/ and creates a new run.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    BASE_RESULTS_DIR = os.path.join(script_dir, "results")

    if args.run_folder:
        # Analyze an existing run in place
        run_dir = args.run_folder
        logs_archive_dir = os.path.join(run_dir, "logs")
        metrics_dir = os.path.join(run_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        log_files = sorted(
            glob.glob(os.path.join(logs_archive_dir, "telephone_game_*.txt")),
            key=lambda x: int(re.search(r'gen_(\d+)\.txt$', x).group(1))
        )
        print(f"Analyzing existing run at: {run_dir}")
    else:
        # Original behavior: consume conversation_logs and create a new run
        SOURCE_LOG_DIR = os.path.join(script_dir, "conversation_logs")
        log_files = glob.glob(os.path.join(SOURCE_LOG_DIR, "telephone_game_*.txt"))
        if not log_files:
            print(f"No telephone game logs found in '{SOURCE_LOG_DIR}'. Nothing to process.")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(BASE_RESULTS_DIR, f"run_{timestamp}")
        logs_archive_dir = os.path.join(run_dir, "logs")
        metrics_dir = os.path.join(run_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        shutil.move(SOURCE_LOG_DIR, logs_archive_dir)
        os.makedirs(SOURCE_LOG_DIR, exist_ok=True)
        log_files = sorted(
            glob.glob(os.path.join(logs_archive_dir, "telephone_game_*.txt")),
            key=lambda x: int(re.search(r'gen_(\d+)\.txt$', x).group(1))
        )
        print(f"Created analysis directory: {run_dir}")
        print(f"Archived log files to: {logs_archive_dir}")

    ground_truth_persona = LeadReport(
        company_name="Innovate Inc.", contact_person_name="Alex", contact_person_role="Project Manager",
        contact_person_email="alex@innovate.com", contact_person_phone="555-0102", is_decision_maker=True,
        industry="Advanced Manufacturing", address="123 Innovation Drive, Tech Park, CA 90210",
        purpose_of_request="We need to increase our production speed and reduce repetitive strain injuries on our assembly line.",
        lead_source="Heard about you from a colleague who attended the A3 Automate trade show.",
        worked_with_ur_before="No, this is our first time exploring a partnership.",
        in_house_automation_resources_description="We have a small maintenance team for our existing conveyor systems, but no dedicated robotics or automation engineers.",
        in_house_team_skill_level="Beginner. Our team is mechanically skilled but has no experience with programming or maintaining collaborative robots.",
        installation_responsibility="We would prefer a turnkey solution where the provider handles the full installation and initial setup.",
        preferred_installation_timeline="Within the next 6 months.",
        deployment_process_preference="A phased deployment would be ideal, starting with one station to prove the concept before rolling it out to the entire line.",
        application_description="The primary application is pick-and-place for our 'widget assembly line'. It involves moving small electronic components from a tray to a circuit board.",
        application_criticality="High. This assembly line is a bottleneck for our entire production process.",
        application_type_and_sub_type="Pick and Place, specifically PCB (Printed Circuit Board) assembly.",
        part_variety_description="There are about 5 different types of components, but they are all similar in size and shape.",
        part_dimensions_description="The components are small, roughly 10mm x 10mm x 5mm.",
        part_weight_description="Extremely light, less than 5 grams each.",
        desired_throughput="We need to achieve at least 20 picks per minute to meet our production targets.",
        budget_for_project="Our initial budget is around $100,000, but it's flexible for a solution that shows a clear ROI.",
        project_completion_timeline_ideal="Ideally, we'd like the first station running within 3-4 months.",
        project_timeline_constraints="We have a hard deadline of 8 months from now to have the full line automated for our next product launch.",
        key_decision_makers_description="I am the primary decision-maker for the technical evaluation, but the final budget approval will come from our Director of Operations, Ms. Jane Smith.",
        company_history_summary="Innovate Inc. was founded 15 years ago with a focus on creating high-end consumer electronics.",
        key_stakeholders_summary="Besides myself and the Director of Operations, the lead assembly line operator will also be a key stakeholder in the user acceptance testing.",
        company_differentiation_summary="Our key differentiator is our commitment to high-quality, durable product design, which requires very precise manufacturing processes.",
        primary_products_services_summary="We design and manufacture premium wireless audio equipment, including headphones and speakers.",
        innovation_approach_summary="We follow a continuous improvement model, always looking for proven technologies that can enhance our quality and efficiency."
    )
    ground_truth_dict = ground_truth_persona.model_dump()
    all_reports = [ground_truth_dict] + [parse_report_from_log(f) for f in log_files]

    print("\nRunning decay, survival, and entropy analysis...")
    decay_df, survival_df = analyze_decay_and_survival(all_reports, len(log_files))

    print("\nRunning fidelity analysis (Levenshtein and Cosine)...")
    fidelity_df = analyze_fidelity(all_reports, ground_truth_dict)

    decay_df.to_csv(os.path.join(metrics_dir, "information_decay_analysis.csv"), index=False)
    survival_csv_path = os.path.join(metrics_dir, "field_survival_analysis.csv")
    survival_comment = """# Explanation of columns:
# - field_name: The name of the data field from the report schema.
# - duration: The generation number at which the field's value was lost (became null). If the field survived all generations, this is the total number of generations.
# - event: A binary flag for survival analysis. 1 indicates the field was lost ('death' event occurred), 0 indicates the field survived to the end (data is 'censored').
"""
    with open(survival_csv_path, 'w', encoding='utf-8') as f:
        f.write(survival_comment)
        survival_df.to_csv(f, index=False, lineterminator='\n')

    fidelity_df.to_csv(os.path.join(metrics_dir, "information_fidelity_analysis.csv"), index=False)
    print(f"\nAll analysis CSVs saved to '{metrics_dir}'")

    plot_and_save_survival_curve(survival_df, os.path.join(metrics_dir, "kaplan_meier_survival_curve.png"))
    plot_and_save_fidelity_curves(fidelity_df, os.path.join(metrics_dir, "fidelity_decay_curve.png"))
    plot_and_save_entropy_curve(decay_df, os.path.join(metrics_dir, "entropy_decay_curve.png"))

if __name__ == "__main__":
    main()