import os
import glob
import json
import pandas as pd
import time
import argparse
import pathlib
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# We no longer need HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import numpy as np

# Use relative imports for modules within the src package
try:
    from src.report_schema import LeadReport
except ImportError:
    # Fallback for running the script directly
    from report_schema import LeadReport

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
            # Handle potential malformed JSON by finding the start and end braces
            json_str_raw = content[report_start_index + len(report_marker):].strip()
            first_brace = json_str_raw.find('{')
            last_brace = json_str_raw.rfind('}')
            if first_brace == -1 or last_brace == -1:
                print(f"Warning: Could not find valid JSON object in {file_path}")
                return None
            json_string = json_str_raw[first_brace:last_brace+1]
            return json.loads(json_string)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def parse_llm_json_output(raw_output: str):
    """
    Extracts and parses a JSON object from a raw string that may contain
    other text, like conversational filler from an LLM.
    """
    # Find the start and end of the JSON object
    try:
        first_brace = raw_output.find('{')
        last_brace = raw_output.rfind('}')
        if first_brace == -1 or last_brace == -1:
            raise json.JSONDecodeError("Could not find JSON object in the output.", raw_output, 0)
        
        json_string = raw_output[first_brace:last_brace+1]
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        # Re-raise with a more informative message
        raise json.JSONDecodeError(f"Failed to parse JSON. Raw output was: '{raw_output}'", e.doc, e.pos)


def get_judge_prompt():
    """Creates the prompt template for the LLM Judge."""
    system_message = """You are an evaluation assistant. Your task is to compare a ground-truth field from a structured customer report with a candidate field from a later generation of a "telephone game". You must judge how well the candidate preserves the semantic meaning of the ground truth. Be strict and consistent.
    
First, think step-by-step about the comparison. Analyze the core semantic elements of the ground truth. Then, compare them to the candidate. Finally, based on your reasoning, provide your evaluation.

Return your answer as valid JSON with keys: "score", "error_type", "explanation".
"""

    user_template = """You are given:
A field name from a structured report.
The original ground-truth value for this field.
A candidate value for the same field, produced after several noisy transmission steps.

Your job is to:
1. Rate the semantic fidelity of the candidate relative to the ground truth on a 0–5 scale:
5 = Fully faithful (same meaning, minor wording differences only)
4 = Mostly faithful (main meaning correct, small omissions or slight generalization)
3 = Partially faithful (general idea preserved but important details missing or quite vague)
2 = Weakly related (loosely on-topic but mostly incomplete or off)
1 = Contradictory or fabricated (conflicts with ground truth or adds wrong information)
0 = No information (empty, "unknown", or not answering the field at all)

2. Assign an error_type label from this list:
"faithful"
"vague"
"overly_general"
"contradictory"
"fabricated"
"omitted"

3. Provide a one-sentence explanation for your score.

IMPORTANT: You MUST return your answer as a single, valid JSON object and nothing else. Do not include any text or markdown formatting before or after the JSON.
Field name: {field_name}
Ground-truth value: {truth_value}
Candidate value: {candidate_value}
"""
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", user_template)
    ])

def plot_judge_scores(df, judge_names, output_path):
    """Plots the average judge score over generations for multiple judges."""
    plt.figure(figsize=(12, 7))
    
    for judge in judge_names:
        score_col = f'score_{judge}'
        # Convert score column to numeric, coercing errors to NaN, then drop them
        df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
        avg_scores = df.dropna(subset=[score_col]).groupby('generation')[score_col].mean().reset_index()
        plt.plot(avg_scores['generation'], avg_scores[score_col], marker='o', linestyle='-', label=f'Avg. Score ({judge})')

    plt.title('Qualitative Fidelity (LLM Judge Scores) Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Average Score (0-5 Scale)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(0, 5)
    
    max_gen = df['generation'].max()
    if not np.isnan(max_gen):
        plt.xticks(range(int(max_gen) + 1))

    plt.savefig(output_path)
    plt.close()
    print(f"Judge score plot saved to '{output_path}'")

def plot_error_types(df, judge_names, output_path_template):
    """Plots a stacked bar chart of error types per generation for each judge."""
    for judge in judge_names:
        error_col = f'error_type_{judge}'
        
        # Filter out rows where error type might be missing
        plot_df = df.dropna(subset=[error_col])
        
        error_counts = plot_df.groupby(['generation', error_col]).size().unstack(fill_value=0)
        
        if error_counts.empty:
            print(f"No data to plot for judge '{judge}'. Skipping error type plot.")
            continue

        # Normalize to get percentages
        error_percentages = error_counts.divide(error_counts.sum(axis=1), axis=0) * 100
        
        ax = error_percentages.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='Paired')
        
        plt.title(f'Distribution of Error Types per Generation (Judge: {judge})')
        plt.xlabel('Generation')
        plt.ylabel('Percentage of Fields (%)')
        plt.xticks(rotation=0)
        plt.legend(title='Error Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        output_path = output_path_template.format(judge_name=judge)
        plt.savefig(output_path)
        plt.close()
        print(f"Error type distribution plot for {judge} saved to '{output_path}'")

def analyze_inter_rater_reliability(df, judges):
    """Calculates and prints inter-rater reliability stats."""
    print("\n--- Inter-Rater Reliability Analysis ---")
    
    if len(judges) < 2:
        print("Need at least two judges to compare.")
        return

    # Use the first two judges for comparison
    judge1, judge2 = judges[0], judges[1]
    score1_col, score2_col = f'score_{judge1}', f'score_{judge2}'
    
    # Ensure scores are numeric for calculation
    df[score1_col] = pd.to_numeric(df[score1_col], errors='coerce')
    df[score2_col] = pd.to_numeric(df[score2_col], errors='coerce')
    
    # Drop rows where either judge failed to provide a score
    comparison_df = df.dropna(subset=[score1_col, score2_col]).copy()
    
    if comparison_df.empty:
        print("Not enough data to compare judges.")
        return

    # 1. Agreement Rate
    agreement = np.sum(comparison_df[score1_col] == comparison_df[score2_col]) / len(comparison_df)
    print(f"Score Agreement Rate ({judge1} vs {judge2}): {agreement:.2%}")

    # 2. Cohen's Kappa
    kappa = cohen_kappa_score(comparison_df[score1_col], comparison_df[score2_col])
    print(f"Cohen's Kappa ({judge1} vs {judge2}): {kappa:.4f}")
    
    # 3. Score Correlation (Pearson)
    correlation = comparison_df[score1_col].corr(comparison_df[score2_col], method='pearson')
    print(f"Pearson Correlation ({judge1} vs {judge2}): {correlation:.4f}")
    print("----------------------------------------")


def main(run_folder):
    """Main function to run the LLM-as-a-Judge analysis."""
    # --- 1. SETUP ---
    script_path = pathlib.Path(__file__).parent
    dotenv_path = script_path / '.env'
    load_dotenv(dotenv_path=dotenv_path)

    # --- Initialize Judges using the OpenAI-compatible endpoint for Hugging Face ---
    judges = {
        "gpt4o": ChatOpenAI(
            model="gpt-4o", 
            temperature=0, 
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        "llama3-70b": ChatOpenAI(
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            base_url="https://router.huggingface.co/v1",
            api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            temperature=0,
        )
    }
    judge_names = list(judges.keys())
    
    # Validate that API keys are present by checking the environment variables
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("API key for judge 'gpt4o' (OPENAI_API_KEY) not found. Please check your .env file.")
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        raise ValueError("API key for judge 'llama3-70b' (HUGGINGFACEHUB_API_TOKEN) not found. Please check your .env file.")

    judge_prompt = get_judge_prompt()
    output_parser = JsonOutputParser()

    # --- 2. Find and Load Data ---
    logs_dir = os.path.join(run_folder, "logs")
    metrics_dir = os.path.join(run_folder, "metrics")
    
    log_files = sorted(glob.glob(os.path.join(logs_dir, "telephone_game_*.txt")), key=lambda x: int(re.search(r'gen_(\d+)\.txt$', x).group(1)))

    if not log_files:
        print(f"No log files found in {logs_dir}. Exiting.")
        return

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
    all_reports = [parse_report_from_log(f) for f in log_files]

    # --- 3. Run Judging Analysis ---
    print(f"Starting LLM-as-a-Judge analysis with judges: {judge_names}")
    results = []
    for gen_idx, report in enumerate(all_reports):
        if not report: continue
        
        for field_name, truth_value in ground_truth_dict.items():
            base_result = {'generation': gen_idx, 'field_name': field_name}
            candidate_value = report.get(field_name)

            if candidate_value is None or str(candidate_value).strip() == "":
                # Handle omitted fields locally for all judges
                for name in judge_names:
                    base_result[f'score_{name}'] = 0
                    base_result[f'error_type_{name}'] = 'omitted'
                    base_result[f'explanation_{name}'] = 'The field was empty or null.'
            else:
                # Call each LLM judge for non-empty fields
                for name, judge in judges.items():
                    try:
                        print(f"  - Gen {gen_idx}, Field: {field_name} (Judge: {name})")
                        # The chain now returns raw model output (AIMessage)
                        judge_chain = judge_prompt | judge
                        raw_response = judge_chain.invoke({
                            "field_name": field_name,
                            "truth_value": str(truth_value),
                            "candidate_value": str(candidate_value)
                        })
                        
                        # Use our robust parser on the string content
                        response = parse_llm_json_output(raw_response.content)

                        base_result[f'score_{name}'] = response.get('score')
                        base_result[f'error_type_{name}'] = response.get('error_type')
                        base_result[f'explanation_{name}'] = response.get('explanation')
                        time.sleep(1) # Be kind to the APIs
                    except Exception as e:
                        print(f"    ERROR: Could not judge with {name}. Reason: {e}")
                        base_result[f'score_{name}'] = -1
                        base_result[f'error_type_{name}'] = 'judgement_error'
                        base_result[f'explanation_{name}'] = str(e)
            results.append(base_result)

    # --- 4. Save Results ---
    judge_df = pd.DataFrame(results)
    output_csv_path = os.path.join(metrics_dir, "multi_judge_analysis.csv")
    judge_df.to_csv(output_csv_path, index=False)
    print(f"\nMulti-judge analysis complete. Results saved to {output_csv_path}")

    # --- 5. Analyze and Generate Plots ---
    if len(judge_names) > 1:
        analyze_inter_rater_reliability(judge_df, judge_names)
    
    plot_judge_scores(judge_df, judge_names, os.path.join(metrics_dir, "multi_judge_score_decay.png"))
    plot_error_types(judge_df, judge_names, os.path.join(metrics_dir, "error_type_dist_{judge_name}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-as-a-Judge analysis on a specific experiment run.")
    parser.add_argument(
        '--run_folder', 
        type=str, 
        help="Path to the specific run folder (e.g., 'results/run_20251126_195954'). If not provided, the latest run will be used."
    )
    args = parser.parse_args()

    # Get the directory where the script is located
    script_dir = pathlib.Path(__file__).parent.resolve()

    if args.run_folder:
        # If a run_folder is provided, it might be relative, so resolve it
        run_to_analyze = script_dir / args.run_folder
    else:
        # Find the latest run folder automatically relative to the script
        results_dir = script_dir / "results"
        all_runs = [d for d in results_dir.iterdir() if d.is_dir()]
        if not all_runs:
            print(f"No run folders found in the '{results_dir}' directory.")
            exit()
        latest_run = max(all_runs, key=lambda p: p.stat().st_mtime)
        run_to_analyze = latest_run
        print(f"No --run_folder specified. Analyzing the latest run: {run_to_analyze}")

    main(run_to_analyze)