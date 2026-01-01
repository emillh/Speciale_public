# filepath: c:\Users\elhe\Documents\Speciale\olmo-project\src\main.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from datetime import datetime
import pathlib
import time # Import the time module

# Use relative imports for modules within the src package
try:
    from .graph import build_graph
    from .report_schema import LeadReport
except ImportError:
    # Fallback for running the script directly
    from graph import build_graph
    from report_schema import LeadReport

def save_conversation(final_state, generation_number, timestamp):
    """Saves the full conversation and final report to a uniquely named text file for the experiment."""
    log_dir = "conversation_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Use the generation number and a shared timestamp for the filename
    filename = f"telephone_game_{timestamp}_gen_{generation_number}.txt"
    filepath = os.path.join(log_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"--- Telephone Game: Generation {generation_number} ---\n\n")
        f.write("--- Full Conversation History ---\n")
        for message in final_state["messages"]:
            # The 'name' attribute is set in our graph nodes
            if hasattr(message, 'name') and message.name:
                 f.write(f"{message.name}: {message.content}\n")
            else:
                 # Fallback for any messages that might not have a name
                 f.write(f"System/Unknown: {message.content}\n")

        f.write("\n\n--- Final Extracted Report (Input for Next Generation's Persona) ---\n")
        f.write(final_state["report"].model_dump_json(indent=2))
    print(f"Conversation for generation {generation_number} saved to {filepath}")

def main():
    """
    Main function to run the "telephone game" experiment.
    """
    # --- 1. SETUP ---
    # Build an absolute path to the .env file.
    # This makes the script runnable from any directory.
    script_path = pathlib.Path(__file__).parent.parent # This goes from /src up to /openai-project
    dotenv_path = script_path / '.env'
    load_dotenv(dotenv_path=dotenv_path)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Make sure it's in an 'openai-project/.env' file.")

    # --- Define LLMs for the experiment ---
    # Use the powerful model for critical data extraction
    llm_extractor = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=api_key)
    # Use a cheaper, faster model for the conversational parts
    llm_conversation = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)


    # --- 2. DEFINE THE "GROUND TRUTH" ---
    # This is the perfect, complete persona for the very first generation.
    ground_truth_persona = LeadReport(
        company_name="Innovate Inc.",
        contact_person_name="Alex",
        contact_person_role="Project Manager",
        contact_person_email="alex@innovate.com",
        contact_person_phone="555-0102",
        is_decision_maker=True,
        industry="Advanced Manufacturing",
        address="123 Innovation Drive, Tech Park, CA 90210",
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

    # --- 3. RUN THE EXPERIMENT LOOP ---
    num_generations = 10
    current_persona_report = ground_truth_persona
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(num_generations):
        print(f"\n\n{'='*20} STARTING GENERATION {i} {'='*20}\n")

        # Build a new graph for this generation, passing in the models
        app = build_graph(
            llm_conversation=llm_conversation, 
            llm_extractor=llm_extractor, 
            customer_persona_report=current_persona_report
        )

        # The Sales Agent always starts with a blank report
        initial_state = {
            "messages": [],
            "report": LeadReport(),
            "turn_number": 0,
        }

        # Run the conversation
        final_state = app.invoke(initial_state, config={'recursion_limit': 10000})

        # Save the results for this generation
        save_conversation(final_state, generation_number=i, timestamp=experiment_timestamp)

        # The final report from this generation becomes the persona for the next one
        current_persona_report = final_state['report']

        # --- Add a delay to respect API rate limits ---
        print(f"\n--- PAUSING FOR 10 SECONDS TO MANAGE RATE LIMITS ---")
        time.sleep(10)

    print(f"\n\n{'='*20} EXPERIMENT FINISHED {'='*20}\n")
    print(f"Completed {num_generations} generations. Check the 'conversation_logs' directory.")

if __name__ == "__main__":
    main()