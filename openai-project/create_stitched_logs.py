import os
import glob

def get_turn_count(run_folder_path):
    """Reads the gen_0 log file to determine the number of turns in a run."""
    try:
        log_files = glob.glob(os.path.join(run_folder_path, "logs", "*_gen_0.txt"))
        if not log_files: return -1
        with open(log_files[0], 'r', encoding='utf-8') as f:
            content = f.read()
            # Count SalesAgent turns to determine length
            count = content.count("SalesAgent:")
            if count <= 10: return 5
            elif count <= 20: return 15
            else: return 25
    except Exception:
        return -1

def is_run_complete(run_folder_path):
    """Checks if a run has all 10 generations (0-9)."""
    for i in range(10):
        pattern = os.path.join(run_folder_path, "logs", f"*_gen_{i}.txt")
        if not glob.glob(pattern):
            return False
    return True

def stitch_logs(run_folder, turn_count, output_dir):
    """Reads gen_0 to gen_9 and writes them into one file."""
    
    run_id = os.path.basename(run_folder)
    output_filename = f"stitched_full_run_{turn_count}turns_{run_id}.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Stitching logs for {turn_count} turns (Source: {run_id})...")

    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write(f"STITCHED LOGS FOR {turn_count} TURNS\n")
        outfile.write(f"Source Run ID: {run_id}\n")
        outfile.write("="*60 + "\n\n")

        for i in range(10):
            # Find the specific log file for this generation
            pattern = os.path.join(run_folder, "logs", f"*_gen_{i}.txt")
            files = glob.glob(pattern)
            
            if not files:
                print(f"  Warning: Missing log for gen {i}")
                continue
                
            input_file = files[0]
            
            with open(input_file, 'r', encoding='utf-8') as infile:
                content = infile.read()
            
            # Add a clear header for the generation
            outfile.write(f"\n{'#'*20} GENERATION {i} {'#'*20}\n")
            outfile.write(f"Original File: {os.path.basename(input_file)}\n")
            outfile.write(f"{'-'*54}\n\n")
            
            outfile.write(content)
            outfile.write("\n\n") # Extra spacing between generations

    print(f"  -> Saved to: {output_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    output_dir = os.path.join(script_dir, "stitched_logs")
    
    os.makedirs(output_dir, exist_ok=True)

    # We need to find ONE valid run for each category
    needed_turns = {5: None, 15: None, 25: None}
    
    run_folders = sorted(glob.glob(os.path.join(results_dir, "run_*")), reverse=True)

    print("--- Searching for representative runs ---")
    
    for run_folder in run_folders:
        # If we have found examples for all 3, stop searching
        if all(v is not None for v in needed_turns.values()):
            break

        if not is_run_complete(run_folder):
            continue

        turns = get_turn_count(run_folder)
        
        if turns in needed_turns and needed_turns[turns] is None:
            needed_turns[turns] = run_folder
            print(f"Found candidate for {turns} turns: {os.path.basename(run_folder)}")

    print("\n--- Creating Stitched Files ---")
    
    for turns, run_folder in needed_turns.items():
        if run_folder:
            stitch_logs(run_folder, turns, output_dir)
        else:
            print(f"Warning: Could not find a complete run for {turns} turns.")

    print("\nDone. Check the 'stitched_logs' folder.")

if __name__ == "__main__":
    main()