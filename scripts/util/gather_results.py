import os
import glob
import statistics
import argparse
from pathlib import Path

def process_results(base_dir):
    # Dictionary to store: { task_name: [sr1, sr2, ...] }
    task_data = {}
    
    # Path pattern: base_dir / <seed> / <task> / ... / _result.txt
    # We use rglob to find all _result.txt files under the base directory
    result_files = Path(base_dir).rglob("_result.txt")
    
    for file_path in result_files:
        try:
            # Based on your structure: base_dir/seed/task/...
            # file_path.parts: (..., base_dir, seed, task, ...)
            # We find the index of the base_dir to reliably pick seed and task
            parts = file_path.parts
            base_idx = parts.index(Path(base_dir).name)
            
            seed_id = parts[base_idx + 1]
            task_name = parts[base_idx + 2]
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    continue
                
                # Get the last line and convert to float
                last_line = lines[-1].strip()
                if last_line:
                    sr_value = float(last_line)
                    
                    if task_name not in task_data:
                        task_data[task_name] = []
                    task_data[task_name].append(sr_value)
                    
        except (ValueError, IndexError) as e:
            print(f"Error parsing {file_path}: {e}")
            continue

    # Check for missing seeds per task (Optional: assumes seeds are 0, 1, 2...)
    # This fulfills your request to see what's still running
    print(f"{'Task':<25} | {'Mean SR':<10} | {'SEM':<10}")
    print("-" * 50)

    # Prepare data for Excel (TSV format is easiest to paste)
    excel_output = []
    
    for task, values in task_data.items():
        n = len(values)
        mean_sr = statistics.mean(values)
        
        # SEM is standard deviation / sqrt(n)
        if n > 1:
            stdev = statistics.stdev(values)
            sem = stdev / (n ** 0.5)
        else:
            sem = 0.0  # Cannot calculate SEM with one data point
            
        print(f"{task:<25} | {mean_sr:<10.4f} | {sem:<10.4f} (n={n})")
        excel_output.append(f"{task}\t{mean_sr}\t{sem}")

    print("\n--- Copy/Paste below into Excel ---")
    print("Task Name\tMean SR\tSEM")
    for row in excel_output:
        print(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="The base directory containing seed folders")
    args = parser.parse_args()
    
    if os.path.isdir(args.dir):
        process_results(args.dir)
    else:
        print(f"Error: {args.dir} is not a valid directory.")