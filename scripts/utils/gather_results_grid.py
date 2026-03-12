import os
import csv
import argparse
import statistics
import re
from pathlib import Path
from collections import defaultdict

def parse_exp_tag(tag):
    """Extracts guidance, mmd, ema, and ensemble from the directory tag."""
    # Try the new format with mmd first
    match_new = re.match(r"gs_(.+?)_mmd_(.+?)_ema_(.+?)_ens_(.+)", tag)
    if match_new:
        return match_new.groups()
    
    # Fallback to the older format without mmd
    match_old = re.match(r"gs_(.+?)_ema_(.+?)_ens_(.+)", tag)
    if match_old:
        gs, ema, ens = match_old.groups()
        return gs, "N/A", ema, ens
        
    return tag, "N/A", "N/A", "N/A"

def generate_heatmaps(csv_rows, output_dir):
    """Generates heatmaps of the grid search results."""
    try:
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[Notice] Skipping heatmap generation.")
        print("To generate heatmaps, please install the required libraries in your environment:")
        print("pip install pandas seaborn matplotlib")
        return

    print("\nGenerating heatmaps...")
    
    # Convert list of dicts to a pandas DataFrame
    df = pd.DataFrame(csv_rows)
    
    # Ensure axes are numeric where applicable
    df['Guidance Scale'] = pd.to_numeric(df['Guidance Scale'])
    df['EMA Alpha'] = pd.to_numeric(df['EMA Alpha'])
    df['Mean SR'] = pd.to_numeric(df['Mean SR'])
    
    # Keep Ensemble and MMD as strings for reliable grouping/sorting
    df['Ensemble Weights'] = df['Ensemble Weights'].astype(str)
    df['MMD Score'] = df['MMD Score'].astype(str)
    
    # Aggregate data: Average across all 7 tasks for each hyperparameter combo
    agg_df = df.groupby(['Guidance Scale', 'MMD Score', 'EMA Alpha', 'Ensemble Weights'])['Mean SR'].mean().reset_index()

    unique_ens = sorted(agg_df['Ensemble Weights'].unique())
    unique_mmd = sorted(agg_df['MMD Score'].unique())
    
    num_cols = len(unique_ens)
    num_rows = len(unique_mmd)
    
    # Setup subplots: Rows = MMD Score, Cols = Ensemble Weights
    # squeeze=False ensures `axes` is always a 2D array, even if it's 1x1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), squeeze=False)

    for r, mmd in enumerate(unique_mmd):
        for c, ens in enumerate(unique_ens):
            ax = axes[r, c]
            
            # Filter for current MMD and ensemble configuration
            subset = agg_df[(agg_df['Ensemble Weights'] == ens) & (agg_df['MMD Score'] == mmd)]
            
            if subset.empty:
                ax.set_visible(False) # Hide empty plots if grid search is asymmetrical
                continue
                
            # Create a pivot table for the heatmap
            pivot_table = subset.pivot(index="Guidance Scale", columns="EMA Alpha", values="Mean SR")
            
            # Render the heatmap
            sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".3f", ax=ax, 
                        cbar_kws={'label': 'Overall Mean Success Rate'})
            
            ax.set_title(f"Ensemble: {ens} | MMD: {mmd}")
            ax.invert_yaxis() # Puts lowest guidance scale at the bottom

    plt.suptitle("Hyperparameter Grid Search Performance", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save the plot in the same directory as the CSV
    plot_path = Path(output_dir) / "grid_search_heatmaps.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap successfully saved to: {plot_path}")


def process_results(base_dir, output_csv):
    data = defaultdict(lambda: defaultdict(list))
    base_path = Path(base_dir).resolve() 
    result_files = list(base_path.rglob("_result.txt"))
    
    if not result_files:
        print(f"No '_result.txt' files found in {base_path}")
        return

    for file_path in result_files:
        try:
            rel_path = file_path.relative_to(base_path)
            
            if len(rel_path.parts) < 3:
                continue 
                
            exp_tag = rel_path.parts[0]    
            seed_id = rel_path.parts[1]    
            task_name = rel_path.parts[2]  
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    continue
                
                last_line = lines[-1].strip()
                if last_line:
                    sr_value = float(last_line)
                    data[exp_tag][task_name].append(sr_value)
                    
        except (ValueError, IndexError) as e:
            print(f"Error parsing {file_path}: {e}")
            continue

    csv_rows = []
    
    # Expanded output header width to fit MMD variables nicely
    print(f"{'Exp Tag':<40} | {'Task':<25} | {'Mean SR':<10} | {'SEM':<10} | {'n'}")
    print("-" * 100)

    for exp_tag, tasks in data.items():
        gs, mmd, ema, ens = parse_exp_tag(exp_tag)
        all_srs_for_tag = []
        
        for task, values in tasks.items():
            n = len(values)
            mean_sr = statistics.mean(values)
            all_srs_for_tag.extend(values)
            
            if n > 1:
                sem = statistics.stdev(values) / (n ** 0.5)
            else:
                sem = 0.0
                
            print(f"{exp_tag:<40} | {task:<25} | {mean_sr:<10.4f} | {sem:<10.4f} | {n}")
            
            csv_rows.append({
                "Exp Tag": exp_tag,
                "Guidance Scale": gs,
                "MMD Score": mmd,
                "EMA Alpha": ema,
                "Ensemble Weights": ens,
                "Task": task,
                "Mean SR": mean_sr,
                "SEM": sem,
                "n": n
            })
            
        if all_srs_for_tag:
            tag_mean = statistics.mean(all_srs_for_tag)
            print(f"{'-> OVERALL FOR CONFIG':<40} | {'All Tasks Average':<25} | {tag_mean:<10.4f} | {'-':<10} | {len(all_srs_for_tag)}")
            print("-" * 100)

    # Write CSV
    fieldnames = ["Exp Tag", "Guidance Scale", "MMD Score", "EMA Alpha", "Ensemble Weights", "Task", "Mean SR", "SEM", "n"]
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
        
    print(f"\n✅ Results successfully saved to: {output_csv}")
    
    # Generate Heatmaps using the parsed data
    output_dir = Path(output_csv).parent
    generate_heatmaps(csv_rows, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse grid search results into a CSV and Heatmap.")
    parser.add_argument("dir", help="Base directory containing grid search results")
    parser.add_argument("--out", default="grid_search_results.csv", help="Output CSV filename")
    args = parser.parse_args()
    
    if os.path.isdir(args.dir):
        process_results(args.dir, args.out)
    else:
        print(f"Error: Directory '{args.dir}' does not exist.")