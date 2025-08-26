import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 18,          # base font size
    "axes.titlesize": 20,     # subplot titles
    "axes.labelsize": 18,     # x/y axis labels
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18
})

def plot_permutation_importance(files_dict, out_dir):
    """
    Create a 2x3 subplot figure for permutation importance for two models across 3 lead times.

    Args:
        files_dict (dict): Dictionary mapping model names to a list of dicts with:
            'path' (str): CSV file path
            'lead_time' (str): e.g., '30min'
        out_dir (str): Directory to save plots
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Channel/group names for the two models
    earthformer_groups = {
        "Group_[1]": "IR 0.16",
        "Group_[2]": "IR 0.39",
        "Group_[3]": "WV 0.62",
        "Group_[4]": "WV 0.73",
        "Group_[5]": "IR 0.87",
        "Group_[6]": "IR 0.97",
        "Group_[7]": "IR 1.08",
        "Group_[8]": "IR 1.20",
        "Group_[9]": "IR 1.34"
    }
    
    lnowcaster_groups = {
        "Group_[1, [0]]": "IR 0.16",
        "Group_[1, [1]]": "IR 0.39",
        "Group_[1, [2]]": "WV 0.62",
        "Group_[1, [3]]": "WV 0.73",
        "Group_[1, [4]]": "IR 0.87",
        "Group_[1, [5]]": "IR 0.97",
        "Group_[1, [6]]": "IR 1.08",
        "Group_[1, [7]]": "IR 1.20",
        "Group_[1, [8]]": "IR 1.34"
    }


    # earthformer_groups = {
    #     "Group_[1]": "Pressure",
    #     "Group_[2]": "Temp",
    #     "Group_[3]": "Dew Pt",
    #     "Group_[4]": "U-Wind",
    #     "Group_[5]": "V-Wind",
    #     "Group_[6]": "RH",
    #     "Group_[7]": "Precip",
    #     "Group_[8]": "Cloud",
    #     "Group_[9]": "Graupel"
    # }
    
    # lnowcaster_groups = {
    #     "Group_[1, [0]]": "Pressure",
    #     "Group_[1, [1]]": "Temp",
    #     "Group_[1, [2]]": "Dew Pt",
    #     "Group_[1, [3]]": "U-Wind",
    #     "Group_[1, [4]]": "V-Wind",
    #     "Group_[1, [5]]": "RH",
    #     "Group_[1, [6]]": "Precip",
    #     "Group_[1, [7]]": "Cloud",
    #     "Group_[1, [8]]": "Graupel"
    # }
        
    group_maps = {
        "Earthformer": earthformer_groups,
        "LNowcaster": lnowcaster_groups
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 12), sharex=False)
    
    for row_idx, model in enumerate(files_dict.keys()):
        runs = files_dict[model]
        group_map = group_maps[model]
        
        for col_idx, run in enumerate(runs):
            ax = axes[row_idx, col_idx]
            df = pd.read_csv(run['path'])
            
            # Map group names to readable labels
            labels = ["Baseline" if g=="no-permutation" else group_map.get(g, g) 
                      for g in df['Group']]
            
            # Convert metric to percentage of baseline
            baseline_metric = df.loc[df['Group'] == "no-permutation", 'Metric'].values[0]
            percentages = df['Metric'] / baseline_metric * 100
            
            # Reverse order so baseline is on top
            labels = labels[::-1]
            percentages = percentages[::-1]
            
            # Plot horizontal bars
            ax.barh(labels, percentages, color='black', alpha=0.8)
            
            # # Add value labels
            # for i, pct in enumerate(percentages):
            #     ax.text(pct + 1, i, f"{pct:.1f}%", va='center', fontweight='bold')

            for i, pct in enumerate(percentages):
                if pct > 25:  # enough space to write inside
                    ax.text(pct - 2, i, f"{pct:.1f}%", va='center',
                            ha='right', fontsize=17, fontweight='bold', color='white')
                else:
                    ax.text(pct + 1, i, f"{pct:.1f}%", va='center',
                            ha='left', fontsize=17, fontweight='bold', color='black')
            
            # Set x-axis label only on bottom row
            if row_idx == 1:
                ax.set_xlabel("Performance (% of baseline)", fontsize=20, fontweight='bold')
            else:
                ax.set_xlabel("")
            
            # Show lead times only for top row
            if row_idx == 0:
                ax.set_title(f"Lead time {run.get('lead_time', run.get('time', ''))}", fontsize=24, fontweight='bold')
            else:
                ax.set_title("")
            
            ax.set_xlim(0, max(percentages)*1.15)

        # Add model name on the left side, rotated 90°
        if row_idx == 0:
            fig.text(0.05, 0.75 - row_idx*0.5, model, va='center', rotation=90, fontsize=22, fontweight='bold')
        else:
            fig.text(0.05, 0.80 - row_idx*0.5, model, va='center', rotation=90, fontsize=22, fontweight='bold')
    
    plt.tight_layout(rect=[0.06, 0.03, 1, 0.97])
    save_path = os.path.join(out_dir, f"permutation_importance_sat.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved combined 2x3 plot at {save_path}")


if __name__ == "__main__":
    # Satellite
    files_dict = {
        "Earthformer": [
            {"path": "/projects/prjs1634/nowcasting/results/scores/evaluations/ef_rad_sat_in/permutation_importance/results_run1.csv", "lead_time": "30min"},
            {"path": "/projects/prjs1634/nowcasting/results/scores/evaluations/ef_rad_sat_in/permutation_importance/results_run3.csv", "lead_time": "60min"},
            {"path": "/projects/prjs1634/nowcasting/results/scores/evaluations/ef_rad_sat_in/permutation_importance/results_run2.csv", "lead_time": "90min"},
        ],
        "LNowcaster": [
            {"path": "/projects/prjs1634/nowcasting/results/scores/evaluations/ldcast_nowcast_sat/permutation_importance/results_run1.csv", "lead_time": "30min"},
            {"path": "/projects/prjs1634/nowcasting/results/scores/evaluations/ldcast_nowcast_sat/permutation_importance/results_run2.csv", "lead_time": "60min"},
            {"path": "/projects/prjs1634/nowcasting/results/scores/evaluations/ldcast_nowcast_sat/permutation_importance/results_run3.csv", "lead_time": "90min"},
        ]
    }
    
    # Harmonie
    # files_dict = {
    #     "Earthformer": [
    #         {"path": "/projects/prjs1634/nowcasting/results/scores/evaluations/ef_rad_har/permutation_importance/results_run2.csv", "lead_time": "30min"},
    #         {"path": "/projects/prjs1634/nowcasting/results/scores/evaluations/ef_rad_har/permutation_importance/results_run1.csv", "lead_time": "60min"},
    #         {"path": "/projects/prjs1634/nowcasting/results/scores/evaluations/ef_rad_har/permutation_importance/results_run3.csv", "lead_time": "90min"},
    #     ],
    #     "LNowcaster": [
    #         {"path": "/projects/prjs1634/nowcasting/results/scores/evaluations/ldcast_nowcast_har/permutation_importance/results_run1.csv", "lead_time": "30min"},
    #         {"path": "/projects/prjs1634/nowcasting/results/scores/evaluations/ldcast_nowcast_har/permutation_importance/results_run2.csv", "lead_time": "60min"},
    #         {"path": "/projects/prjs1634/nowcasting/results/scores/evaluations/ldcast_nowcast_har/permutation_importance/results_run3.csv", "lead_time": "90min"},
    #     ]
    # }

    save_dir = "/projects/prjs1634/nowcasting/results/figures/paper_figures"
    plot_permutation_importance(files_dict, out_dir=save_dir)