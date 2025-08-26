#!/bin/bash
#SBATCH --partition=gpu_a100           # request full A100 GPU partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00               # short 2 hours for testing
#SBATCH --job-name=ldcast_nowcast_interpret
#SBATCH --output=/projects/prjs1634/nowcasting/results/logs/%x/%j.out
#SBATCH --error=/projects/prjs1634/nowcasting/results/logs/%x/%j.err

echo "Starting on $(hostname) at $(date)"
cd /projects/prjs1634/nowcasting

# Load necessary modules
module purge
module load 2024
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0

# Activate the Python virtual environment
source ../.venv/bin/activate

# Execute the Python evaluation script
# The `--config` argument specifies the YAML configuration file for the evaluation.
# Uncomment the lines that should be run for evaluation. 
# Note: PySTEPS does not require a GPU, while the others benifit greatly from it. 

# Examples:
# python scripts/interpret.py --config="./results/tb_logs/earthformer/version_17/earthformer_harmonie.yaml"
python scripts/interpret.py --config="./results/tb_logs/ldcast_nowcast/version_12/ldcast_nowcast_rad_har.yaml"
# python scripts/eval.py --config="./results/tb_logs/ldcast/version_2/ldcast.yaml"