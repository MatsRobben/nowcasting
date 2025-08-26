#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --job-name=benchmark
#SBATCH --output=/projects/prjs1634/nowcasting/results/logs/%x/%j.out
#SBATCH --error=/projects/prjs1634/nowcasting/results/logs/%x/%j.err

echo "Starting on $(hostname) at $(date)"
cd /projects/prjs1634/nowcasting

# Load environment
module purge
module load 2024
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0
source ../.venv/bin/activate

# Run benchmarks
echo "Benchmarking SPROG (PySTEPS)..."
python scripts/benchmark_inference.py --config="./config/sprog.yaml"

echo "Benchmarking Earthformer..."
python scripts/benchmark_inference.py --config="./results/tb_logs/earthformer/version_7/earthformer.yaml"

echo "Benchmarking LDcast..."
python scripts/benchmark_inference.py --config="./results/tb_logs/ldcast_nowcast/version_10/ldcast_nowcast_radar.yaml"
