#!/usr/bin/env bash
#SBATCH -A naiss2025-22-612
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH -J deeptopic_macaque
#SBATCH -o /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/logs/macaque_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/logs/macaque_%j.err

set -euo pipefail
echo "Starting job on $(hostname)"
nvidia-smi || true

source /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/miniconda3/etc/profile.d/conda.sh
conda activate work

python /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/scripts/02_train_macaque.py

echo "Finished job"