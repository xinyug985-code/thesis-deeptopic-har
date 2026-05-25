#!/usr/bin/env bash
#SBATCH -A naiss2025-22-612
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -J stage2_topic_both
#SBATCH -o /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/logs/stage2_topic_both_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/logs/stage2_topic_both_%j.err

set -euo pipefail

echo "Starting job on $(hostname)"
date
nvidia-smi || true

source /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/miniconda3/etc/profile.d/conda.sh
conda activate work

cd /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt

export KERAS_BACKEND=torch
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/scripts/02_stage2_topic_modisco_top1000.py --species both

echo "Finished job"
date