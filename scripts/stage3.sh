#!/usr/bin/env bash
#SBATCH -A naiss2025-22-612
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1
#SBATCH --cpus-per-task=8
#SBATCH -t 10:00:00
#SBATCH -J stage3_modisco_match
#SBATCH -o /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/logs/stage3_match_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/logs/stage3_match_%j.err

set -euo pipefail

echo "Starting job on $(hostname)"
date

source /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/miniconda3/etc/profile.d/conda.sh
conda activate work

cd /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python scripts/04_stage3_modisco_pattern_matching.py --species both

echo "Finished job"
date