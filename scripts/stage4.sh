#!/usr/bin/env bash
#SBATCH -A naiss2025-22-612
#SBATCH -p alvis
#SBATCH -C NOGPU
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH -J stage4_pattern_matrix
#SBATCH -o /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/logs/stage4_matrix_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/logs/stage4_matrix_%j.err

set -euo pipefail

echo "Starting job on $(hostname)"
date

source /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/miniconda3/etc/profile.d/conda.sh
conda activate work

cd /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python scripts/05_stage4_process_patterns_matrix.py --species both

echo "Finished job"
date