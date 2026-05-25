#!/usr/bin/env bash
#SBATCH -A naiss2025-22-612
#SBATCH -p alvis
#SBATCH -C NOGPU
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH -J tomtom_tf
#SBATCH -o /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/logs/tomtom_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/logs/tomtom_%j.err

set -euo pipefail

echo "Starting job on $(hostname)"
date

source /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/miniconda3/etc/profile.d/conda.sh
conda activate work
cd /mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt

export PATH=$CONDA_PREFIX/bin:/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/miniconda3/envs/meme_env/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

which python
which tomtom

python scripts/09_annotate_patterns_to_tf.py

echo "Finished job"
date