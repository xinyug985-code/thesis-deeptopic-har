#!/bin/bash
#SBATCH -A naiss2025-22-612
#SBATCH -t 12:00:00
#SBATCH -p main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -J cobj_macaque
#SBATCH -o /cfs/klemming/projects/supr/snic2022-23-547/xinyu/runs/logs/step1_macaque_%j.out
#SBATCH -e /cfs/klemming/projects/supr/snic2022-23-547/xinyu/runs/logs/step1_macaque_%j.err

set -euo pipefail

source /cfs/klemming/projects/supr/snic2022-23-547/xinyu/conda_clean/etc/profile.d/conda.sh
unset PYTHONHOME
unset PYTHONPATH
conda activate topic

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /cfs/klemming/projects/supr/snic2022-23-547/xinyu

mkdir -p runs/logs

which python
python -c "import sys; print(sys.executable); import pycisTopic; print('pycisTopic:', pycisTopic.__version__)"

python runs/step1_make_cobj_macaque.py