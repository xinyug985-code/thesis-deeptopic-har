#!/bin/bash
#SBATCH -A naiss2025-22-612
#SBATCH -p main
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=440G
#SBATCH -J lda_macaque
#SBATCH -o /cfs/klemming/projects/supr/snic2022-23-547/xinyu/runs/logs/step2_macaque_%j.out
#SBATCH -e /cfs/klemming/projects/supr/snic2022-23-547/xinyu/runs/logs/step2_macaque_%j.err

set -euo pipefail

source /cfs/klemming/projects/supr/snic2022-23-547/xinyu/conda_clean/etc/profile.d/conda.sh
unset PYTHONHOME
unset PYTHONPATH
conda activate topic

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MALLET_MEMORY=400G

cd /cfs/klemming/projects/supr/snic2022-23-547/xinyu
mkdir -p runs/logs

which python
python -c "import sys; print(sys.executable); import pycisTopic; print('pycisTopic:', pycisTopic.__version__)"

python runs/step2_run_lda_macaque.py