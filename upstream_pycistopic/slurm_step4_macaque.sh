#!/usr/bin/env bash
#SBATCH -A naiss2025-22-612
#SBATCH -p long
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH -t 00:50:00
#SBATCH -J mk_regions_macaque
#SBATCH -o /cfs/klemming/projects/supr/snic2022-23-547/xinyu/runs/logs/%x_%j.out
#SBATCH -e /cfs/klemming/projects/supr/snic2022-23-547/xinyu/runs/logs/%x_%j.err

set -euo pipefail

source /cfs/klemming/projects/supr/snic2022-23-547/xinyu/conda_clean/etc/profile.d/conda.sh
unset PYTHONHOME
unset PYTHONPATH
conda activate topic

cd /cfs/klemming/projects/supr/snic2022-23-547/xinyu

python runs/step4_get_region_macaque.py