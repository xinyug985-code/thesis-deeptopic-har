#!/usr/bin/env bash
#SBATCH -A naiss2025-22-612
#SBATCH -p shared
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH -t 00:10:00
#SBATCH -J meta
#SBATCH -o /cfs/klemming/projects/supr/snic2022-23-547/xinyu/runs/logs/%x_%j.out
#SBATCH -e /cfs/klemming/projects/supr/snic2022-23-547/xinyu/runs/logs/%x_%j.err

set -euo pipefail

source /cfs/klemming/projects/supr/snic2022-23-547/xinyu/conda_clean/etc/profile.d/conda.sh
unset PYTHONHOME
unset PYTHONPATH
conda activate topic

cd /cfs/klemming/projects/supr/snic2022-23-547/xinyu

python runs/step1b_addmeta.py