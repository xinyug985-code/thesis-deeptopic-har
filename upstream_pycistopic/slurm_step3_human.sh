#!/bin/bash
#SBATCH -A naiss2025-22-612
#SBATCH -p main
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH -J step3_human
#SBATCH -o /cfs/klemming/projects/supr/snic2022-23-547/xinyu/runs/logs/step3_human_%j.out
#SBATCH -e /cfs/klemming/projects/supr/snic2022-23-547/xinyu/runs/logs/step3_human_%j.err

set -euo pipefail

source /cfs/klemming/projects/supr/snic2022-23-547/xinyu/conda_clean/etc/profile.d/conda.sh
unset PYTHONHOME
unset PYTHONPATH
conda activate topic

cd /cfs/klemming/projects/supr/snic2022-23-547/xinyu
python runs/step3_export_beds_human.py