#!/bin/bash
#BSUB -R "rusage[mem=2GB] span[hosts=1]"
#BSUB -q hpc
#BSUB -W 10:00
#BSUB -J slm
#BSUB -n 4
#BSUB -o hpcoutput/one_%J.out
#BSUB -e hpcoutput/one_%J.err

# InitializePythonenvironment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda deactivate

conda activate ~/my_env

python src/train.py --exp one
