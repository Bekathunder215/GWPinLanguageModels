#!/bin/bash
#BSUB -R "rusage[mem=7GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q gpuv100
#BSUB -W 10:00
#BSUB -J slm2
#BSUB -n 4
#BSUB -o main_%J.out
#BSUB -e main_%J.err

# InitializePythonenvironment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda deactivate

conda activate ~/my_env

python src/train.py --exp two
