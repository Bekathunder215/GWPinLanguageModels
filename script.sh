#!/bin/bash
#BSUB -R "rusage[mem=3GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q gpuv100
#BSUB -W 8:00
#BSUB -J slm
#BSUB -n 4
#BSUB -o hpcoutput/one_%J.out
#BSUB -e hpcoutput/one_%J.err

# InitializePythonenvironment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda deactivate

conda activate ~/my_env

python src/train.py --exp one
