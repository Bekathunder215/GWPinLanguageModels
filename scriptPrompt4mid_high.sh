#!/bin/bash
#BSUB -R "rusage[mem=1GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q gpuv100
#BSUB -W 00:05
#BSUB -J prompt4midHigh
#BSUB -n 4
#BSUB -o hpcoutput/prompt4midHigh_%J.out
#BSUB -e hpcoutput/prompt4midHigh_%J.err



source /dtu/projects/02613_2025/conda/conda_init.sh
conda deactivate

conda activate ~/my_env

python src/prompt.py --exp four_mid_high

