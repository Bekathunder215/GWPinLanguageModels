#!/bin/bash
#BSUB -R "rusage[mem=3GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q gpuv100
#BSUB -W 1:00
#BSUB -J prompt5mid
#BSUB -n 4
#BSUB -o hpcoutput/prompt5mid_%J.out
#BSUB -e hpcoutput/prompt5mid_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda deactivate

conda activate ~/my_env

python src/prompt.py --exp five_mid
