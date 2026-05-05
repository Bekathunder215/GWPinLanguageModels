#!/bin/bash
#BSUB -R "rusage[mem=3GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q gpuv100
#BSUB -W 8:00
#BSUB -J s2_h1
#BSUB -n 4
#BSUB -o hpcoutput/s2_h1_%J.out
#BSUB -e hpcoutput/s2_h1_%J.err



source /dtu/projects/02613_2025/conda/conda_init.sh
conda deactivate

conda activate ~/my_env


PYTHONPATH=src python src/train.py --exp s2_h1
