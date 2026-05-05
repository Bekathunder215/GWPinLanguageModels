#!/bin/bash
#BSUB -R "rusage[mem=3GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q gpuv100
#BSUB -W 00:15
#BSUB -J s2_h32
#BSUB -n 4
#BSUB -o hpcoutput/s2_h32_%J.out
#BSUB -e hpcoutput/s2_h32_%J.err



source /dtu/projects/02613_2025/conda/conda_init.sh
conda deactivate

conda activate ~/my_env


python src/train.py --exp s2_h32
