#!/bin/bash
#BSUB -R "rusage[mem=3GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q gpuv100
#BSUB -W 8:00
#BSUB -J s2_h8
#BSUB -n 4
#BSUB -o hpcoutput/s2_h8_%J.out
#BSUB -e hpcoutput/s2_h8_%J.err



source ~/.local/bin/env
source ~/GWPinLanguageModels/.venv/bin/activate

python src/train.py --exp s2_h8
