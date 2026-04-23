#!/bin/bash
#BSUB -R "rusage[mem=3GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q gpuv100
#BSUB -W 1:00
#BSUB -J prompt5low
#BSUB -n 4
#BSUB -o hpcoutput/prompt5low_%J.out
#BSUB -e hpcoutput/prompt5low_%J.err




source ~/.local/bin/env
source ~/GWPinLanguageModels/.venv/bin/activate

PYTHONPATH=src python src/prompt.py --exp five_low
