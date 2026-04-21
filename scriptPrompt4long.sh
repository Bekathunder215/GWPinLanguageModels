#!/bin/bash
#BSUB -R "rusage[mem=3GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q gpuv100
#BSUB -W 1:00
#BSUB -J prompt4long
#BSUB -n 4
#BSUB -o hpcoutput/prompt4long_%J.out
#BSUB -e hpcoutput/prompt4long_%J.err




source ~/.local/bin/env
source ~/GWPinLanguageModels/.venv/bin/activate

python src/prompt.py --exp four_long
