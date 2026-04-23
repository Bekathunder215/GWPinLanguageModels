#!/bin/bash
#BSUB -R "rusage[mem=2GB] span[hosts=1]"
#BSUB -q hpc
#BSUB -W 10:00
#BSUB -J slm
#BSUB -n 4
#BSUB -o hpcoutput/one_%J.out
#BSUB -e hpcoutput/one_%J.err

# InitializePythonenvironment



source ~/.local/bin/env
source ~/GWPinLanguageModels/.venv/bin/activate

PYTHONPATH=src python src/train.py --exp one
