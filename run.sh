#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=node08
#SBATCH --job-name="ast-sc"

python3 trainer.py
#python3 testing.py -saved_model save_models/check_point_48_0.9694038245219347
