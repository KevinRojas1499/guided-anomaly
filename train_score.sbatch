#!/bin/bash

#SBATCH -J OCT_score_train
#SBATCH --account=gts-mtao8
#SBATCH -N1 --gres=gpu:2
#SBATCH --mem-per-gpu=40G
#SBATCH -t8:00:00
#SBATCH -q embers
#SBATCH -o score_training.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=krojas6@gatech.edu

cd /storage/home/hcoda1/7/krojas6/p-mtao8-0/repos/guided-anomaly

module load anaconda3
module load gcc
conda deactivate
conda activate guided
bash training_scripts/train_score.sh