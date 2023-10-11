#!/bin/bash
# sbatch train.sh
#
#SBATCH --job-name=SAM
#SBATCH --output=/share/home/liuy/project/SAM_finetune/logs/job_%J.out
#SBATCH --error=/share/home/liuy/project/SAM_finetune/logs/job_%J.err
#
#SBATCH --partition=tao
#SBATCH --nodelist=t001
#SBATCH --gres gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8


python finetune.py