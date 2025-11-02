#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=00:01:00
#SBATCH --mem=500MB

# Notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sreehari.suraj@adelaide.edu.au

#module load Python/3.9.6-GCCcore-11.2.0
#module load SciPy-bundle/2021.10-foss-2021b
#module load matplotlib/3.5.2-foss-2021b

# module load Anaconda3/2023.03

# python run.py

source ~/.bashrc
conda activate vision-llm
python llavaov.py
