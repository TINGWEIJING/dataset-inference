#!/bin/bash -l

#SBATCH --partition=gpu-k10c
#SBATCH --job-name=first_run
#SBATCH --output=out-%x-%j.txt
#SBATCH --error=err-%x-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=19G
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread
#SBATCH --mail-user=tingweijingting2000@gmail.com

export CUDA_VISIBLE_DEVICES=0,1,2,3
source ./venv/bin/activate
source ./scripts/test.sh &> ./_log/test.txt

