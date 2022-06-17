#!/bin/bash -l

#SBATCH --partition=gpu-titan
#SBATCH --job-name=first_run
#SBATCH --output=./_log/out-%x-%j.txt
#SBATCH --error=./_log/err-%x-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=39G
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mail-user=tingweijingting2000@gmail.com

export CUDA_VISIBLE_DEVICES=0
source ./venv/bin/activate
source ./scripts/test.sh &> ./_log/test.txt

