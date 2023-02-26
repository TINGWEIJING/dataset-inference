#!/bin/bash -l

#SBATCH --partition=gpu-v100s
#SBATCH --job-name=run_54
#SBATCH --output=./log/out/%x-%j.txt
#SBATCH --error=./log/err/%x-%j.txt
#SBATCH --gpus=v100s:2
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=16G
#SBATCH --qos=long
#SBATCH --mail-type=ALL
#SBATCH --hint=nomultithread
#SBATCH --mail-user=tingweijingting2000@gmail.com

module load miniconda/miniconda3
module load cuda/cuda-10.2
source activate /home/user/tingweijing/env/DIenv
source ./script/run_54.sh &> ./log/run_54.txt

