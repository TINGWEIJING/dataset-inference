#!/bin/bash -l

#SBATCH --partition=gpu-titan
#SBATCH --job-name=debug_03
#SBATCH --output=./log/out/%x-%j.txt
#SBATCH --error=./log/err/%x-%j.txt
#SBATCH --gpus=titanxp:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=8G
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --hint=nomultithread
#SBATCH --mail-user=tingweijingting2000@gmail.com

module load miniconda/miniconda3
module load cuda/cuda-10.2
source activate /home/user/tingweijing/env/DIenv
source ./script/debug_03.sh &> ./log/debug_03.txt

