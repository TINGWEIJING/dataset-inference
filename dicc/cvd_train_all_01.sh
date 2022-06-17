#!/bin/bash -l

#SBATCH --partition=gpu-v100s
#SBATCH --job-name=cvd_train_all_01
#SBATCH --output=./_log/out-%x-%j.txt
#SBATCH --error=./_log/err-%x-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=120G
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:2
#SBATCH --hint=nomultithread
#SBATCH --mail-user=tingweijingting2000@gmail.com

export CUDA_VISIBLE_DEVICES=0,1
source ./venv/bin/activate
source ./scripts/lazy_cat_dog_train_02.sh &> ./_log/lazy_cat_dog_train_02.txt

