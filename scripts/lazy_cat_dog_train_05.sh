#!/bin/sh
export CUDA_VISIBLE_DEVICES=1,2

# Case: Train resnet of splitted Kaggle train dataset

# echo "# CIFAR10-Cat-Dog"
# python3 ./src/train.py \
#   --batch_size 512 \
#   --mode teacher \
#   --normalize 1 \
#   --model_id resnet18_victim_normalized \
#   --lr_mode 5 \
#   --lr_max 0.03 \
#   --opt_type Adam \
#   --epochs 100 \
#   --dataset CIFAR10-Cat-Dog \
#   --pseudo_labels 0 \
#   --use_data_parallel

echo "# Kaggle-Cat-Dog"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id resnet18_pure_victim_normalized \
  --lr_mode 5 \
  --lr_max 0.03 \
  --opt_type Adam \
  --epochs 50 \
  --dataset Kaggle-Cat-Dog \
  --pseudo_labels 0 \
  --use_data_parallel