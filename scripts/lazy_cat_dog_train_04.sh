#!/bin/sh
export CUDA_VISIBLE_DEVICES=1,2

# Case: Train all teacher model using ResNet18 with different combination of cat dog dataset

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

echo "# STL10-Cat-Dog"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id resnet18_victim_normalized \
  --lr_mode 5 \
  --lr_max 0.03 \
  --opt_type Adam \
  --epochs 200 \
  --dataset STL10-Cat-Dog \
  --pseudo_labels 0 \
  --use_data_parallel

echo "# Kaggle-Cat-Dog"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id resnet18_victim_normalized \
  --lr_mode 5 \
  --lr_max 0.03 \
  --opt_type Adam \
  --epochs 50 \
  --dataset Kaggle-Cat-Dog \
  --pseudo_labels 0 \
  --use_data_parallel

echo "# CIFAR10-STL10-Cat-Dog"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id resnet18_victim_normalized \
  --lr_mode 5 \
  --lr_max 0.03 \
  --opt_type Adam \
  --epochs 100 \
  --dataset CIFAR10-STL10-Cat-Dog \
  --pseudo_labels 0 \
  --use_data_parallel

echo "# CIFAR10-Kaggle-Cat-Dog"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id resnet18_victim_normalized \
  --lr_mode 5 \
  --lr_max 0.03 \
  --opt_type Adam \
  --epochs 50 \
  --dataset CIFAR10-Kaggle-Cat-Dog \
  --pseudo_labels 0 \
  --use_data_parallel

echo "# STL10-Kaggle-Cat-Dog"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id resnet18_victim_normalized \
  --lr_mode 5 \
  --lr_max 0.03 \
  --opt_type Adam \
  --epochs 50 \
  --dataset STL10-Kaggle-Cat-Dog \
  --pseudo_labels 0 \
  --use_data_parallel

echo "# CIFAR10-STL10-Kaggle-Cat-Dog"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id resnet18_victim_normalized \
  --lr_mode 5 \
  --lr_max 0.03 \
  --opt_type Adam \
  --epochs 50 \
  --dataset CIFAR10-STL10-Kaggle-Cat-Dog \
  --pseudo_labels 0 \
  --use_data_parallel