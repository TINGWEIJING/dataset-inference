#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1

# echo $CUDA_VISIBLE_DEVICES
# echo "# CIFAR10-Cat-Dog"
# python3 ./src/train.py \
#   --batch_size 512 \
#   --mode teacher \
#   --normalize 1 \
#   --model_id teacher_normalized \
#   --lr_mode 2 \
#   --epochs 50 \
#   --dataset CIFAR10-Cat-Dog \
#   --dropRate 0.3 \
#   --pseudo_labels 0 \
#   --download_dataset

echo "# A.2 'fine-tune' | Fine tuning"
python3 ./src/train.py \
  --batch_size 512 \
  --mode fine-tune \
  --lr_max 0.01 \
  --normalize 1 \
  --model_id fine-tune_normalized \
  --pseudo_labels 0 \
  --lr_mode 2 \
  --epochs 5 \
  --dataset CIFAR10-Cat-Dog \
  --download_dataset

echo "# Kaggle-Cat-Dog"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10-Cat-Dog \
  --dropRate 0.3 \
  --pseudo_labels 0

echo "# CIFAR10-STL10-Cat-Dog"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10-Cat-Dog \
  --dropRate 0.3 \

echo "# CIFAR10-Kaggle-Cat-Dog"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10-Cat-Dog \
  --dropRate 0.3 \
  --pseudo_labels 0

echo "# STL10-Kaggle-Cat-Dog"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10-Cat-Dog \
  --dropRate 0.3 \
  --pseudo_labels 0