#!/bin/sh
# export CUDA_VISIBLE_DEVICES=1,2

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

echo "# STL10-Cat-Dog"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset STL10-Cat-Dog \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --download_dataset

echo "# Kaggle-Cat-Dog"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset Kaggle-Cat-Dog \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --download_dataset

echo "# CIFAR10-STL10-Cat-Dog"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10-STL10-Cat-Dog \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --download_dataset

echo "# CIFAR10-Kaggle-Cat-Dog"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10-Kaggle-Cat-Dog \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --download_dataset

echo "# STL10-Kaggle-Cat-Dog"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset STL10-Kaggle-Cat-Dog \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --download_dataset

echo "# CIFAR10-STL10-Kaggle-Cat-Dog"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10-STL10-Kaggle-Cat-Dog \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --download_dataset