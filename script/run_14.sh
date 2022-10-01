#!/bin/sh
# nohup ./script/run_14.sh &> ./log/run_14.txt &
# ! run in DICC v100
# using ting/cifar-cinic branch
# * Train model with diff ratio combination CIFAR10 CINIC10 EXCL, batch size 1000 (continue with 0.3)

echo "# Ratio 0.3"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10-CINIC10-EXCL \
  --dropRate 0.3 \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0.3

echo "# Ratio 0.2"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10-CINIC10-EXCL \
  --dropRate 0.3 \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0.2

echo "# Ratio 0.1"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10-CINIC10-EXCL \
  --dropRate 0.3 \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0.1

echo "# Ratio 0"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10-CINIC10-EXCL \
  --dropRate 0.3 \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0