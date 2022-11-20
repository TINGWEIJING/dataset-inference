#!/bin/sh
# nohup ./script/run_29.sh &> ./log/run_29.txt &
# ! run in 4 TITAN GPUs
# using ting/diff-normalization branch
# * Train model with different normalization technique

echo "# data-normalization"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode teacher \
  --normalize 0 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment diff-normalization \
  --normalization_type data-normalization \
  --num_workers 16

echo "# normalization-without-mean"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode teacher \
  --normalize 0 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment diff-normalization \
  --normalization_type normalization-without-mean \
  --num_workers 16

echo "# normalization-without-std"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode teacher \
  --normalize 0 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment diff-normalization \
  --normalization_type normalization-without-std \
  --num_workers 16

echo "# rgb-grayscale"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode teacher \
  --normalize 0 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment diff-normalization \
  --normalization_type rgb-grayscale \
  --num_workers 16

echo "# min-max--1-and-1"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode teacher \
  --normalize 0 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment diff-normalization \
  --normalization_type min-max--1-and-1 \
  --num_workers 16
