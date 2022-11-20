#!/bin/sh
# nohup ./script/run_32.sh &> ./log/run_32.txt &
# ! run in 4 TITAN GPUs
# using ting/diff-normalization branch
# * Feature extraction model with different normalization technique

echo "# RAND"
echo "# data-normalization"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-normalization \
  --normalization_type data-normalization

echo "# normalization-without-mean"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-normalization \
  --normalization_type normalization-without-mean

echo "# normalization-without-std"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-normalization \
  --normalization_type normalization-without-std

echo "# rgb-grayscale"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-normalization \
  --normalization_type rgb-grayscale

echo "# min-max--1-and-1"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-normalization \
  --normalization_type min-max--1-and-1
