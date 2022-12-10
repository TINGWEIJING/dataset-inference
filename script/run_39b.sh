#!/bin/sh
# nohup ./script/run_39b.sh &> ./log/run_39b.txt &
# ! run in 4 TITAN GPUs
# using ting/diff-normalization branch
# * Model training with different mean & w/wo std

echo "# Default"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode teacher \
  --normalize 0 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment diff-norm-value \
  --num_workers 16

echo "# Mingd"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --load_model_normalize 1

echo "# Rand"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --load_model_normalize 1
