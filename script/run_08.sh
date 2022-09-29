#!/bin/sh
# nohup ./script/run_08.sh &> ./log/run_08.txt &
# ! run in DICC v100
# using ting/reproduce branch
# * Train model with author settings, batch size 100

echo "# Teacher/Source/Victim model"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3

echo "# 'independent'"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode independent \
  --normalize 1 \
  --model_id independent_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10