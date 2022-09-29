#!/bin/sh
# nohup ./script/run_11.sh &> ./log/run_11.txt &
# ! run in DICC v100
# using ting/reproduce branch
# * Retry Train fine tuning model with author settings, batch size 100, retrain independent use 500 epoch

echo "# 'independent'"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode independent \
  --normalize 1 \
  --model_id independent_normalized \
  --lr_mode 2 \
  --epochs 500 \
  --dataset CIFAR10

echo "# A.2 'fine-tune' | Fine tuning"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode fine-tune \
  --lr_max 0.01 \
  --normalize 1 \
  --model_id fine-tune_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 5 \
  --dataset CIFAR10
