#!/bin/sh
# nohup ./script/run_17.sh &> ./log/run_17.txt &
# ! run in DICC v100
# using ting/cifar-cinic branch
# * Train Teacher model with 3 variables (Epoch, Batch size & Acc), max epoch 500, batch size 1000, 500, 250, 125

echo "# Teacher Batch Size 500, epoch 500"
python3 ./src/train.py \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 500 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment 3-var

echo "# Teacher Batch Size 250, epoch 500"
python3 ./src/train.py \
  --batch_size 250 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 500 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment 3-var

echo "# Teacher Batch Size 125, epoch 500"
python3 ./src/train.py \
  --batch_size 125 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 500 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment 3-var
