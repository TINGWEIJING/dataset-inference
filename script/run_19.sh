#!/bin/sh
# nohup ./script/run_19.sh &> ./log/run_19.txt &
# ! run in DICC v100
# using ting/cifar-cinic branch
# * Train Teacher model with 3 variables (Epoch, Batch size & Acc), max epoch 500, batch size 1000, 500, 250, 125

echo "# Teacher Batch Size 1000, epoch 500"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 500 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment 3-var