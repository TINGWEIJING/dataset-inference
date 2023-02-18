#!/bin/sh
# nohup ./script/run_51.sh &> ./log/run_51.txt &
# ! run in DICC v100
# using ting/diff-norm-value-cossim-01-fp branch
# * Model training with diff mean normalization with false positive cases

echo "# Mean: 0.365 -0.892 -0.559"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode independent \
  --normalize 0 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment diff-norm-value \
  --normalization_mean 0.365 -0.892 -0.559 \
  --num_workers 16

echo "# Mean: -0.631 -0.648  0.624"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode independent \
  --normalize 0 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment diff-norm-value \
  --normalization_mean -0.631 -0.648  0.624 \
  --num_workers 16

echo "# Mean: 0.847 -0.447  0.64"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode independent \
  --normalize 0 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment diff-norm-value \
  --normalization_mean 0.847 -0.447  0.64 \
  --num_workers 16

echo "# Mean: 0.78   0.026 -0.51"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode independent \
  --normalize 0 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment diff-norm-value \
  --normalization_mean 0.78   0.026 -0.51 \
  --num_workers 16

echo "# Mean: 0.648 -0.572  0.483"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode independent \
  --normalize 0 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment diff-norm-value \
  --normalization_mean 0.648 -0.572  0.483 \
  --num_workers 16

echo "# Mean: 0.26   0.855 -0.536"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode independent \
  --normalize 0 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment diff-norm-value \
  --normalization_mean 0.26   0.855 -0.536 \
  --num_workers 16

echo "# Mean: 0.598  0.036 -0.537"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode independent \
  --normalize 0 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment diff-norm-value \
  --normalization_mean 0.598  0.036 -0.537 \
  --num_workers 16

echo "# Mean: -0.668 -0.004  0.165"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode independent \
  --normalize 0 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment diff-norm-value \
  --normalization_mean -0.668 -0.004  0.165 \
  --num_workers 16

echo "# Mean: -0.631 -0.97  -0.058"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode independent \
  --normalize 0 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment diff-norm-value \
  --normalization_mean -0.631 -0.97  -0.058 \
  --num_workers 16

echo "# Mean: 0.456 0.837 0.251"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode independent \
  --normalize 0 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment diff-norm-value \
  --normalization_mean 0.456 0.837 0.251 \
  --num_workers 16