#!/bin/sh
# nohup ./script/run_47.sh &> ./log/run_47.txt &
# ! run in DICC v100
# using ting/diff-normalization-cossim-02 branch
# * Model training with diff mean normalization

echo "# Mean: -0.477  0.49  -0.996"
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
  --normalization_mean -0.477  0.49  -0.996 \
  --num_workers 16

echo "# Mean: -0.767 -0.475  0.515"
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
  --normalization_mean -0.767 -0.475  0.515 \
  --num_workers 16

echo "# Mean: -0.261 -0.614  0.349"
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
  --normalization_mean -0.261 -0.614  0.349 \
  --num_workers 16

echo "# Mean:  0.369 -0.311 -0.81"
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
  --normalization_mean  0.369 -0.311 -0.81 \
  --num_workers 16

echo "# Mean: -0.574 -0.897  0.64 "
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
  --normalization_mean -0.574 -0.897  0.64  \
  --num_workers 16

echo "# Mean: -0.937  0.102  0.19 "
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
  --normalization_mean -0.937  0.102  0.19  \
  --num_workers 16

echo "# Mean: -0.767  0.2    0.033"
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
  --normalization_mean -0.767  0.2    0.033 \
  --num_workers 16

echo "# Mean: -0.426  0.063  0.079"
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
  --normalization_mean -0.426  0.063  0.079 \
  --num_workers 16

echo "# Mean: -0.462 -0.85   0.601"
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
  --normalization_mean -0.462 -0.85   0.601 \
  --num_workers 16

echo "# Mean: -0.56   0.306 -0.188"
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
  --normalization_mean -0.56   0.306 -0.188 \
  --num_workers 16
