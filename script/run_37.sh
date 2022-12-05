#!/bin/sh
# nohup ./script/run_37.sh &> ./log/run_37.txt &
# ! run in 4 TITAN GPUs
# using ting/diff-normalization branch
# * Model training with different mean & std normalization value + rgb to grayscale before normalization

echo "# Mean Std (Baseline)"
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
  --extra_preprocessing_type rgb-grayscale \
  --num_workers 16

echo "# Mean: 0.774 0.439 0.859"
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
  --normalization_mean 0.774 0.439 0.859 \
  --extra_preprocessing_type rgb-grayscale \
  --num_workers 16

echo "# Mean: 0.697 0.094 0.976"
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
  --normalization_mean 0.697 0.094 0.976 \
  --extra_preprocessing_type rgb-grayscale \
  --num_workers 16

echo "# Mean: 0.761 0.786 0.128"
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
  --normalization_mean 0.761 0.786 0.128 \
  --extra_preprocessing_type rgb-grayscale \
  --num_workers 16

echo "# Std: 0.3471 0.3435 0.3616"
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
  --normalization_std 0.3471 0.3435 0.3616 \
  --extra_preprocessing_type rgb-grayscale \
  --num_workers 16

echo "# Std: 0.2971 0.2935 0.3116"
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
  --normalization_std 0.2971 0.2935 0.3116 \
  --extra_preprocessing_type rgb-grayscale \
  --num_workers 16

echo "# Std: 0.2721 0.2685 0.2866"
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
  --normalization_std 0.2721 0.2685 0.2866 \
  --extra_preprocessing_type rgb-grayscale \
  --num_workers 16

echo "# Std: 0.1471 0.1435 0.1616"
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
  --normalization_std 0.1471 0.1435 0.1616 \
  --extra_preprocessing_type rgb-grayscale \
  --num_workers 16

echo "# Std: 0.1971 0.1935 0.2116"
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
  --normalization_std 0.1971 0.1935 0.2116 \
  --extra_preprocessing_type rgb-grayscale \
  --num_workers 16

echo "# Std: 0.2221 0.2185 0.2366"
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
  --normalization_std 0.2221 0.2185 0.2366 \
  --extra_preprocessing_type rgb-grayscale \
  --num_workers 16