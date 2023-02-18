#!/bin/sh
# nohup ./script/run_52.sh &> ./log/run_52.txt &
# ! run in DICC v100
# using ting/diff-norm-value-cossim-01-fp branch
# * Mingd & Rand feature extraction for model training with diff mean normalization with false positive cases

echo "# MINGD"
echo "# Mean: 0.365 -0.892 -0.559"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.365 -0.892 -0.559 \
  --load_model_normalize 1

echo "# Mean: -0.631 -0.648  0.624"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.631 -0.648  0.624 \
  --load_model_normalize 1

echo "# Mean: 0.847 -0.447  0.64"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.847 -0.447  0.64 \
  --load_model_normalize 1

echo "# Mean: 0.78   0.026 -0.51"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.78   0.026 -0.51 \
  --load_model_normalize 1

echo "# Mean: 0.648 -0.572  0.483"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.648 -0.572  0.483 \
  --load_model_normalize 1

echo "# Mean: 0.26   0.855 -0.536"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.26   0.855 -0.536 \
  --load_model_normalize 1

echo "# Mean: 0.598  0.036 -0.537"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.598  0.036 -0.537 \
  --load_model_normalize 1

echo "# Mean: -0.668 -0.004  0.165"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.668 -0.004  0.165 \
  --load_model_normalize 1

echo "# Mean: -0.631 -0.97  -0.058"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.631 -0.97  -0.058 \
  --load_model_normalize 1

echo "# Mean: 0.456 0.837 0.251"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.456 0.837 0.251 \
  --load_model_normalize 1

echo "# RAND"
echo "# Mean: 0.365 -0.892 -0.559"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.365 -0.892 -0.559 \
  --load_model_normalize 1

echo "# Mean: -0.631 -0.648  0.624"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.631 -0.648  0.624 \
  --load_model_normalize 1

echo "# Mean: 0.847 -0.447  0.64"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.847 -0.447  0.64 \
  --load_model_normalize 1

echo "# Mean: 0.78   0.026 -0.51"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.78   0.026 -0.51 \
  --load_model_normalize 1

echo "# Mean: 0.648 -0.572  0.483"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.648 -0.572  0.483 \
  --load_model_normalize 1

echo "# Mean: 0.26   0.855 -0.536"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.26   0.855 -0.536 \
  --load_model_normalize 1

echo "# Mean: 0.598  0.036 -0.537"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.598  0.036 -0.537 \
  --load_model_normalize 1

echo "# Mean: -0.668 -0.004  0.165"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.668 -0.004  0.165 \
  --load_model_normalize 1

echo "# Mean: -0.631 -0.97  -0.058"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.631 -0.97  -0.058 \
  --load_model_normalize 1

echo "# Mean: 0.456 0.837 0.251"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.456 0.837 0.251 \
  --load_model_normalize 1
