#!/bin/sh
# nohup ./script/run_40.sh &> ./log/run_40.txt &
# ! run in 4 TITAN GPUs
# using ting/diff-normalization branch
# * Mingd & Rand feature extraction for model training with different mean & w/wo std

echo "# MINGD"
echo "# Mean: 0.792 0.415 0.84"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.792 0.415 0.84 \
  --load_model_normalize 1

echo "# Mean: 0.852 0.402 0.919"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.852 0.402 0.919 \
  --load_model_normalize 1

echo "# Mean: 0.913 0.388 0.998"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.913 0.388 0.998 \
  --load_model_normalize 1

echo "# Mean: 0.642 0.173 0.809"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.642 0.173 0.809 \
  --load_model_normalize 1

echo "# Mean: 0.672 0.111 0.882"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.672 0.111 0.882 \
  --load_model_normalize 1

echo "# Mean: 0.702 0.049 0.954"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.702 0.049 0.954 \
  --load_model_normalize 1

echo "# Mean: 0.724 0.265 0.832"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.724 0.265 0.832 \
  --load_model_normalize 1

echo "# Mean: 0.77  0.221 0.909"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.77  0.221 0.909 \
  --load_model_normalize 1

echo "# Mean: 0.817 0.178 0.986"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.817 0.178 0.986 \
  --load_model_normalize 1

# ============================= Std: 1

echo "# Mean: 0.792 0.415 0.84 Std: 1"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.792 0.415 0.84 \
  --normalization_std 1 1 1 \
  --load_model_normalize 1

echo "# Mean: 0.852 0.402 0.919 Std: 1"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.852 0.402 0.919 \
  --normalization_std 1 1 1 \
  --load_model_normalize 1

echo "# Mean: 0.913 0.388 0.998 Std: 1"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.913 0.388 0.998 \
  --normalization_std 1 1 1 \
  --load_model_normalize 1

echo "# Mean: 0.642 0.173 0.809 Std: 1"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.642 0.173 0.809 \
  --normalization_std 1 1 1 \
  --load_model_normalize 1

echo "# Mean: 0.672 0.111 0.882 Std: 1"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.672 0.111 0.882 \
  --normalization_std 1 1 1 \
  --load_model_normalize 1

echo "# Mean: 0.702 0.049 0.954 Std: 1"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.702 0.049 0.954 \
  --normalization_std 1 1 1 \
  --load_model_normalize 1

echo "# Mean: 0.724 0.265 0.832 Std: 1"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.724 0.265 0.832 \
  --normalization_std 1 1 1 \
  --load_model_normalize 1

echo "# Mean: 0.77  0.221 0.909 Std: 1"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.77  0.221 0.909 \
  --normalization_std 1 1 1 \
  --load_model_normalize 1

echo "# Mean: 0.817 0.178 0.986 Std: 1"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.817 0.178 0.986 \
  --normalization_std 1 1 1 \
  --load_model_normalize 1

echo "# RAND"
echo "# Mean: 0.792 0.415 0.84"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.792 0.415 0.84 \
  --load_model_normalize 1

echo "# Mean: 0.852 0.402 0.919"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.852 0.402 0.919 \
  --load_model_normalize 1

echo "# Mean: 0.913 0.388 0.998"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.913 0.388 0.998 \
  --load_model_normalize 1

echo "# Mean: 0.642 0.173 0.809"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.642 0.173 0.809 \
  --load_model_normalize 1

echo "# Mean: 0.672 0.111 0.882"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.672 0.111 0.882 \
  --load_model_normalize 1

echo "# Mean: 0.702 0.049 0.954"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.702 0.049 0.954 \
  --load_model_normalize 1

echo "# Mean: 0.724 0.265 0.832"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.724 0.265 0.832 \
  --load_model_normalize 1

echo "# Mean: 0.77  0.221 0.909"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.77  0.221 0.909 \
  --load_model_normalize 1

echo "# Mean: 0.817 0.178 0.986"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.817 0.178 0.986 \
  --load_model_normalize 1

# ============================= Std: 1

echo "# Mean: 0.792 0.415 0.84 Std: 1"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.792 0.415 0.84 \
  --normalization_std 1 1 1 \
  --load_model_normalize 1

echo "# Mean: 0.852 0.402 0.919 Std: 1"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.852 0.402 0.919 \
  --normalization_std 1 1 1 \
  --load_model_normalize 1

echo "# Mean: 0.913 0.388 0.998 Std: 1"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.913 0.388 0.998 \
  --normalization_std 1 1 1 \
  --load_model_normalize 1

echo "# Mean: 0.642 0.173 0.809 Std: 1"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.642 0.173 0.809 \
  --normalization_std 1 1 1 \
  --load_model_normalize 1

echo "# Mean: 0.672 0.111 0.882 Std: 1"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.672 0.111 0.882 \
  --normalization_std 1 1 1 \
  --load_model_normalize 1

echo "# Mean: 0.702 0.049 0.954 Std: 1"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.702 0.049 0.954 \
  --normalization_std 1 1 1 \
  --load_model_normalize 1

echo "# Mean: 0.724 0.265 0.832 Std: 1"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.724 0.265 0.832 \
  --normalization_std 1 1 1 \
  --load_model_normalize 1

echo "# Mean: 0.77  0.221 0.909 Std: 1"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.77  0.221 0.909 \
  --normalization_std 1 1 1 \
  --load_model_normalize 1

echo "# Mean: 0.817 0.178 0.986 Std: 1"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.817 0.178 0.986 \
  --normalization_std 1 1 1 \
  --load_model_normalize 1
