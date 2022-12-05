#!/bin/sh
# nohup ./script/run_36.sh &> ./log/run_36.txt &
# ! run in 4 TITAN GPUs
# using ting/diff-normalization branch
# * Mingd & Rand feature extraction for model training with different mean & std normalization value

echo "# MINGD"
echo "# Mean Std (Baseline)"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --load_model_normalize 1

echo "# Mean: 0.774 0.439 0.859"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.774 0.439 0.859 \
  --load_model_normalize 1

echo "# Mean: 0.697 0.094 0.976"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.697 0.094 0.976 \
  --load_model_normalize 1

echo "# Mean: 0.761 0.786 0.128"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.761 0.786 0.128 \
  --load_model_normalize 1

echo "# Std: 0.3471 0.3435 0.3616"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_std 0.3471 0.3435 0.3616 \
  --load_model_normalize 1

echo "# Std: 0.2971 0.2935 0.3116"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_std 0.2971 0.2935 0.3116 \
  --load_model_normalize 1

echo "# Std: 0.2721 0.2685 0.2866"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_std 0.2721 0.2685 0.2866 \
  --load_model_normalize 1

echo "# Std: 0.1471 0.1435 0.1616"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_std 0.1471 0.1435 0.1616 \
  --load_model_normalize 1

echo "# Std: 0.1971 0.1935 0.2116"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_std 0.1971 0.1935 0.2116 \
  --load_model_normalize 1

echo "# Std: 0.2221 0.2185 0.2366"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_std 0.2221 0.2185 0.2366 \
  --load_model_normalize 1

echo "# RAND"
echo "# Mean Std (Baseline)"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --load_model_normalize 1

echo "# Mean: 0.774 0.439 0.859"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.774 0.439 0.859 \
  --load_model_normalize 1

echo "# Mean: 0.697 0.094 0.976"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.697 0.094 0.976 \
  --load_model_normalize 1

echo "# Mean: 0.761 0.786 0.128"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.761 0.786 0.128 \
  --load_model_normalize 1

echo "# Std: 0.3471 0.3435 0.3616"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_std 0.3471 0.3435 0.3616 \
  --load_model_normalize 1

echo "# Std: 0.2971 0.2935 0.3116"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_std 0.2971 0.2935 0.3116 \
  --load_model_normalize 1

echo "# Std: 0.2721 0.2685 0.2866"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_std 0.2721 0.2685 0.2866 \
  --load_model_normalize 1

echo "# Std: 0.1471 0.1435 0.1616"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_std 0.1471 0.1435 0.1616 \
  --load_model_normalize 1

echo "# Std: 0.1971 0.1935 0.2116"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_std 0.1971 0.1935 0.2116 \
  --load_model_normalize 1

echo "# Std: 0.2221 0.2185 0.2366"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_std 0.2221 0.2185 0.2366 \
  --load_model_normalize 1