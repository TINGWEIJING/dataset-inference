#!/bin/sh
# nohup ./script/run_41.sh &> ./log/run_41.txt &
# ! run in DICC v100
# using ting/diff-normalization-dist branch
# * Mingd & Rand feature extraction for model training with different mean & w/wo std (continue)

echo "# MINGD"
echo "# Mean: 0.547 0.539 0.507"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.547 0.539 0.507 \
  --load_model_normalize 1

echo "# Mean: 0.603 0.596 0.567"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.603 0.596 0.567 \
  --load_model_normalize 1

echo "# Mean: 0.659 0.652 0.628"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.659 0.652 0.628 \
  --load_model_normalize 1

echo "# Mean: 0.715 0.709 0.688"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.715 0.709 0.688 \
  --load_model_normalize 1

echo "# Mean: 0.771 0.766 0.748"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.771 0.766 0.748 \
  --load_model_normalize 1

echo "# Mean: 0.799 0.794 0.779"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.799 0.794 0.779 \
  --load_model_normalize 1

echo "# Mean: 0.827 0.823 0.809"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.827 0.823 0.809 \
  --load_model_normalize 1

echo "# Mean: 0.855 0.851 0.839"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.855 0.851 0.839 \
  --load_model_normalize 1

echo "# Mean: 0.883 0.879 0.869"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.883 0.879 0.869 \
  --load_model_normalize 1

echo "# Mean: 0.911 0.908 0.899"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.911 0.908 0.899 \
  --load_model_normalize 1

echo "# Mean: 0.939 0.936 0.93"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.939 0.936 0.93 \
  --load_model_normalize 1

echo "# Mean: 0.967 0.965 0.96"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.967 0.965 0.96 \
  --load_model_normalize 1

echo "# Mean: 0.995 0.993 0.99"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.995 0.993 0.99 \
  --load_model_normalize 1

echo "# RAND"
echo "# Mean: 0.547 0.539 0.507"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.547 0.539 0.507 \
  --load_model_normalize 1

echo "# Mean: 0.603 0.596 0.567"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.603 0.596 0.567 \
  --load_model_normalize 1

echo "# Mean: 0.659 0.652 0.628"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.659 0.652 0.628 \
  --load_model_normalize 1

echo "# Mean: 0.715 0.709 0.688"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.715 0.709 0.688 \
  --load_model_normalize 1

echo "# Mean: 0.771 0.766 0.748"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.771 0.766 0.748 \
  --load_model_normalize 1

echo "# Mean: 0.799 0.794 0.779"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.799 0.794 0.779 \
  --load_model_normalize 1

echo "# Mean: 0.827 0.823 0.809"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.827 0.823 0.809 \
  --load_model_normalize 1

echo "# Mean: 0.855 0.851 0.839"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.855 0.851 0.839 \
  --load_model_normalize 1

echo "# Mean: 0.883 0.879 0.869"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.883 0.879 0.869 \
  --load_model_normalize 1

echo "# Mean: 0.911 0.908 0.899"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.911 0.908 0.899 \
  --load_model_normalize 1

echo "# Mean: 0.939 0.936 0.93"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.939 0.936 0.93 \
  --load_model_normalize 1

echo "# Mean: 0.967 0.965 0.96"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.967 0.965 0.96 \
  --load_model_normalize 1

echo "# Mean: 0.995 0.993 0.99"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean 0.995 0.993 0.99 \
  --load_model_normalize 1
