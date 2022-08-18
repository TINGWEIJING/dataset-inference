#!/bin/sh
export CUDA_VISIBLE_DEVICES=3

# Case: Feature generation CIFAR & CINIC ratio combination for mingd

echo "# CIFAR-CINIC-100-0"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR-CINIC-100-0 \
  --victim_dataset CIFAR10 \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 500 \
  --model_id wrn-28-10_normalized

echo "# CIFAR-CINIC-90-10"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR-CINIC-90-10 \
  --victim_dataset CIFAR10 \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 500 \
  --model_id wrn-28-10_normalized

echo "# CIFAR-CINIC-80-20"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR-CINIC-80-20 \
  --victim_dataset CIFAR10 \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 500 \
  --model_id wrn-28-10_normalized

echo "# CIFAR-CINIC-60-40"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR-CINIC-60-40 \
  --victim_dataset CIFAR10 \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 500 \
  --model_id wrn-28-10_normalized

echo "# CIFAR-CINIC-40-60"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR-CINIC-40-60 \
  --victim_dataset CIFAR10 \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 500 \
  --model_id wrn-28-10_normalized

echo "# CIFAR-CINIC-20-80"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR-CINIC-20-80 \
  --victim_dataset CIFAR10 \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 500 \
  --model_id wrn-28-10_normalized

echo "# CIFAR-CINIC-10-90"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR-CINIC-10-90 \
  --victim_dataset CIFAR10 \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 500 \
  --model_id wrn-28-10_normalized

echo "# CIFAR-CINIC-0-100"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR-CINIC-0-100 \
  --victim_dataset CIFAR10 \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 500 \
  --model_id wrn-28-10_normalized