#!/bin/sh
# nohup ./script/run_33.sh &> ./log/run_33.txt &
# ! run in 4 TITAN GPUs
# using ting/diff-normalization branch
# * Feature extraction model with different normalization technique

echo "# MINGD"
echo "# data-normalization"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-normalization \
  --normalization_type data-normalization \
  --data_normalize 0 \
  --load_model_normalize 1

echo "# normalization-without-mean"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-normalization \
  --normalization_type normalization-without-mean \
  --data_normalize 0 \
  --load_model_normalize 1

echo "# normalization-without-std"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-normalization \
  --normalization_type normalization-without-std \
  --data_normalize 0 \
  --load_model_normalize 1

echo "# rgb-grayscale"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-normalization \
  --normalization_type rgb-grayscale \
  --data_normalize 0 \
  --load_model_normalize 1

echo "# min-max--1-and-1"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-normalization \
  --normalization_type min-max--1-and-1 \
  --data_normalize 0 \
  --load_model_normalize 1

echo "# RAND"
echo "# data-normalization"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-normalization \
  --normalization_type data-normalization \
  --data_normalize 0 \
  --load_model_normalize 1

echo "# normalization-without-mean"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-normalization \
  --normalization_type normalization-without-mean \
  --data_normalize 0 \
  --load_model_normalize 1

echo "# normalization-without-std"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-normalization \
  --normalization_type normalization-without-std \
  --data_normalize 0 \
  --load_model_normalize 1

echo "# rgb-grayscale"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-normalization \
  --normalization_type rgb-grayscale \
  --data_normalize 0 \
  --load_model_normalize 1

echo "# min-max--1-and-1"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-normalization \
  --normalization_type min-max--1-and-1 \
  --data_normalize 0 \
  --load_model_normalize 1
