#!/bin/sh
# nohup ./script/run_07.sh &> ./log/run_07.txt &
# run for rand for SSIM CIFAR10 experiment
export CUDA_VISIBLE_DEVICES=2,3

echo $CUDA_VISIBLE_DEVICES

echo "# Noise Sigma: 0.17 SSIM: 0.508"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --model_dataset CIFAR10 \
  --batch_size 250 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment ssim-cifar10 \
  --noise_sigma 0.17

echo "# Noise Sigma: 0.09 SSIM: 0.745"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --model_dataset CIFAR10 \
  --batch_size 250 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment ssim-cifar10 \
  --noise_sigma 0.09

echo "# Noise Sigma: 0.4 SSIM: 0.249"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --model_dataset CIFAR10 \
  --batch_size 250 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment ssim-cifar10 \
  --noise_sigma 0.4

echo "# Noise Sigma: 0.34 SSIM: 0.301"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --model_dataset CIFAR10 \
  --batch_size 250 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment ssim-cifar10 \
  --noise_sigma 0.34