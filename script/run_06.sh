#!/bin/sh
# nohup ./script/run_06.sh &> ./log/run_06.txt &
# Run 2 SSIM CIFAR10 model training
export CUDA_VISIBLE_DEVICES=2,3

echo $CUDA_VISIBLE_DEVICES

echo "# Noise Sigma: 0.17 SSIM: 0.508"
python3 ./src/train.py \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment ssim-cifar10 \
  --noise_sigma 0.17

echo "# Noise Sigma: 0.09 SSIM: 0.745"
python3 ./src/train.py \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment ssim-cifar10 \
  --noise_sigma 0.09

