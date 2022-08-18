#!/bin/sh
export CUDA_VISIBLE_DEVICES=1,3

# Case: Train CIFAR & CINIC ratio combination

# echo "# CIFAR10-Cat-Dog"
# python3 ./src/train.py \
#   --batch_size 512 \
#   --mode teacher \
#   --normalize 1 \
#   --model_id resnet18_victim_normalized \
#   --lr_mode 5 \
#   --lr_max 0.03 \
#   --opt_type Adam \
#   --epochs 100 \
#   --dataset CIFAR10-Cat-Dog \
#   --pseudo_labels 0 \
#   --use_data_parallel

# python3 ./src/train.py \
#   --batch_size 512 \
#   --mode teacher \
#   --normalize 1 \
#   --model_id wrn-28-10_normalized \
#   --lr_mode 5 \
#   --lr_max 0.03 \
#   --opt_type Adam \
#   --epochs 50 \
#   --dataset CIFAR-CINIC-100-0 \
#   --pseudo_labels 0 \
#   --use_data_parallel

echo "# CIFAR-CINIC-100-0"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id wrn-28-10_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR-CINIC-100-0 \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --use_data_parallel

echo "# CIFAR-CINIC-90-10"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id wrn-28-10_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR-CINIC-90-10 \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --use_data_parallel

echo "# CIFAR-CINIC-80-20"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id wrn-28-10_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR-CINIC-80-20 \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --use_data_parallel

echo "# CIFAR-CINIC-60-40"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id wrn-28-10_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR-CINIC-60-40 \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --use_data_parallel

echo "# CIFAR-CINIC-40-60"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id wrn-28-10_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR-CINIC-40-60 \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --use_data_parallel

echo "# CIFAR-CINIC-20-80"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id wrn-28-10_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR-CINIC-20-80 \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --use_data_parallel

echo "# CIFAR-CINIC-10-90"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id wrn-28-10_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR-CINIC-10-90 \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --use_data_parallel

echo "# CIFAR-CINIC-0-100"
python3 ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id wrn-28-10_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR-CINIC-0-100 \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --use_data_parallel