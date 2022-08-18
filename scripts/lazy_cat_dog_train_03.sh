#!/bin/sh
export CUDA_VISIBLE_DEVICES=1,2

# echo "Teacher"
# python3 ./src/train.py \
#   --batch_size 512 \
#   --mode teacher \
#   --normalize 1 \
#   --model_id teacher_normalized \
#   --lr_mode 2 \
#   --epochs 100 \
#   --dataset CIFAR10 \
#   --dropRate 0.3 \
#   --use_data_parallel

# echo "Independent"
# python3 ./src/train.py \
#   --batch_size 512 \
#   --mode independent \
#   --normalize 1 \
#   --model_id independent_normalized \
#   --lr_mode 2 \
#   --epochs 50 \
#   --dataset CIFAR10 \
#   --use_data_parallel

echo "# C.1 'distillation' | Data distillation"
python3 ./src/train.py \
  --batch_size 512 \
  --mode distillation \
  --normalize 1 \
  --model_id distillation_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10 \
  --use_data_parallel