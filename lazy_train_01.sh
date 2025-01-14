#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo $CUDA_VISIBLE_DEVICES
# echo "Independent"

# python3 ./src/train.py \
#   --gpu_id 0 \
#   --batch_size 1000 \
#   --mode independent \
#   --normalize 1 \
#   --model_id independent_normalized \
#   --lr_mode 2 \
#   --epochs 50 \
#   --dataset CIFAR10

echo "# A.2 'fine-tune' | Fine tuning"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode fine-tune \
  --lr_max 0.01 \
  --normalize 1 \
  --model_id fine-tune_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 5 \
  --dataset CIFAR10


echo "# B.1 'extract-label' | Model extraction"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode extract-label \
  --normalize 1 \
  --model_id extract-label_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 20 \
  --dataset CIFAR10


echo "# B.2 'extract-logit' | Model extraction"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode extract-logit \
  --normalize 1 \
  --model_id extract-logit_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 20 \
  --dataset CIFAR10


echo "# C.1 'distillation' | Data distillation"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode distillation \
  --normalize 1 \
  --model_id distillation_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10


echo "# C.2 'pre-act-18' | Different architecture"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10