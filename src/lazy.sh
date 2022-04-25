#!/bin/sh
echo "# B.1 'extract-label' | Model extraction using unlabeled data and victim labels"
python3 train.py \
  --batch_size 64 \
  --mode extract-label \
  --normalize 1 \
  --model_id extract-label_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 15 \
  --dataset CIFAR10

echo "# B.2 'extract-logit' | Model extraction using unlabeled data and victim confidence"
python3 train.py \
  --batch_size 64 \
  --mode extract-logit \
  --normalize 1 \
  --model_id extract_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 15 \
  --dataset CIFAR10

echo "# C.1 'distillation' | Data distillation"
python3 train.py \
  --batch_size 64 \
  --mode distillation \
  --normalize 1 \
  --model_id distillation_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10

echo "# C.2 'pre-act-18' | Different architecture/learning rate/optimizer/training epochs"
python3 train.py \
  --batch_size 64 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10