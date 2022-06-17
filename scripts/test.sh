#!/bin/sh

echo "# CIFAR10-Cat-Dog"
python ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10-Cat-Dog \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --download_dataset