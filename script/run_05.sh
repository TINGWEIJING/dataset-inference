#!/bin/sh
# nohup ./script/run_05.sh &> ./log/run_05.txt &
# run for mingd & rand for unrelated dataset svhn
export CUDA_VISIBLE_DEVICES=2,3

echo $CUDA_VISIBLE_DEVICES

echo "Mingd SVHN"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --model_dataset SVHN \
  --batch_size 250 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment unrelated-dataset

echo "Rand SVHN"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --model_dataset SVHN \
  --batch_size 250 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment unrelated-dataset
