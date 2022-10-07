#!/bin/sh
# nohup ./script/debug_05.sh &> ./log/debug_05.txt &
# ! run in DICC v100
# * Debugging reason why low test accuracy & cuda out of memory
# * Run using author setting, only for teacher model

echo "# Teacher/Source/Victim model"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3

echo "RAND"
echo "# Teacher/Source/Victim model"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized

echo "MINGD"
echo "# Teacher/Source/Victim model"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized