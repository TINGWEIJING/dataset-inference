#!/bin/sh
# nohup ./script/debug_01.sh &> ./log/debug_01.txt &
# ! run in DICC v100
# * Debugging 3-var model loading for feature extraction

echo "# Epoch 100"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 1000 \
  --target_epoch 100

echo "# Train accuracy 99"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 1000 \
  --target_tr_acc 99

echo "# Test accuracy 80"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 500 \
  --target_te_acc 80
