#!/bin/sh
# nohup ./script/run_18.sh &> ./log/run_18.txt &
# ! run in DICC v100
# using 3-var branch
# * 3-var feature extraction experiment, fixed batch 1000, diff tr acc & te acc

echo "# MINGD"
echo "# Batch size 1000, Train accuracy 60"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 1000 \
  --target_tr_acc 60

echo "# Batch size 1000, Train accuracy 70"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 1000 \
  --target_tr_acc 70

echo "# Batch size 1000, Train accuracy 75"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 1000 \
  --target_tr_acc 75

echo "# Batch size 1000, Train accuracy 80"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 1000 \
  --target_tr_acc 80

echo "# Batch size 1000, Train accuracy 85"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 1000 \
  --target_tr_acc 85

echo "# Batch size 1000, Train accuracy 90"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 1000 \
  --target_tr_acc 90

echo "# Batch size 1000, Train accuracy 95"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 1000 \
  --target_tr_acc 95

echo "# Batch size 1000, Train accuracy 98"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 1000 \
  --target_tr_acc 98

echo "# Batch size 1000, Train accuracy 99"
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

echo "# RAND"
echo "# Batch size 1000, Train accuracy 60"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 1000 \
  --target_tr_acc 60

echo "# Batch size 1000, Train accuracy 70"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 1000 \
  --target_tr_acc 70

echo "# Batch size 1000, Train accuracy 75"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 1000 \
  --target_tr_acc 75

echo "# Batch size 1000, Train accuracy 80"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 1000 \
  --target_tr_acc 80

echo "# Batch size 1000, Train accuracy 85"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 1000 \
  --target_tr_acc 85

echo "# Batch size 1000, Train accuracy 90"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 1000 \
  --target_tr_acc 90

echo "# Batch size 1000, Train accuracy 95"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 1000 \
  --target_tr_acc 95

echo "# Batch size 1000, Train accuracy 98"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 1000 \
  --target_tr_acc 98

echo "# Batch size 1000, Train accuracy 99"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment 3-var \
  --target_batch_size 1000 \
  --target_tr_acc 99
