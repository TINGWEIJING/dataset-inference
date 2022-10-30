#!/bin/sh
# nohup ./script/run_28.sh &> ./log/run_28.txt &
# ! run in TITAN 4 GPU
# using ting/cifar-cinic branch
# * Feature extraction mingd & rand with diff ratio combination CIFAR10 CINIC10 EXCL, batch size 500

echo "MINGD"
echo "# Ratio 1"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --model_dataset CIFAR10-CINIC10-EXCL \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 1

echo "# Ratio 0.9"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --model_dataset CIFAR10-CINIC10-EXCL \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0.9

echo "# Ratio 0.8"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --model_dataset CIFAR10-CINIC10-EXCL \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0.8

echo "# Ratio 0.7"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --model_dataset CIFAR10-CINIC10-EXCL \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0.7

echo "# Ratio 0.5"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --model_dataset CIFAR10-CINIC10-EXCL \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0.5

echo "# Ratio 0.3"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --model_dataset CIFAR10-CINIC10-EXCL \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0.3

echo "# Ratio 0.2"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --model_dataset CIFAR10-CINIC10-EXCL \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0.2

echo "# Ratio 0.1"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --model_dataset CIFAR10-CINIC10-EXCL \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0.1

echo "# Ratio 0"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --model_dataset CIFAR10-CINIC10-EXCL \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0


echo "RAND"
echo "# Ratio 1"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --model_dataset CIFAR10-CINIC10-EXCL \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 1

echo "# Ratio 0.9"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --model_dataset CIFAR10-CINIC10-EXCL \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0.9

echo "# Ratio 0.8"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --model_dataset CIFAR10-CINIC10-EXCL \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0.8

echo "# Ratio 0.7"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --model_dataset CIFAR10-CINIC10-EXCL \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0.7

echo "# Ratio 0.5"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --model_dataset CIFAR10-CINIC10-EXCL \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0.5

echo "# Ratio 0.3"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --model_dataset CIFAR10-CINIC10-EXCL \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0.3

echo "# Ratio 0.2"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --model_dataset CIFAR10-CINIC10-EXCL \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0.2

echo "# Ratio 0.1"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --model_dataset CIFAR10-CINIC10-EXCL \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0.1

echo "# Ratio 0"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --model_dataset CIFAR10-CINIC10-EXCL \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --experiment cifar10-cinic10-excl \
  --combine_ratio 0