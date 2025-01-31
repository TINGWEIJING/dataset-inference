#!/bin/sh
# export CUDA_VISIBLE_DEVICES=1,2

# echo $CUDA_VISIBLE_DEVICES

echo "# CIFAR-Cat-Dog"
python ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 256 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized

echo "# STL10-Cat-Dog"
python ./src/generate_features.py \
  --feature_type rand \
  --dataset STL10-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 256 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized

echo "# Kaggle-Cat-Dog"
python ./src/generate_features.py \
  --feature_type rand \
  --dataset Kaggle-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 256 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized

echo "# CIFAR10-STL10-Cat-Dog"
python ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10-STL10-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 256 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized

echo "# CIFAR10-Kaggle-Cat-Dog"
python ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10-Kaggle-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 256 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized

echo "# STL10-Kaggle-Cat-Dog"
python ./src/generate_features.py \
  --feature_type rand \
  --dataset STL10-Kaggle-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 256 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized

echo "# CIFAR10-STL10-Kaggle-Cat-Dog"
python ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10-STL10-Kaggle-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 256 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized