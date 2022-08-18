#!/bin/sh
export CUDA_VISIBLE_DEVICES=1,2

# echo $CUDA_VISIBLE_DEVICES

echo "Teacher"
python ./src/generate_features.py \
  --batch_size 256 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --feature_type rand \
  --dataset CIFAR10 \
  --victim_dataset CIFAR10

echo "Independent"
python ./src/generate_features.py \
  --batch_size 256 \
  --mode independent \
  --normalize 1 \
  --model_id independent_normalized \
  --feature_type rand \
  --dataset CIFAR10 \
  --victim_dataset CIFAR10

echo "# C.1 'distillation' | Data distillation"
python ./src/generate_features.py \
  --batch_size 256 \
  --mode distillation \
  --normalize 1 \
  --model_id distillation_normalized \
  --feature_type rand \
  --dataset CIFAR10 \
  --victim_dataset CIFAR10
