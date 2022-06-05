#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo $CUDA_VISIBLE_DEVICES

echo "Teacher"
python ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized

echo "# A.2 'fine-tune' | Fine tuning"
python ./src/generate_features.py \
  --batch_size 500 \
  --mode fine-tune \
  --normalize 1 \
  --model_id fine-tune_normalized \
  --feature_type rand \
  --dataset CIFAR10


echo "# B.1 'extract-label' | Model extraction"
python ./src/generate_features.py \
  --batch_size 500 \
  --mode extract-label \
  --normalize 1 \
  --model_id extract-label_normalized \
  --feature_type rand \
  --dataset CIFAR10


echo "# B.2 'extract-logit' | Model extraction"
python ./src/generate_features.py \
  --batch_size 500 \
  --mode extract-logit \
  --normalize 1 \
  --model_id extract-logit_normalized \
  --feature_type rand \
  --dataset CIFAR10


echo "# C.1 'distillation' | Data distillation"
python ./src/generate_features.py \
  --batch_size 500 \
  --mode distillation \
  --normalize 1 \
  --model_id distillation_normalized \
  --feature_type rand \
  --dataset CIFAR10


echo "# C.2 'pre-act-18' | Different architecture"
python ./src/generate_features.py \
  --batch_size 500 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18_normalized \
  --feature_type rand \
  --dataset CIFAR10

echo "Independent"
python ./src/generate_features.py \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id independent_normalized \
  --feature_type rand \
  --dataset CIFAR10

echo "Ended"