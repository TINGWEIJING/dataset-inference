#!/bin/sh
# nohup ./_script/run_03.sh &> ./_log/run_03.txt &
# run for mingd
export CUDA_VISIBLE_DEVICES=2,3

echo $CUDA_VISIBLE_DEVICES

echo "# Teacher/Source/Victim model"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 250 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized

echo "# A.2 'fine-tune' | Fine tuning"
python3 ./src/generate_features.py \
  --batch_size 250 \
  --mode fine-tune \
  --normalize 1 \
  --model_id fine-tune_normalized \
  --feature_type mingd \
  --dataset CIFAR10

echo "# B.1 'extract-label' | Model extraction"
python3 ./src/generate_features.py \
  --batch_size 250 \
  --mode extract-label \
  --normalize 1 \
  --model_id extract-label_normalized \
  --feature_type mingd \
  --dataset CIFAR10

echo "# B.2 'extract-logit' | Model extraction"
python3 ./src/generate_features.py \
  --batch_size 250 \
  --mode extract-logit \
  --normalize 1 \
  --model_id extract-logit_normalized \
  --feature_type mingd \
  --dataset CIFAR10

echo "# C.1 'distillation' | Data distillation"
python3 ./src/generate_features.py \
  --batch_size 250 \
  --mode distillation \
  --normalize 1 \
  --model_id distillation_normalized \
  --feature_type mingd \
  --dataset CIFAR10

echo "# C.2 'pre-act-18' | Different architecture"
python3 ./src/generate_features.py \
  --batch_size 250 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18_normalized \
  --feature_type mingd \
  --dataset CIFAR10

echo "# 'independent'"
python3 ./src/generate_features.py \
  --batch_size 250 \
  --mode independent \
  --normalize 1 \
  --model_id independent_normalized \
  --feature_type mingd \
  --dataset CIFAR10
