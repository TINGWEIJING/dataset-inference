#!/bin/sh
# nohup ./script/run_22.sh &> ./log/run_22.txt &
# ! run in DICC v100
# using ting/normalization branch
# * Rerun Mingd & Rand Feature extraction on trained model (independent & preactresnet) with/without model normalize and data normalize, using unnoramlized dataset

echo "# MINGD"
echo "# --model_normalize 0, --data_normalize 0"
echo "# 'independent'"
python3 ./src/generate_features.py \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id independent \
  --feature_type mingd \
  --dataset CIFAR10 \
  --experiment normalization \
  --model_normalize 0 \
  --data_normalize 0

echo "# C.2 'pre-act-18' | Different architecture"
python3 ./src/generate_features.py \
  --batch_size 500 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18 \
  --feature_type mingd \
  --dataset CIFAR10 \
  --experiment normalization \
  --model_normalize 0 \
  --data_normalize 0

echo "# --model_normalize 0, --data_normalize 1"
echo "# 'independent'"
python3 ./src/generate_features.py \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id independent \
  --feature_type mingd \
  --dataset CIFAR10 \
  --experiment normalization \
  --model_normalize 0 \
  --data_normalize 1

echo "# C.2 'pre-act-18' | Different architecture"
python3 ./src/generate_features.py \
  --batch_size 500 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18 \
  --feature_type mingd \
  --dataset CIFAR10 \
  --experiment normalization \
  --model_normalize 0 \
  --data_normalize 1

echo "# --model_normalize 1, --data_normalize 0"
echo "# 'independent'"
python3 ./src/generate_features.py \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id independent \
  --feature_type mingd \
  --dataset CIFAR10 \
  --experiment normalization \
  --model_normalize 1 \
  --data_normalize 0

echo "# C.2 'pre-act-18' | Different architecture"
python3 ./src/generate_features.py \
  --batch_size 500 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18 \
  --feature_type mingd \
  --dataset CIFAR10 \
  --experiment normalization \
  --model_normalize 1 \
  --data_normalize 0

echo "# --model_normalize 1, --data_normalize 1"
echo "# 'independent'"
python3 ./src/generate_features.py \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id independent \
  --feature_type mingd \
  --dataset CIFAR10 \
  --experiment normalization \
  --model_normalize 1 \
  --data_normalize 1

echo "# C.2 'pre-act-18' | Different architecture"
python3 ./src/generate_features.py \
  --batch_size 500 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18 \
  --feature_type mingd \
  --dataset CIFAR10 \
  --experiment normalization \
  --model_normalize 1 \
  --data_normalize 1

echo "# RAND"
echo "# --model_normalize 0, --data_normalize 0"
echo "# 'independent'"
python3 ./src/generate_features.py \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id independent \
  --feature_type rand \
  --dataset CIFAR10 \
  --experiment normalization \
  --model_normalize 0 \
  --data_normalize 0

echo "# C.2 'pre-act-18' | Different architecture"
python3 ./src/generate_features.py \
  --batch_size 500 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18 \
  --feature_type rand \
  --dataset CIFAR10 \
  --experiment normalization \
  --model_normalize 0 \
  --data_normalize 0

echo "# --model_normalize 0, --data_normalize 1"
echo "# 'independent'"
python3 ./src/generate_features.py \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id independent \
  --feature_type rand \
  --dataset CIFAR10 \
  --experiment normalization \
  --model_normalize 0 \
  --data_normalize 1

echo "# C.2 'pre-act-18' | Different architecture"
python3 ./src/generate_features.py \
  --batch_size 500 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18 \
  --feature_type rand \
  --dataset CIFAR10 \
  --experiment normalization \
  --model_normalize 0 \
  --data_normalize 1

echo "# --model_normalize 1, --data_normalize 0"
echo "# 'independent'"
python3 ./src/generate_features.py \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id independent \
  --feature_type rand \
  --dataset CIFAR10 \
  --experiment normalization \
  --model_normalize 1 \
  --data_normalize 0

echo "# C.2 'pre-act-18' | Different architecture"
python3 ./src/generate_features.py \
  --batch_size 500 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18 \
  --feature_type rand \
  --dataset CIFAR10 \
  --experiment normalization \
  --model_normalize 1 \
  --data_normalize 0

echo "# --model_normalize 1, --data_normalize 1"
echo "# 'independent'"
python3 ./src/generate_features.py \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id independent \
  --feature_type rand \
  --dataset CIFAR10 \
  --experiment normalization \
  --model_normalize 1 \
  --data_normalize 1

echo "# C.2 'pre-act-18' | Different architecture"
python3 ./src/generate_features.py \
  --batch_size 500 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18 \
  --feature_type rand \
  --dataset CIFAR10 \
  --experiment normalization \
  --model_normalize 1 \
  --data_normalize 1