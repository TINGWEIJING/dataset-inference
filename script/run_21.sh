#!/bin/sh
# nohup ./script/run_21.sh &> ./log/run_21.txt &
# ! run in DICC v100
# using ting/normalization branch
# * Train model with/without model normalize and data normalize

echo "# --normalize 0, --data_normalize 0"
echo "# Teacher/Source/Victim model"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode teacher \
  --normalize 0 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment normalization \
  --data_normalize 0

echo "# 'independent'"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode independent \
  --normalize 0 \
  --model_id independent \
  --lr_mode 2 \
  --epochs 300 \
  --dataset CIFAR10 \
  --experiment normalization \
  --data_normalize 0

echo "# C.2 'pre-act-18' | Different architecture"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode pre-act-18 \
  --normalize 0 \
  --model_id pre-act-18 \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10 \
  --experiment normalization \
  --data_normalize 0

echo "# --normalize 0, --data_normalize 1"
echo "# Teacher/Source/Victim model"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode teacher \
  --normalize 0 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment normalization \
  --data_normalize 1

echo "# 'independent'"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode independent \
  --normalize 0 \
  --model_id independent \
  --lr_mode 2 \
  --epochs 300 \
  --dataset CIFAR10 \
  --experiment normalization \
  --data_normalize 1

echo "# C.2 'pre-act-18' | Different architecture"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode pre-act-18 \
  --normalize 0 \
  --model_id pre-act-18 \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10 \
  --experiment normalization \
  --data_normalize 1

echo "# --normalize 1, --data_normalize 0"
echo "# Teacher/Source/Victim model"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment normalization \
  --data_normalize 0

echo "# 'independent'"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode independent \
  --normalize 1 \
  --model_id independent \
  --lr_mode 2 \
  --epochs 300 \
  --dataset CIFAR10 \
  --experiment normalization \
  --data_normalize 0

echo "# C.2 'pre-act-18' | Different architecture"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18 \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10 \
  --experiment normalization \
  --data_normalize 0

echo "# --normalize 1, --data_normalize 1"
echo "# Teacher/Source/Victim model"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3 \
  --experiment normalization \
  --data_normalize 1

echo "# 'independent'"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode independent \
  --normalize 1 \
  --model_id independent \
  --lr_mode 2 \
  --epochs 300 \
  --dataset CIFAR10 \
  --experiment normalization \
  --data_normalize 1

echo "# C.2 'pre-act-18' | Different architecture"
python3 ./src/train.py \
  --batch_size 1000 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18 \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10 \
  --experiment normalization \
  --data_normalize 1
