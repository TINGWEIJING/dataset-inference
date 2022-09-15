#!/bin/sh
# nohup ./_script/run_01a.sh &> ./_log/run_01a.txt &
export CUDA_VISIBLE_DEVICES=2,3

echo $CUDA_VISIBLE_DEVICES

echo "# Teacher/Source/Victim model"
python3 ./src/train.py \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3
