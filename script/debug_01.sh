#!/bin/sh
# nohup ./script/debug_01.sh &> ./log/debug_01.txt &
# ! run in DICC v100
# * Debugging model test performance inconsistent

echo "# Teacher/Source/Victim model"
python3 ./src/train.py \
  --batch_size 100 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 5 \
  --dataset CIFAR10 \
  --dropRate 0.3
