#!/bin/sh
# nohup ./script/debug_01.sh &> ./log/debug_01.txt &
# ! run in DICC v100
# * Debugging model test performance inconsistent

echo "# Teacher/Source/Victim model"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized
