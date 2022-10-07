#!/bin/sh
# nohup ./script/debug_06.sh &> ./log/debug_06.txt &
# ! run in DICC v100
# * Debugging reason why low test accuracy & cuda out of memory
# * Run using author setting, only for teacher model

echo "# Batch size 1000, final"
python3 ./src/debug_acc.py \
  --model_path "/home/user/tingweijing/DI-ting_3-var/models/CIFAR10_temp/model_teacher_normalized_1000/final.pt" \
  --training_batch_size 500 \
  --feature_batch_size 500

echo "# Batch size 1000, Train accuracy 99"
python3 ./src/debug_acc.py \
  --model_path "/home/user/tingweijing/DI-ting_3-var/models/CIFAR10_temp/model_teacher_normalized_1000/tr_acc_99.pt" \
  --training_batch_size 500 \
  --feature_batch_size 500

echo "# Batch size 1000, Train accuracy 90"
python3 ./src/debug_acc.py \
  --model_path "/home/user/tingweijing/DI-ting_3-var/models/CIFAR10_temp/model_teacher_normalized_1000/tr_acc_90.pt" \
  --training_batch_size 500 \
  --feature_batch_size 500

echo "# Batch size 1000, Train accuracy 80"
python3 ./src/debug_acc.py \
  --model_path "/home/user/tingweijing/DI-ting_3-var/models/CIFAR10_temp/model_teacher_normalized_1000/tr_acc_80.pt" \
  --training_batch_size 500 \
  --feature_batch_size 500