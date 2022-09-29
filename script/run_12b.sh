#!/bin/sh
# nohup ./script/run_12b.sh &> ./log/run_12b.txt &
# ! run in DICC v100
# using ting/reproduce branch
# * Rerun Feature extraction mingd & rand for extract-logit because of model path error, batch size 500

echo "RAND"
echo "# B.2 'extract-logit' | Model extraction"
python3 ./src/generate_features.py \
  --batch_size 500 \
  --mode extract-logit \
  --normalize 1 \
  --model_id extract-logit_normalized \
  --feature_type rand \
  --dataset CIFAR10

echo "MINGD"
echo "# B.2 'extract-logit' | Model extraction"
python3 ./src/generate_features.py \
  --batch_size 500 \
  --mode extract-logit \
  --normalize 1 \
  --model_id extract-logit_normalized \
  --feature_type mingd \
  --dataset CIFAR10