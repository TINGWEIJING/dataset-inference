#!/bin/sh
# nohup ./_script/run_03b.sh &> ./_log/run_03b.txt &
# rerun for mingd & rand extract-logit because model folder incorrect naming
export CUDA_VISIBLE_DEVICES=2,3

echo $CUDA_VISIBLE_DEVICES

echo "# B.2 'extract-logit' | Model extraction"
python3 ./src/generate_features.py \
  --batch_size 250 \
  --mode extract-logit \
  --normalize 1 \
  --model_id extract-logit_normalized \
  --feature_type mingd \
  --dataset CIFAR10

echo "# B.2 'extract-logit' | Model extraction"
python3 ./src/generate_features.py \
  --batch_size 250 \
  --mode extract-logit \
  --normalize 1 \
  --model_id extract-logit_normalized \
  --feature_type rand \
  --dataset CIFAR10

