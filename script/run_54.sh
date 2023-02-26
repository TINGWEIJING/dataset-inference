#!/bin/sh
# nohup ./script/run_54.sh &> ./log/run_54.txt &
# ! run in DICC v100
# using ting/diff-norm-value-cossim-02-03-fp branch
# * Mingd & Rand feature extraction for model training with diff mean normalization with false positive cases

echo "# MINGD"
echo "# Mean: -0.477  0.49  -0.996"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.477  0.49  -0.996 \
  --load_model_normalize 1

echo "# Mean: -0.767 -0.475  0.515"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.767 -0.475  0.515 \
  --load_model_normalize 1

echo "# Mean: -0.261 -0.614  0.349"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.261 -0.614  0.349 \
  --load_model_normalize 1

echo "# Mean:  0.369 -0.311 -0.81 "
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean  0.369 -0.311 -0.81  \
  --load_model_normalize 1

echo "# Mean: -0.574 -0.897  0.64 "
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.574 -0.897  0.64  \
  --load_model_normalize 1

echo "# Mean: -0.937  0.102  0.19 "
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.937  0.102  0.19  \
  --load_model_normalize 1

echo "# Mean: -0.767  0.2    0.033"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.767  0.2    0.033 \
  --load_model_normalize 1

echo "# Mean: -0.426  0.063  0.079"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.426  0.063  0.079 \
  --load_model_normalize 1

echo "# Mean: -0.462 -0.85   0.601"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.462 -0.85   0.601 \
  --load_model_normalize 1

echo "# Mean: -0.56   0.306 -0.188"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.56   0.306 -0.188 \
  --load_model_normalize 1

echo "# RAND"
echo "# Mean: -0.477  0.49  -0.996"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.477  0.49  -0.996 \
  --load_model_normalize 1

echo "# Mean: -0.767 -0.475  0.515"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.767 -0.475  0.515 \
  --load_model_normalize 1

echo "# Mean: -0.261 -0.614  0.349"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.261 -0.614  0.349 \
  --load_model_normalize 1

echo "# Mean:  0.369 -0.311 -0.81 "
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean  0.369 -0.311 -0.81  \
  --load_model_normalize 1

echo "# Mean: -0.574 -0.897  0.64 "
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.574 -0.897  0.64  \
  --load_model_normalize 1

echo "# Mean: -0.937  0.102  0.19 "
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.937  0.102  0.19  \
  --load_model_normalize 1

echo "# Mean: -0.767  0.2    0.033"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.767  0.2    0.033 \
  --load_model_normalize 1

echo "# Mean: -0.426  0.063  0.079"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.426  0.063  0.079 \
  --load_model_normalize 1

echo "# Mean: -0.462 -0.85   0.601"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.462 -0.85   0.601 \
  --load_model_normalize 1

echo "# Mean: -0.56   0.306 -0.188"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.56   0.306 -0.188 \
  --load_model_normalize 1


echo "# MINGD"
echo "# Mean: -0.731 -0.311  0.544"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.731 -0.311  0.544 \
  --load_model_normalize 1

echo "# Mean:  0.219 -0.247 -0.174"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean  0.219 -0.247 -0.174 \
  --load_model_normalize 1

echo "# Mean: -0.721  0.778 -0.601"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.721  0.778 -0.601 \
  --load_model_normalize 1

echo "# Mean:  0.749 -0.448 -0.776"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean  0.749 -0.448 -0.776 \
  --load_model_normalize 1

echo "# Mean: -0.311 -0.857  0.897"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.311 -0.857  0.897 \
  --load_model_normalize 1

echo "# Mean:  0.205  0.181 -0.549"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean  0.205  0.181 -0.549 \
  --load_model_normalize 1

echo "# Mean: -0.804  0.685  0.01 "
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.804  0.685  0.01  \
  --load_model_normalize 1

echo "# Mean:  0.6   -0.734  0.089"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean  0.6   -0.734  0.089 \
  --load_model_normalize 1

echo "# Mean: -0.511 -0.221  0.837"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.511 -0.221  0.837 \
  --load_model_normalize 1

echo "# Mean:  0.853 -0.615 -0.149"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean  0.853 -0.615 -0.149 \
  --load_model_normalize 1

echo "# RAND"
echo "# Mean: -0.731 -0.311  0.544"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.731 -0.311  0.544 \
  --load_model_normalize 1

echo "# Mean:  0.219 -0.247 -0.174"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean  0.219 -0.247 -0.174 \
  --load_model_normalize 1

echo "# Mean: -0.721  0.778 -0.601"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.721  0.778 -0.601 \
  --load_model_normalize 1

echo "# Mean:  0.749 -0.448 -0.776"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean  0.749 -0.448 -0.776 \
  --load_model_normalize 1

echo "# Mean: -0.311 -0.857  0.897"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.311 -0.857  0.897 \
  --load_model_normalize 1

echo "# Mean:  0.205  0.181 -0.549"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean  0.205  0.181 -0.549 \
  --load_model_normalize 1

echo "# Mean: -0.804  0.685  0.01 "
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.804  0.685  0.01  \
  --load_model_normalize 1

echo "# Mean:  0.6   -0.734  0.089"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean  0.6   -0.734  0.089 \
  --load_model_normalize 1

echo "# Mean: -0.511 -0.221  0.837"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean -0.511 -0.221  0.837 \
  --load_model_normalize 1

echo "# Mean:  0.853 -0.615 -0.149"
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id teacher \
  --experiment diff-norm-value \
  --normalization_mean  0.853 -0.615 -0.149 \
  --load_model_normalize 1

