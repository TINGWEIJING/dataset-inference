#!/bin/sh
export CUDA_VISIBLE_DEVICES=1,2

# echo $CUDA_VISIBLE_DEVICES
# case: use whitebox settings, different num_embedding_samples [512 & 1024]

# echo "# CIFAR-Cat-Dog"
# python3 ./src/generate_features.py \
#   --feature_type mingd \
#   --dataset CIFAR10-Cat-Dog \
#   --victim_dataset CIFAR10-Cat-Dog \
#   --batch_size 128 \
#   --mode teacher \
#   --normalize 1 \
#   --num_embedding_samples 500 \
#   --model_id resnet18_victim_normalized

echo "# STL10-Cat-Dog"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset STL10-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 500 \
  --model_id resnet18_victim_normalized

echo "# Kaggle-Cat-Dog"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset Kaggle-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 500 \
  --model_id resnet18_victim_normalized

echo "# CIFAR10-STL10-Cat-Dog"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10-STL10-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 500 \
  --model_id resnet18_victim_normalized

echo "# CIFAR10-Kaggle-Cat-Dog"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10-Kaggle-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 500 \
  --model_id resnet18_victim_normalized

echo "# STL10-Kaggle-Cat-Dog"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset STL10-Kaggle-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 500 \
  --model_id resnet18_victim_normalized

echo "# CIFAR10-STL10-Kaggle-Cat-Dog"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10-STL10-Kaggle-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 500 \
  --model_id resnet18_victim_normalized

## size [1024]

echo "# CIFAR-Cat-Dog"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 1000 \
  --model_id resnet18_victim_normalized

echo "# STL10-Cat-Dog"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset STL10-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 1000 \
  --model_id resnet18_victim_normalized

echo "# Kaggle-Cat-Dog"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset Kaggle-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 1000 \
  --model_id resnet18_victim_normalized

echo "# CIFAR10-STL10-Cat-Dog"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10-STL10-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 1000 \
  --model_id resnet18_victim_normalized

echo "# CIFAR10-Kaggle-Cat-Dog"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10-Kaggle-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 1000 \
  --model_id resnet18_victim_normalized

echo "# STL10-Kaggle-Cat-Dog"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset STL10-Kaggle-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 1000 \
  --model_id resnet18_victim_normalized

echo "# CIFAR10-STL10-Kaggle-Cat-Dog"
python3 ./src/generate_features.py \
  --feature_type mingd \
  --dataset CIFAR10-STL10-Kaggle-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 128 \
  --mode teacher \
  --normalize 1 \
  --num_embedding_samples 1000 \
  --model_id resnet18_victim_normalized