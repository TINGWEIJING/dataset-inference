# Author Feacture Extraction
## CIFAR10
```bash
# Teacher/Source/Victim model
python ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized

# A.1 'zero-shot' | Datafree distillation / Zero shot learning
# Unknown command
# Unable to train the model, thus no model to extract

# A.2 'fine-tune' | Fine tuning
python ./src/generate_features.py \
  --batch_size 500 \
  --mode fine-tune \
  --normalize 1 \
  --model_id fine-tune_normalized \
  --feature_type rand \
  --dataset CIFAR10

# B.1 'extract-label' | Model extraction
# same hyperparam as 'extract-logit'
python ./src/generate_features.py \
  --batch_size 500 \
  --mode extract-label \
  --normalize 1 \
  --model_id extract-label_normalized \
  --feature_type rand \
  --dataset CIFAR10

# B.2 'extract-logit' | Model extraction
python ./src/generate_features.py \
  --batch_size 500 \
  --mode extract-logit \
  --normalize 1 \
  --model_id extract-logit_normalized \
  --feature_type rand \
  --dataset CIFAR10

# C.1 'distillation' | Data distillation
python ./src/generate_features.py \
  --batch_size 500 \
  --mode distillation \
  --normalize 1 \
  --model_id distillation_normalized \
  --feature_type rand \
  --dataset CIFAR10

# C.2 'pre-act-18' | Different architecture
# Unknown, assume same with other command
python ./src/generate_features.py \
  --batch_size 500 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18_normalized \
  --feature_type rand \
  --dataset CIFAR10

# 'independent'
# Assume same hyperparam with 'distillation'
python ./src/generate_features.py \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id independent_normalized \
  --feature_type rand \
  --dataset CIFAR10
```

## Run 02
- Use 250 batch_size instead of 500 because not enough GPU memory
```bash
export CUDA_VISIBLE_DEVICES=2,3
# Teacher/Source/Victim model
python3 ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 250 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized

# A.2 'fine-tune' | Fine tuning
python3 ./src/generate_features.py \
  --batch_size 250 \
  --mode fine-tune \
  --normalize 1 \
  --model_id fine-tune_normalized \
  --feature_type rand \
  --dataset CIFAR10

# B.1 'extract-label' | Model extraction
python3 ./src/generate_features.py \
  --batch_size 250 \
  --mode extract-label \
  --normalize 1 \
  --model_id extract-label_normalized \
  --feature_type rand \
  --dataset CIFAR10

# B.2 'extract-logit' | Model extraction
python3 ./src/generate_features.py \
  --batch_size 250 \
  --mode extract-logit \
  --normalize 1 \
  --model_id extract-logit_normalized \
  --feature_type rand \
  --dataset CIFAR10

# C.1 'distillation' | Data distillation
python3 ./src/generate_features.py \
  --batch_size 250 \
  --mode distillation \
  --normalize 1 \
  --model_id distillation_normalized \
  --feature_type rand \
  --dataset CIFAR10

# C.2 'pre-act-18' | Different architecture
python3 ./src/generate_features.py \
  --batch_size 250 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18_normalized \
  --feature_type rand \
  --dataset CIFAR10

# 'independent'
python3 ./src/generate_features.py \
  --batch_size 250 \
  --mode independent \
  --normalize 1 \
  --model_id independent_normalized \
  --feature_type rand \
  --dataset CIFAR10
```