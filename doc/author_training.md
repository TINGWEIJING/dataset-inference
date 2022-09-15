# Author Training
## CIFAR10
```bash
# Teacher/Source/Victim model
python3 ./src/train.py \
  --gpu_id 0 \
  --batch_size 1000 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3

python train.py \
  --batch_size 1000 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3

# A.1 'zero-shot' | Datafree distillation / Zero shot learning
# Unknown command
# Required "../models/{args.dataset}/wrn-16-1/Base/STUDENT3"

# A.2 'fine-tune' | Fine tuning
# i_500K_pseudo_labeled.pickle
python3 ./src/train.py \
  --batch_size 1000 \
  --mode fine-tune \
  --lr_max 0.01 \
  --normalize 1 \
  --model_id fine-tune_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 5 \
  --dataset CIFAR10

python train.py \
  --batch_size 1000 \
  --mode fine-tune \
  --lr_max 0.01 \
  --normalize 1 \
  --model_id fine-tune_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 5 \
  --dataset CIFAR10

# B.1 'extract-label' | Model extraction
# same hyperparam as 'extract-logit'
# i_500K_pseudo_labeled.pickle
python3 ./src/train.py \
  --batch_size 1000 \
  --mode extract-label \
  --normalize 1 \
  --model_id extract-label_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 20 \
  --dataset CIFAR10

python train.py \
  --batch_size 1000 \
  --mode extract-label \
  --normalize 1 \
  --model_id extract-label_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 20 \
  --dataset CIFAR10

# B.2 'extract-logit' | Model extraction
python3 ./src/train.py \
  --batch_size 1000 \
  --mode extract-logit \
  --normalize 1 \
  --model_id extract_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 20 \
  --dataset CIFAR10

python train.py \
  --batch_size 1000 \
  --mode extract-logit \
  --normalize 1 \
  --model_id extract_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 20 \
  --dataset CIFAR10

# C.1 'distillation' | Data distillation
python3 ./src/train.py \
  --batch_size 1000 \
  --mode distillation \
  --normalize 1 \
  --model_id distillation_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10

python train.py \
  --batch_size 1000 \
  --mode distillation \
  --normalize 1 \
  --model_id distillation_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10

# C.2 'pre-act-18' | Different architecture
# Unknown
python3 ./src/train.py \
  --batch_size 1000 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10

# 'independent'
# Assume same hyperparam with 'distillation'
python3 ./src/train.py \
  --gpu_id 0 \
  --batch_size 1000 \
  --mode independent \
  --normalize 1 \
  --model_id independent_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10

python train.py \
  --batch_size 1000 \
  --mode independent \
  --normalize 1 \
  --model_id independent_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10
```

## Run 01
- Use 500 batch_size instead of 1000 because not enough GPU memory
```bash
export CUDA_VISIBLE_DEVICES=2,3
# Teacher/Source/Victim model
python3 ./src/train.py \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 100 \
  --dataset CIFAR10 \
  --dropRate 0.3

# A.2 'fine-tune' | Fine tuning
python3 ./src/train.py \
  --batch_size 500 \
  --mode fine-tune \
  --lr_max 0.01 \
  --normalize 1 \
  --model_id fine-tune_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 5 \
  --dataset CIFAR10

# B.1 'extract-label' | Model extraction
python3 ./src/train.py \
  --batch_size 500 \
  --mode extract-label \
  --normalize 1 \
  --model_id extract-label_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 20 \
  --dataset CIFAR10

# B.2 'extract-logit' | Model extraction
python3 ./src/train.py \
  --batch_size 500 \
  --mode extract-logit \
  --normalize 1 \
  --model_id extract_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 20 \
  --dataset CIFAR10

# C.1 'distillation' | Data distillation
python3 ./src/train.py \
  --batch_size 500 \
  --mode distillation \
  --normalize 1 \
  --model_id distillation_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10

# C.2 'pre-act-18' | Different architecture
python3 ./src/train.py \
  --batch_size 500 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10

# 'independent'
python3 ./src/train.py \
  --gpu_id 0 \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id independent_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10

```