# Notes
```python
rename_dict = {
    "threat_model": "Threat Model", 
    "teacher": "Source", 
    "distillation": "Distillation", 
    "extract-label": "Label-Query", 
    "extract-logit": "Logit-Query", 
    "zero-shot": "Zero-Shot Learning",
    "fine-tune": "Fine-Tuning",
    "pre-act-18": "Diff. Architecture",
    }
# A) complete model theft
# --> A.1 "zero-shot" | Datafree distillation / Zero shot learning
# --> A.2 "fine-tune" | Fine tuning (on unlabeled data to slightly change decision surface)
# B) Extraction over an API:
# --> B.1 "extract-label" | Model extraction using unlabeled data and victim labels
# --> B.2 "extract-logit" | Model extraction using unlabeled data and victim confidence
# C) Complete data theft:
# --> C.1 "distillation" | Data distillation
# --> C.2 "pre-act-18" | Different architecture/learning rate/optimizer/training epochs
# --> C.3 ? | Coresets
# D) ? | Train a teacher model on a separate dataset (test set)

gdown https://drive.google.com/uc?id=1LTw3Sb5QoiCCN-6Y5PEKkq9C9W60w-Hi
unzip drive-download-20220317T135303Z-001.zip -d files
```

## Training Case
```bash
# train teacher model
python3 train.py \
  --batch_size 64 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 30 \
  --dataset CIFAR10 \
  --dropRate 0.3

# B.1 "extract-label" | Model extraction using unlabeled data and victim labels
python3 train.py \
  --batch_size 64 \
  --mode extract-label \
  --normalize 1 \
  --model_id extract-label_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 15 \
  --dataset CIFAR10

# B.2 "extract-logit" | Model extraction using unlabeled data and victim confidence
python3 train.py \
  --batch_size 64 \
  --mode extract-logit \
  --normalize 1 \
  --model_id extract_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 15 \
  --dataset CIFAR10

# C.1 "distillation" | Data distillation
python3 train.py \
  --batch_size 64 \
  --mode distillation \
  --normalize 1 \
  --model_id distillation_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10

# C.2 "pre-act-18" | Different architecture/learning rate/optimizer/training epochs
python3 train.py \
  --batch_size 64 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10
```

## Feature Extraction Case
```bash
# train teacher model
python3 train.py \
  --batch_size 64 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 30 \
  --dataset CIFAR10 \
  --dropRate 0.3

python3 ./src/generate_features.py \
  --batch_size 64 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --dataset CIFAR10 \
  --feature_type mingd \
  --gpu_id 3

# B.1 "extract-label" | Model extraction using unlabeled data and victim labels
python3 train.py \
  --batch_size 64 \
  --mode extract-label \
  --normalize 1 \
  --model_id extract-label_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 15 \
  --dataset CIFAR10

python3 ./src/generate_features.py \
  --batch_size 64 \
  --mode extract-label \
  --normalize 1 \
  --model_id extract-label_normalized \
  --dataset CIFAR10 \
  --feature_type mingd \
  --gpu_id 3

# B.2 "extract-logit" | Model extraction using unlabeled data and victim confidence
python3 train.py \
  --batch_size 64 \
  --mode extract-logit \
  --normalize 1 \
  --model_id extract_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 15 \
  --dataset CIFAR10

python3 ./src/generate_features.py \
  --batch_size 64 \
  --mode extract-logit \
  --normalize 1 \
  --model_id extract_normalized \
  --dataset CIFAR10 \
  --feature_type mingd \
  --gpu_id 3

# C.1 "distillation" | Data distillation
python3 train.py \
  --batch_size 64 \
  --mode distillation \
  --normalize 1 \
  --model_id distillation_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10

python3 ./src/generate_features.py \
  --batch_size 64 \
  --mode distillation \
  --normalize 1 \
  --model_id distillation_normalized \
  --dataset CIFAR10 \
  --feature_type mingd \
  --gpu_id 3

# C.2 "pre-act-18" | Different architecture/learning rate/optimizer/training epochs
python3 train.py \
  --batch_size 64 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10

python3 ./src/generate_features.py \
  --batch_size 64 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18_normalized \
  --dataset CIFAR10 \
  --feature_type mingd \
  --gpu_id 3

```

## Author Training Case
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Teacher model
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


# A.1 "zero-shot" | Datafree distillation / Zero shot learning


# A.2 "fine-tune" | Fine tuning
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


# B.1 "extract-label" | Model extraction
python3 ./src/train.py \
  --batch_size 1000 \
  --mode extract-label \
  --normalize 1 \
  --model_id extract-label_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 20 \
  --dataset CIFAR10


# B.2 "extract-logit" | Model extraction
python3 ./src/train.py \
  --batch_size 1000 \
  --mode extract-logit \
  --normalize 1 \
  --model_id extract_normalized \
  --pseudo_labels 1 \
  --lr_mode 2 \
  --epochs 20 \
  --dataset CIFAR10


# C.1 "distillation" | Data distillation
python3 ./src/train.py \
  --batch_size 1000 \
  --mode distillation \
  --normalize 1 \
  --model_id distillation_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10


# C.2 "pre-act-18" | Different architecture
python3 ./src/train.py \
  --batch_size 1000 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10


python3 ./src/train.py \
  --gpu_id 0 \
  --batch_size 1000 \
  --mode independent \
  --normalize 1 \
  --model_id independent_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10

nohup ./lazy_train_01.sh &> log_lazy_train_01.txt &
ps aux | grep -i 'python3'
ps aux | grep -i 'generate_features.py'
```

## Author Feature Extraction Case
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# teacher model
python ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10 \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized

# A.1 "zero-shot" | Datafree distillation / Zero shot learning


# A.2 "fine-tune" | Fine tuning
python ./src/generate_features.py \
  --batch_size 500 \
  --mode fine-tune \
  --normalize 1 \
  --model_id fine-tune_normalized \
  --feature_type rand \
  --dataset CIFAR10

# B.1 "extract-label" | Model extraction
python ./src/generate_features.py \
  --batch_size 500 \
  --mode extract-label \
  --normalize 1 \
  --model_id extract-label_normalized \
  --feature_type rand \
  --dataset CIFAR10

# B.2 "extract-logit" | Model extraction
python ./src/generate_features.py \
  --batch_size 500 \
  --mode extract-logit \
  --normalize 1 \
  --model_id extract-logit_normalized \
  --feature_type rand \
  --dataset CIFAR10

# C.1 "distillation" | Data distillation
python ./src/generate_features.py \
  --batch_size 500 \
  --mode distillation \
  --normalize 1 \
  --model_id distillation_normalized \
  --feature_type rand \
  --dataset CIFAR10

# C.2 "pre-act-18" | Different architecture
python ./src/generate_features.py \
  --batch_size 500 \
  --mode pre-act-18 \
  --normalize 1 \
  --model_id pre-act-18_normalized \
  --feature_type rand \
  --dataset CIFAR10

# independent
python ./src/generate_features.py \
  --batch_size 500 \
  --mode independent \
  --normalize 1 \
  --model_id independent_normalized \
  --feature_type rand \
  --dataset CIFAR10

nohup ./lazy_fg_02.sh &> log_lazy_fg_02.txt &
ps aux | grep -i 'python3'
ps aux | grep -i 'generate_features.py'
```

## Running Commands
```bash
# generate_features
python ./src/generate_features.py \
  --batch_size 64 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --dataset CIFAR10\
  --feature_type mingd \
  --gpu_id 3

./lazy_fg.sh &> log_01.txt
./lazy_fg.sh &>> log_01.txt

nohup long-running-command &
nohup ./lazy_fg.sh &> log_01.txt &
ps aux | grep -i 'python3'
ps aux | grep -i 'generate_features.py'
kill ##

# fixed GPU used
export CUDA_VISIBLE_DEVICES=1,2
```

## Links
- https://towardsdatascience.com/visualizing-regularization-and-the-l1-and-l2-norms-d962aa769932
- https://montjoile.medium.com/l0-norm-l1-norm-l2-norm-l-infinity-norm-7a7d18a4f40c



Objective1: Able to detect 4 faces simultaneously

Objective2: To classify whether a person is wearing face mask based on the detected faces.


## Trained Model
- CIFAR10
- Normalized

### By author
| Type                   | Model          | Batch Size | Epochs |
| ---------------------- | -------------- | ---------- | ------ |
| Teacher                | WideResNet     | 1000       | 100    |
| Extract Label          | WideResNet     | 1000       | 20     |
| Extract Logit          | WideResNet     | 1000       | 20     |
| Distillation           | WideResNet     | 1000       | 50     |
| Different Architecture | PreActResNet18 | ?          | ?      |

### Me
| Type                   | Model          | Batch Size | Epochs |
| ---------------------- | -------------- | ---------- | ------ |
| Teacher                | WideResNet     | 64         | 30     |
| Extract Label          | WideResNet     | 64         | 15     |
| Extract Logit          | WideResNet     | 64         | 15     |
| Distillation           | WideResNet     | 64         | 50     |
| Different Architecture | PreActResNet18 | 64         | 50     |


## Cat Dogs
### Training Commands
```bash
nohup ./lazy_cat_dog_train_01.sh &> log_lazy_cat_dog_train_01.txt &
ps aux | grep -i 'python3'
ps aux | grep -i './src/train.py'

export CUDA_VISIBLE_DEVICES=1,2

python ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10-Cat-Dog \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --use_data_parallel

## start

python ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset STL10-Cat-Dog \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --use_data_parallel

python ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset Kaggle-Cat-Dog \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --use_data_parallel

python ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10-STL10-Cat-Dog \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --use_data_parallel

python ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10-Kaggle-Cat-Dog \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --use_data_parallel

python ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset STL10-Kaggle-Cat-Dog \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --use_data_parallel

python ./src/train.py \
  --batch_size 512 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 50 \
  --dataset CIFAR10-STL10-Kaggle-Cat-Dog \
  --dropRate 0.3 \
  --pseudo_labels 0 \
  --use_data_parallel

# testing
python ./src/train.py \
  --batch_size 1024 \
  --mode teacher \
  --normalize 1 \
  --model_id test_teacher_normalized \
  --lr_mode 4 \
  --epochs 2 \
  --dataset STL10-Kaggle-Cat-Dog \
  --lr_max 0.01 \
  --pseudo_labels 0 \
  --use_data_parallel

# CIFAR10-Cat-Dog
# STL10-Cat-Dog
# Kaggle-Cat-Dog
# CIFAR10-STL10-Cat-Dog
# CIFAR10-Kaggle-Cat-Dog
# STL10-Kaggle-Cat-Dog
# CIFAR10-STL10-Kaggle-Cat-Dog
```

### Features Commands
```bash
nohup ./lazy_cat_dog_feature_01.sh &> log_lazy_cat_dog_feature_01.txt &
ps aux | grep -i 'python3'
ps aux | grep -i './src/generate_features.py'

export CUDA_VISIBLE_DEVICES=1,2

# test
python ./src/generate_features.py \
  --feature_type rand \
  --dataset CIFAR10-Cat-Dog \
  --victim_dataset CIFAR10-Cat-Dog \
  --batch_size 256 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized
```

### Other commands
```bash
rm -r ./folder
cp -r ./src/. ./dst
zip -r compressed.zip ./folder
unzip -l _model.zip
scp -P 9413 "tingweijing@ssh.jiuntian.com:/data/weijing/ting-dataset-inference/_model.zip" "/home/ting/Downloads/"
scp -P 9413 "tingweijing@ssh.jiuntian.com:/data/weijing/ting-dataset-inference/_feature.zip" "/home/ting/Downloads/"

ls -aR | grep ":$" | perl -pe 's/:$//;s/[^-][^\/]*\//    /g;s/^    (\S)/└── \1/;s/(^    |    (?= ))/│   /g;s/    (\S)/└── \1/
```