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