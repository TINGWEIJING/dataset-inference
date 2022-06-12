# Dataset Inference: Ownership Resolution in Machine Learning

Repository for the paper [Dataset Inference: Ownership Resolution in Machine Learning](https://openreview.net/pdf?id=hvdKKV2yt7T) by [Pratyush Maini](https://pratyushmaini.github.io), [Mohammad Yaghini]() and [Nicolas Papernot](https://papernot.fr). This work was presented at [ICLR 2021](http://iclr.cc/Conferences/2021/) as a Spotlight Presentation.

## Additional Notes 
### Quick Start
1. Install additional dependency using `pip install typed-argument-parser`
2. Download `_model.zip` file and unzip
    ```
    └── _model
    │   └── CIFAR10-Cat-Dog
    │   │   └── model_teacher_normalized
    │   └── CIFAR10-Kaggle-Cat-Dog
    │   │   └── model_teacher_normalized
    │   └── CIFAR10-STL10-Cat-Dog
    │   │   └── model_teacher_normalized
    │   └── CIFAR10-STL10-Kaggle-Cat-Dog
    │   │   └── model_teacher_normalized
    │   └── Kaggle-Cat-Dog
    │   │   └── model_teacher_normalized
    │   └── STL10-Cat-Dog
    │   │   └── model_teacher_normalized
    │   └── STL10-Kaggle-Cat-Dog
    │   │   └── model_teacher_normalized
    ```
3. Download `_feature.zip` file and unzip
    ```
    └── _feature
    │   └── CIFAR10-Cat-Dog
    │   │   └── model_teacher_normalized-CIFAR10-Cat-Dog
    │   │   └── model_teacher_normalized-CIFAR10-Kaggle-Cat-Dog
    │   │   └── model_teacher_normalized-CIFAR10-STL10-Cat-Dog
    │   │   └── model_teacher_normalized-CIFAR10-STL10-Kaggle-Cat-Dog
    │   │   └── model_teacher_normalized-Kaggle-Cat-Dog
    │   │   └── model_teacher_normalized-STL10-Cat-Dog
    │   │   └── model_teacher_normalized-STL10-Kaggle-Cat-Dog
    ```
4. Download Kaggle Dog vs Cats train dataset `dogs-vs-cats.zip`, create `_dataset` folder and unzip into it
    ```
    └── _dataset
    │   └── dogs-vs-cats
    │   │   └── train
    │   │   │   └── cat
    │   │   │   └── dog
    ```

5. If you want to train model, please refer the command scripts in `lazy_cat_dog_train_01.sh`
6. Please use `--download_dataset` flag to download STL10 & CIFAR10 datasets, e.g:
    ```bash
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
      --use_data_parallel \
      --download_dataset
    ```
7. If you want to generate features, please refer the command scripts in `lazy_cat_dog_feature_01.sh`

8. Use `notebooks/CIFAR10_rand.ipynb` to generate result

## What does this repository contain?
Code for training and evaluating all the experiments that support the aforementioned paper are provided in this repository. 
The instructions for reproducing the results can be found below.

## Dependencies
The repository is written using `python 3.8`. To install dependencies run the command:

`pip install -r requirements.txt`
```bash
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio cudatoolkit==11.3 -c pytorch -c nvidia -c conda-forge 
conda clean --all
```

## Resolving Ownership
If you already have the extracted featured for the victim and potentially stolen models, you can proceed to inferring potential theft. A sample `jupyter notebook` to perform the same can be found at:
`src/notebooks/CIFAR10_rand.ipynb`   
You can download extracted features for our models from [this link](https://drive.google.com/drive/folders/1CLJ2a3H_oTX5b_4GLurVYoCpZUXQVpFr?usp=sharing). Save them in a directory names `files` in the root directory.

## Training your own models
`python train.py --batch_size 1000 --mode $MODE --normalize $NORMALIZE --model_id $MODEL_ID --lr_mode $LR_MODE --epochs $EPOCHS --dataset $DATASET --lr_max $LR_MAX --pseudo_labels $PSEUDO`
  > `batch_size` - Batch Size for Test Set -`default = 1000`  
  > `mode` - "Various attack strategies", type = str, default = 'teacher', choices = ['zero-shot', 'fine-tune', 'extract-label', 'extract-logit', 'distillation', 'teacher','independent','pre-act-18']  
  > `normalize`  - The normalization is performed within the model and not in the dataloader to ease adversarial attack implementation. Please take note.  
  > `model_id` - Used to compute location to load the model. See directory structure in code. 
  > `pseudo_labels` - Used in case of label only model extraction

## Generating Features
`python generate_features.py --batch_size 500 --mode $MODE --normalize $NORMALIZE --model_id $MODEL_ID --dataset $DATASET --feature_type $FEATURE`
  > `batch_size` - Batch Size for Test Set -`default = 500`  
  > `mode` - "Various attack strategies", type = str, default = 'teacher', choices = ['zero-shot', 'fine-tune', 'extract-label', 'extract-logit', 'distillation', 'teacher','independent','pre-act-18']  
  > `normalize`  - The normalization is performed within the model and not in the dataloader to ease adversarial attack implementation. Please take note.  
  > `model_id` - Used to compute location to load the model. See directory structure in code.   
  > `feature_type` - 'topgd', 'mingd', 'rand'. For black-box method use Random

