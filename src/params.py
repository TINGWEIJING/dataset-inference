from pathlib import Path
import sys
from optparse import Option
from typing import Optional

import torch
import yaml
from tap import Tap
from typing_extensions import Literal


class Args(Tap):
    '''
    Reimplementation of `def parse_args():`
    '''
    # Basics
    config_file: Optional[str] = None
    '''Configuration file containing parameters'''
    dataset: Literal[
        "ImageNet",
        "MNIST",
        "SVHN",
        "CIFAR10",
        "CIFAR100",
        "AFAD",
        "CIFAR10-Cat-Dog",
        "STL10-Cat-Dog",
        "Kaggle-Cat-Dog",
        "CIFAR10-STL10-Cat-Dog",
        "CIFAR10-Kaggle-Cat-Dog",
        "STL10-Kaggle-Cat-Dog",
        "CIFAR10-STL10-Kaggle-Cat-Dog",
    ] = "CIFAR10"
    '''MNIST/CIFAR10'''
    model_type: Literal["cnn", "wrn-40-2", "wrn-28-10", "preactresnet", "resnet34"] \
        = "wrn-28-10"
    '''cnn/wrn-40-2/wrn-28-10/preactresnet/resnet34'''
    gpu_id: int = 0
    '''Id of GPU to be used'''
    batch_size: int = 100
    '''Batch Size for Train Set (Default = 100)'''
    model_id: str = '0'
    '''For Saving'''
    seed: int = 0
    '''Seed'''
    normalize: Literal[0, 1] = 1
    '''Normalize training data inside the model'''
    # Only valid if restarts = 1: #0 -> Always start from Zero, #1-> Always start with 1, #2-> Start from 0/rand with prob = 1/2
    device: str = 'cuda:0'
    '''To be assigned later'''
    epochs: int = 50
    '''Number of Epochs'''
    dropRate: float = 0.0
    '''DropRate for Teacher model'''
    imagenet_architecture: Literal["wrn", "alexnet", "inception"] = "wrn"
    '''Imagenet Architecture'''

    use_data_parallel: bool = False
    '''Use `nn.DataParallel` to train model'''

    # Threat models
    mode: Literal['zero-shot',
                  'prune',
                  'fine-tune',
                  'extract-label',
                  'extract-logit',
                  'distillation',
                  'teacher',
                  'independent',
                  'pre-act-18',
                  'random'] = 'teacher'
    '''Various threat models'''
    pseudo_labels: Literal[0, 1] = 0
    '''Use alternate dataset'''
    reverse_train_test: Literal[0, 1] = 0
    '''Use alternate dataset'''
    data_path: Optional[str] = None
    '''Use alternate dataset'''
    concat: Literal[0, 1] = 0
    '''For Overlap Exps'''
    concat_factor: float = 1.0
    '''For Overlap Exps'''

    # LR
    lr_mode: int = 1
    '''Step wise or Cyclic'''
    opt_type: str = "SGD"
    '''Optimizer'''
    lr_max: float = 0.1
    '''Max LR'''
    lr_min: float = 0.0
    '''Min LR'''

    # Resume
    resume: int = 0
    '''For Resuming from checkpoint'''
    resume_iter: int = -1
    '''Epoch to resume from'''

    # Lp Norm Dependent
    distance: Optional[str] = None
    '''Type of Adversarial Perturbation'''
    # , choices = ["linf", "l1", "l2", "vanilla"])
    randomize: Literal[0, 1, 2] = 0
    '''For the individual attacks'''
    alpha_l_1: float = 1.0
    '''Step Size for L1 attacks'''
    alpha_l_2: float = 0.01
    '''Step Size for L2 attacks'''
    alpha_l_inf: float = 0.001
    '''Step Size for Linf attacks'''
    num_iter: int = 500
    '''PGD iterations'''

    epsilon_l_1: float = 12
    '''Step Size for L1 attacks'''
    epsilon_l_2: float = 0.5
    '''Epsilon Radius for L2 attacks'''
    epsilon_l_inf: float = 8/255.
    '''Epsilon Radius for Linf attacks'''
    restarts: int = 1
    '''Random Restarts'''
    smallest_adv: int = 1
    '''Early Stop on finding adv'''
    gap: float = 0.001
    '''For L1 attack'''
    k: int = 100
    '''For L1 attack'''

    # TEST
    path: Optional[str] = None
    '''Path for test model load'''
    feature_type: Literal['pgd', 'topgd', 'mingd', 'rand'] = 'mingd'
    '''Feature type for generation'''
    regressor_embed: Literal[0, 1] = 0
    '''Victim Embeddings for training regressor'''

    # Others
    # TODO: add other non related to training options
    model_complete_dir_path: Path = Path.cwd()
    '''(TBAL) model complete dir path'''

    feature_complete_dir_path: Path = Path.cwd()
    '''(TBAL) feature complete dir path'''

    num_classes: int = 0
    '''(TBAL) num of classess of chosen dataset'''

    download_dataset: bool = False
    '''Check & download required datasets through PyTorch'''

    resize_dim: Literal[32, 96, 128] = 128
    '''Image resizing for cat-dog datasets'''

    victim_dataset: Literal[
        "ImageNet",
        "MNIST",
        "SVHN",
        "CIFAR10",
        "CIFAR100",
        "AFAD",
        "CIFAR10-Cat-Dog",
        "STL10-Cat-Dog",
        "Kaggle-Cat-Dog",
        "CIFAR10-STL10-Cat-Dog",
        "CIFAR10-Kaggle-Cat-Dog",
        "STL10-Kaggle-Cat-Dog",
        "CIFAR10-STL10-Kaggle-Cat-Dog",
    ] = "CIFAR10"
    '''Victim dataset for feature extraction'''

    # Hidden/Generated
    _wik: str = 'Ok'
    '''Hidden'''

    def save(self,
             path: str,
             with_reproducibility: bool = True,
             skip_unpicklable: bool = False,
             repo_path: Optional[str] = None) -> None:
        # TODO: filter or process "Others" options
        return super().save(path, with_reproducibility, skip_unpicklable, repo_path)


if __name__ == "__main__":
    args = Args().parse_args()
    # print(args.get_reproducibility_info())
    print(args)
    args.save('args.json')
