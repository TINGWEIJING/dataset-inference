import os
import shutil
from pathlib import Path
from pprint import pprint

import numpy as np
import PIL
import torch
from matplotlib import pyplot as plt
from PIL import ImageStat
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms


def get_cifar_cat_dog_dataset(
    data_dir: str = "./_dataset",
    train: bool = True,
    download: bool = True,
    transform: transforms.Compose = transforms.Compose([
        transforms.ToTensor()
    ])
) -> Subset:
    # load dataset
    cifar_datasets = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=download,
        transform=transform,
    )
    # convert to numpy data
    cifar_datasets.targets = np.array(cifar_datasets.targets)
    # change the label value for filter classes
    filter_class_idx = np.isin(cifar_datasets.targets, [0, 1, 2, 4, 6, 7, 8, 9])
    cifar_datasets.targets[filter_class_idx] = -1
    # change label value for cat & dog classes
    cifar_datasets.targets[cifar_datasets.targets == 3] = 0
    cifar_datasets.targets[cifar_datasets.targets == 5] = 1
    # return filtered subset
    focus_class_idx = np.argwhere(np.isin(cifar_datasets.targets, [0, 1])).ravel()
    cats_dogs_subset = Subset(cifar_datasets, focus_class_idx)

    return cats_dogs_subset


def get_stl10_cat_dog_dataset(
    data_dir: str = "./_dataset",
    split: str = "train",
    download: bool = True,
    transform: transforms.Compose = transforms.Compose([
        transforms.ToTensor()
    ])
) -> Subset:
    stl10_datasets = datasets.STL10(
        root=data_dir,
        split=split,
        download=download,
        transform=transform,
    )
    # change the label value for filter classes
    filter_class_idx = np.isin(stl10_datasets.labels, [0, 1, 2, 4, 6, 7, 8, 9])
    stl10_datasets.labels[filter_class_idx] = 10
    # change label value for cat & dog classes
    stl10_datasets.labels[stl10_datasets.labels == 3] = 0
    stl10_datasets.labels[stl10_datasets.labels == 5] = 1
    # return filtered subset
    focus_class_idx = np.argwhere(np.isin(stl10_datasets.labels, [0, 1])).ravel()
    cats_dogs_subset = Subset(stl10_datasets, focus_class_idx)

    return cats_dogs_subset


def get_kaggle_cat_dog_dataset(
    data_dir: str = '_dataset/dogs-vs-cats',
    set_type: str = 'train',
    download: bool = True,
    transform: transforms.Compose = transforms.Compose([
        transforms.ToTensor()
    ])
):
    '''
    All, train or test set only
    '''
    if set_type == "all":
        root_dir = f"{data_dir}/{set_type}"
    elif set_type == "test":
        root_dir = f"{data_dir}/{set_type}"
    else:
        set_type = "train"
        root_dir = f"{data_dir}/{set_type}"

    kaggle_cat_dog_dataset = datasets.ImageFolder(root=root_dir,
                                                  transform=transform)
    return kaggle_cat_dog_dataset


def get_cifar_cinic_ratio_dataset(
    data_dir: str = '_dataset/',
    download: bool = True,
    ratio: float = 0.5,
    total_sample: int = 50000,
    transform: transforms.Compose = transforms.Compose([
        transforms.ToTensor()
    ])
) -> ConcatDataset:
    '''
    Train only
    '''
    # calculate split ratio
    cifar_num_sample = int(total_sample * ratio)
    cinic_num_sample = total_sample - cifar_num_sample

    # check if 100% or 0%
    if cinic_num_sample < 1:
        train_concate_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=False,
            transform=transform,
        )
    elif cifar_num_sample < 1:
        train_concate_dataset = datasets.ImageFolder(
            f"{data_dir}/CINIC-10-EXCLUDE-CIFAR/train",
            transform=transform
        )
    else:
        # load entire datasets from source
        cinic_exc_cifar_train_dataset = datasets.ImageFolder(
            f"{data_dir}/CINIC-10-EXCLUDE-CIFAR/train",
            transform=transform
        )
        cifar_train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=False,
            transform=transform,
        )
        # generate list of indices
        cifar_train_idx_list = list(range(len(cifar_train_dataset)))
        cinic_exc_cifar_train_idx_list = list(range(len(cinic_exc_cifar_train_dataset)))

        # split by ratio
        cifar_train_idx, _ = train_test_split(
            cifar_train_idx_list,
            train_size=cifar_num_sample,
            random_state=42,
            shuffle=True,
            stratify=cifar_train_dataset.targets
        )

        cinic_exc_cifar_train_idx, _ = train_test_split(
            cinic_exc_cifar_train_idx_list,
            train_size=cinic_num_sample,
            random_state=42,
            shuffle=True,
            stratify=cinic_exc_cifar_train_dataset.targets
        )

        # create PyTorch Subset
        cifar_subset = Subset(cifar_train_dataset, cifar_train_idx)
        cinic_subset = Subset(cinic_exc_cifar_train_dataset, cinic_exc_cifar_train_idx)

        # TODO: Remove after debugging
        print(f"cifar_num_sample: {cifar_num_sample}")
        print(f"cinic_num_sample: {cinic_num_sample}")
        print(f"cifar_subset: {len(cifar_subset)}")
        print(f"cinic_subset: {len(cinic_subset)}")

        # combine together
        train_concate_dataset = ConcatDataset([
            cifar_subset,
            cinic_subset,
        ])

    # TODO: Remove after debugging
    print(f"train_concate_dataset: {len(train_concate_dataset)}")

    return train_concate_dataset
