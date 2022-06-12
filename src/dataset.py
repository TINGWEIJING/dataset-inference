import os
import shutil
from pathlib import Path
from pprint import pprint

import numpy as np
import PIL
import torch
from matplotlib import pyplot as plt
from PIL import ImageStat
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
    data_dir: str = '_dataset/dogs-vs-cats/train',
    download: bool = True,
    transform: transforms.Compose = transforms.Compose([
        transforms.ToTensor()
    ])
):
    '''
    Train set only
    '''
    kaggle_cat_dog_dataset = datasets.ImageFolder(root=data_dir,
                                                  transform=transform)
    return kaggle_cat_dog_dataset
