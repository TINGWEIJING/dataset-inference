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


def get_new_dataloader(args):
    # ! Add quick experiment model loading
    if args.experiment == 'unrelated-dataset':
        # ! Use same normalize for both CIFAR & SVHN
        tr_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)) if args.normalize else transforms.Lambda(lambda x: x)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), tr_normalize,
            transforms.Lambda(lambda x: x.float()), ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            tr_normalize,
            transforms.Lambda(lambda x: x.float())])

        if args.dataset == 'CIFAR10':
            train_data = datasets.CIFAR10(
                "./data",
                train=True,
                download=True,
                transform=transform_train
            )  # ! Change
            test_data = datasets.CIFAR10(
                "./data",
                train=False,
                download=True,
                transform=transform_test
            )  # ! Change
        elif args.dataset == 'SVHN':
            train_data = datasets.SVHN(
                "./data",
                split='train',
                download=True,
                transform=transform_train
            )  # ! Change
            test_data = datasets.SVHN(
                "./data",
                split='test',
                download=True,
                transform=transform_test
            )  # ! Change
        else:
            raise NotImplementedError()
        train_loader = DataLoader(train_data,
                                  batch_size=args.batch_size,
                                  shuffle=True
                                  )
        test_loader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 shuffle=False)
        return train_loader, test_loader
    else:
        raise NotImplementedError()
