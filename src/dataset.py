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
from utils.proc import GaussNoise


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
    elif args.experiment == 'ssim-cifar10':
        tr_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)) if args.normalize else transforms.Lambda(lambda x: x)
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            GaussNoise(args.noise_sigma),
            tr_normalize,
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
    elif args.experiment == 'cifar10-cinic10-excl':
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

        if args.dataset == 'CIFAR10-CINIC10-EXCL':
            train_data = get_cifar10_cinic10_excl_ratio_dataset(
                data_dir="./data",
                download=True,
                ratio=args.combine_ratio,
                transform=transform_train
            )  # ! Change
            test_data = datasets.CIFAR10(
                "./data",
                train=False,
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


def get_cifar10_cinic10_excl_ratio_dataset(
    data_dir: str = './data',
    download: bool = True,
    ratio: float = 0.5,
    total_sample: int = 50000,
    transform: transforms.Compose = transforms.Compose([
        transforms.ToTensor()
    ])
) -> ConcatDataset:
    '''
    Train only, combination of CIFAR10 and CINIC10 excluding CIFAR10 data
    '''
    # calculate split ratio
    cifar_num_sample = int(total_sample * ratio)
    cinic_num_sample = total_sample - cifar_num_sample

    # check if 100% or 0%
    if cinic_num_sample < 1:
        train_concate_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=download,
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
            download=download,
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
