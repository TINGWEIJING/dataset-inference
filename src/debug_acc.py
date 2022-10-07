import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_new_dataloader

from funcs import get_dataloaders
from models import WideResNet
from train import epoch, epoch_test

if __name__ == "__main__":
    # Arg parser
    parser = argparse.ArgumentParser(description='Accuracy debugging')
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--training_batch_size", type=int)
    parser.add_argument("--feature_batch_size", type=int)

    args = parser.parse_args()
    print(args)
    device = torch.device("cuda:{0}".format(0) if torch.cuda.is_available() else "cpu")
    args.device = device
    print(f"device: {device}")

    # other required value settings
    args.mode = "teacher"

    # load model CIFAR10
    model = WideResNet(
        n_classes=10,
        depth=34,  # deep_full for CIFAR10
        widen_factor=10,
        normalize=1,
        dropRate=0.3,
    )
    location = args.model_path
    try:
        model = model.to(args.device)
        model.load_state_dict(torch.load(location, map_location=args.device))
    except:
        model = nn.DataParallel(model).to(args.device)
        model.load_state_dict(torch.load(location, map_location=args.device))
    model.eval()

    # new dataloader
    args.experiment = '3-var'
    args.dataset = 'CIFAR10'
    args.normalize = 1
    args.batch_size = 500
    train_loader, test_loader = get_new_dataloader(
        args,
    )   

    train_loss, train_acc = epoch(
        args,
        train_loader,
        model,
        teacher=None,
        lr_schedule=None,
        epoch_i=None,
        opt=None,
    )
    test_loss, test_acc = epoch_test(
        args,
        test_loader,
        model,
    )
    print(f"## Using new datalaoder")
    print(f'Epoch: {0}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')

    # load train.py used dataset
    train_loader, test_loader = get_dataloaders(
        dataset="CIFAR10",
        batch_size=args.training_batch_size,
        pseudo_labels=False,
        concat=False,
        concat_factor=False,
    )

    train_loss, train_acc = epoch(
        args,
        train_loader,
        model,
        teacher=None,
        lr_schedule=None,
        epoch_i=None,
        opt=None,
    )
    test_loss, test_acc = epoch_test(
        args,
        test_loader,
        model,
    )
    print(f"## Using training model datalaoder")
    print(f'Epoch: {0}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')

    # load generate_features.py used dataset
    train_loader, test_loader = get_dataloaders(
        dataset="CIFAR10",
        batch_size=args.feature_batch_size,
        pseudo_labels=False,
        train_shuffle=False,
    )
    test_loss, test_acc = epoch_test(
        args,
        test_loader,
        model,
        stop=True,
    )
    print(f"## Using default feature extraction testing datalaoder")
    print(f'Test Acc: {test_acc:.3f}')

    train_loss, train_acc = epoch(
        args,
        train_loader,
        model,
        teacher=None,
        lr_schedule=None,
        epoch_i=None,
        opt=None,
    )

    test_loss, test_acc = epoch_test(
        args,
        test_loader,
        model,
        stop=False,
    )
    print(f"## Using feature extraction datalaoder")
    print(f'Epoch: {0}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')
