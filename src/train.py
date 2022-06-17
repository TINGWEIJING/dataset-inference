from __future__ import absolute_import
import sys
import time
import argparse
from typing import Callable
import ipdb
import params
import glob
import os
import json
import numpy as np
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim
from funcs import *
from models import *
from pathlib import Path
from params import Args
from utils.logger import Unbuffered
from torchvision.models import mobilenet_v3_large

'''Threat Models'''
# A) complete model theft
# --> A.1 Datafree distillation / Zero shot learning
# --> A.2 Fine tuning (on unlabeled data to slightly change decision surface)
# B) Extraction over an API:
# --> B.1 Model extraction using unlabeled data and victim labels
# --> B.2 Model extraction using unlabeled data and victim confidence
# C) Complete data theft:
# --> C.1 Data distillation
# --> C.2 Different architecture/learning rate/optimizer/training epochs
# --> C.3 Coresets
# D) Train a teacher model on a separate dataset (test set)


def step_lr(lr_max: float, epoch: int, num_epochs: int) -> float:
    '''
    Step learning rate mentioned in author paper in `6.1 MODEL STEALING ATTACK (pg07)`
    '''
    ratio = epoch/float(num_epochs)
    if ratio < 0.3:
        return lr_max
    elif ratio < 0.6:
        return lr_max*0.2
    elif ratio < 0.8:
        return lr_max*0.2*0.2
    else:
        return lr_max*0.2*0.2*0.2


def lr_scheduler(args: Args) -> Callable[[int], int]:
    '''
    Custom lr scheduler implemented by author
    '''
    if args.lr_mode == 0:
        def lr_schedule(t):
            return np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], [args.lr_max, args.lr_max*0.2, args.lr_max*0.04, args.lr_max*0.008])[0]

    elif args.lr_mode == 1:
        def lr_schedule(t):
            return np.interp([t], [0, args.epochs//2, args.epochs], [args.lr_min, args.lr_max, args.lr_min])[0]

    elif args.lr_mode == 2:  # ! used in sample run
        def lr_schedule(t):
            return np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], [args.lr_min, args.lr_max, args.lr_max/10, args.lr_min])[0]

    elif args.lr_mode == 3:
        def lr_schedule(t):
            return np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], [args.lr_max, args.lr_max, args.lr_max/5., args.lr_max/10.])[0]

    elif args.lr_mode == 4:  # ! mentioned in the paper
        def lr_schedule(t):
            return step_lr(args.lr_max, t, args.epochs)

    return lr_schedule


# Load pretrained model for A.1

def epoch(args,
          loader,
          model,
          teacher=None,
          lr_schedule=None,
          epoch_i=None,
          opt=None,
          stop=False) -> Tuple[float, float]:
    # For A.3, B.1, B.2, C.1, C.2
    """Training/evaluation epoch over the dataset"""
    # Teacher is none for C.2, B.1, A.3
    # Pass victim as teacher for B.2, C.1

    train_loss = 0
    train_acc = 0
    train_n = 0
    i = 0
    func = tqdm if stop == False else lambda x: x
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    alpha, T = 1.0, 1.0
    loss: torch.Tensor

    for batch in func(loader):
        X, y = batch[0].to(args.device), batch[1].to(args.device)
        yp = model(X)

        if teacher is not None:
            with torch.no_grad():
                t_p = teacher(X).detach()
                y = t_p.max(1)[1]
            if args.mode in ["extract-label", "fine-tune"]:
                loss = nn.CrossEntropyLoss()(yp, t_p.max(1)[1])
            else:
                loss = criterion_kl(F.log_softmax(yp/T, dim=1), F.softmax(t_p/T, dim=1))*(alpha * T * T)

        else:
            loss = nn.CrossEntropyLoss()(yp, y)

        if opt:
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)
            opt.zero_grad()
            loss.backward()
            opt.step()

        train_loss += loss.item()*y.size(0)
        train_acc += (yp.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        i += 1
        if stop:
            break

    return train_loss / train_n, train_acc / train_n


def epoch_test(args: Args,
               loader: DataLoader,
               model: torch.nn.Module,
               stop=False) -> Tuple[float, float]:
    """
    Evaluation epoch over the dataset \\
    Return mean test loss & mean test accuracy
    """
    test_loss = 0
    test_acc = 0
    test_n = 0
    def func(x: DataLoader): return x
    with torch.no_grad():
        for batch in func(loader):
            X, y = batch[0].to(args.device), batch[1].to(args.device)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)
            test_loss += loss.item()*y.size(0)
            test_acc += (yp.max(1)[1] == y).sum().item()
            test_n += y.size(0)
            if stop:
                break
    return test_loss / test_n, test_acc / test_n


def trainer(args: Args):
    '''
    Originally named `trainer`. \\
    TODO: change function name
    '''
    # output stream
    bothout = Unbuffered()

    # load pytorch dataloaders for training & testing datasets
    train_loader, test_loader = get_dataloaders(args.dataset,
                                                args.batch_size,
                                                normalize=args.normalize,
                                                download=args.download_dataset,
                                                pseudo_labels=args.pseudo_labels,
                                                concat=args.concat,
                                                concat_factor=args.concat_factor)

    # ? Not sure why swap, the paper mentioned using private dataset
    if args.mode == "independent":
        train_loader, test_loader = test_loader, train_loader

    # load student & teacher model
    student, teacher = get_student_teacher(args)

    # optimizer
    if args.opt_type == "SGD":
        opt = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    else:
        opt = optim.Adam(student.parameters(), lr=0.1)

    # learning rate scheduler
    lr_schedule = lr_scheduler(args)
    t_start = 0  # represent iteration/epoch starting number

    # TODO: test resume script
    if args.resume:
        location = f"{args.model_dir}/iter_{str(args.resume_iter)}.pt"
        t_start = args.resume_iter + 1
        student.load_state_dict(torch.load(location, map_location=device))

    for t in range(t_start, args.epochs):
        lr = lr_schedule(t)  # get current lr value for reference

        student.train()
        train_loss, train_acc = epoch(args,
                                      train_loader,
                                      student,
                                      teacher=teacher,
                                      lr_schedule=lr_schedule,
                                      epoch_i=t,
                                      opt=opt)

        student.eval()
        test_loss, test_acc = epoch_test(args,
                                         test_loader,
                                         student)

        print(f'Epoch: {t}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} lr: {lr:.5f}',
              file=bothout)
        print(f'Epoch: {t}, Train Loss: {test_loss:.3f} Test Acc: {test_acc:.3f}',
              file=bothout)

        # ! The paper didn't use MNIST datasets, this is for testing I guess
        if args.dataset == "MNIST":
            torch.save(student.state_dict(), f"{args.model_complete_dir_path}/iter_{t}.pt")
        elif (t+1) % 25 == 0:  # save every 25 iteration
            torch.save(student.state_dict(), f"{args.model_complete_dir_path}/iter_{t}.pt")

    torch.save(student.state_dict(), f"{args.model_complete_dir_path}/final.pt")


def get_student_teacher(args: Args) -> Tuple[torch.nn.Module, torch.nn.Module]:
    '''
    Prepare student (threat) model architecture based on `args.dataset` and `args.mode` for training
    '''
    w_f = 2 if args.dataset == "CIFAR100" else 1
    net_mapper = {"CIFAR10": WideResNet,
                  "CIFAR100": WideResNet,
                  "AFAD": resnet34,
                  "SVHN": ResNet_8x}
    Net_Arch: nn.Module = net_mapper.get(args.dataset, WideResNet) # ! Use WideResNet otherwise
    mode = args.mode
    # ['zero-shot', 'fine-tune', 'extract-label', 'extract-logit', 'distillation', 'teacher']
    deep_full = 34 if args.dataset in ["SVHN", "AFAD"] else 28
    deep_half = 18 if args.dataset in ["SVHN", "AFAD"] else 16

    # loading teach model
    if mode in ["teacher", "independent", "pre-act-18"]:
        teacher = None
    else:
        deep = 34 if args.dataset in ["SVHN", "AFAD"] else 28
        teacher = Net_Arch(n_classes=args.num_classes,
                           depth=deep,
                           widen_factor=10,
                           normalize=args.normalize,
                           dropRate=0.3)
        # TODO: check parallel
        if args.dataset != "SVHN":
            if args.use_data_parallel and torch.cuda.device_count() > 1:
                teacher = nn.DataParallel(teacher)
        teacher.to(args.device)
        # teacher = nn.DataParallel(teacher).to(args.device) if args.dataset != "SVHN" else teacher.to(args.device)
        teacher_dir = "model_teacher_normalized" if args.normalize else "model_teacher_unnormalized"
        path = f"./ting_models/{args.dataset}/{teacher_dir}/final"
        teacher = load(teacher, path)
        teacher.eval()

    if mode == 'zero-shot':
        student = Net_Arch(n_classes=args.num_classes,
                           depth=deep_half,
                           widen_factor=w_f,
                           normalize=args.normalize)
        # TODO: ask author how to get this
        path = f"./models/{args.dataset}/wrn-16-1/Base/STUDENT3"
        student.load_state_dict(torch.load(f"{path}.pth", map_location=device))

        # TODO: check parallel
        if args.use_data_parallel and torch.cuda.device_count() > 1:
            student = nn.DataParallel(student)
        student.to(args.device)
        # student = nn.DataParallel(student).to(args.device)
        student.eval()
        raise("Network needs to be un-normalized")
    elif mode == "prune":
        raise("Not handled")

    elif mode == "fine-tune":
        # python train.py --batch_size 1000 --mode fine-tune --lr_max 0.01 --normalize 0 --model_id fine-tune_unnormalized --pseudo_labels 1 --lr_mode 2 --epochs 5 --dataset CIFAR10
        # python train.py --batch_size 1000 --mode fine-tune --lr_max 0.01 --normalize 1 --model_id fine-tune_normalized --pseudo_labels 1 --lr_mode 2 --epochs 5 --dataset CIFAR10
        student = Net_Arch(n_classes=args.num_classes,
                           depth=deep_full,
                           widen_factor=10,
                           normalize=args.normalize)

        # TODO: check parallel
        if args.dataset != "SVHN":
            if args.use_data_parallel and torch.cuda.device_count() > 1:
                student = nn.DataParallel(student)
        student.to(args.device)
        # student = nn.DataParallel(student).to(args.device) if args.dataset != "SVHN" else student.to(args.device)
        # TODO: change model path
        teacher_dir = "model_teacher_normalized" if args.normalize else "model_teacher_unnormalized"
        path = f"./ting_models/{args.dataset}/{teacher_dir}/final"
        student = load(student, path)
        student.train()
        assert(args.pseudo_labels)

    elif mode in ["extract-label", "extract-logit"]:
        # python train.py --batch_size 1000 --mode extract-label --normalize 0 --model_id extract-label_unnormalized --pseudo_labels 1 --lr_mode 2 --epochs 20 --dataset CIFAR10
        # python train.py --batch_size 1000 --mode extract-label --normalize 1 --model_id extract-label_normalized --pseudo_labels 1 --lr_mode 2 --epochs 20 --dataset CIFAR10
        # python train.py --batch_size 1000 --mode extract-logit --normalize 0 --model_id extract_unnormalized --pseudo_labels 1 --lr_mode 2 --epochs 20 --dataset CIFAR10
        # python train.py --batch_size 1000 --mode extract-logit --normalize 1 --model_id extract_normalized --pseudo_labels 1 --lr_mode 2 --epochs 20 --dataset CIFAR10
        student = Net_Arch(n_classes=args.num_classes,
                           depth=deep_half,
                           widen_factor=w_f,
                           normalize=args.normalize)
        # TODO: check parallel
        if args.use_data_parallel and torch.cuda.device_count() > 1:
            student = nn.DataParallel(student)
        student.to(args.device)
        # student = nn.DataParallel(student).to(args.device)
        student.train()
        assert(args.pseudo_labels)

    elif mode in ["distillation", "independent"]:
        dR = 0.3 if mode == "independent" else 0.0
        # python train.py --batch_size 1000 --mode distillation --normalize 0 --model_id distillation_unnormalized --lr_mode 2 --epochs 50 --dataset CIFAR10
        # python train.py --batch_size 1000 --mode distillation --normalize 1 --model_id distillation_normalized --lr_mode 2 --epochs 50 --dataset CIFAR10
        student = Net_Arch(n_classes=args.num_classes,
                           depth=deep_half,
                           widen_factor=w_f,
                           normalize=args.normalize,
                           dropRate=dR)
        # TODO: check parallel
        if args.use_data_parallel and torch.cuda.device_count() > 1:
            student = nn.DataParallel(student)
        student.to(args.device)
        # student = nn.DataParallel(student).to(args.device)
        student.train()

    elif mode == "pre-act-18":
        student = PreActResNet18(num_classes=args.num_classes,
                                 normalize=args.normalize)
        # TODO: check parallel
        if args.use_data_parallel and torch.cuda.device_count() > 1:
            student = nn.DataParallel(student)
        student.to(args.device)
        # student = nn.DataParallel(student).to(args.device)
        student.train()

    else:  # teacher
        # python train.py --batch_size 1000 --mode teacher --normalize 0 --model_id teacher_unnormalized --lr_mode 2 --epochs 100 --dataset CIFAR10 --dropRate 0.3
        # python train.py --batch_size 1000 --mode teacher --normalize 1 --model_id teacher_normalized --lr_mode 2 --epochs 100 --dataset CIFAR10 --dropRate 0.3
        # TODO: change back
        student = Net_Arch(n_classes=args.num_classes,
                           depth=deep_full,
                           widen_factor=10,
                           normalize=args.normalize,
                           dropRate=0.3)
        # student = mobilenet_v3_large(pretrained=False)
        # student.classifier[-1] = nn.Linear(1280, args.num_classes)
        # TODO: check parallel
        if args.use_data_parallel and torch.cuda.device_count() > 1:
            student = nn.DataParallel(student)
        student.to(args.device)
        # student = nn.DataParallel(student).to(args.device)
        student.train()
        # Alternate student models: [lr_max = 0.01, epochs = 100], [preactresnet], [dropRate]

    return student, teacher


# srun --partition rtx6000 --gres=gpu:4 -c 40 --mem=40G python train.py --batch_size 1000 --mode teacher --normalize 0 --model_id teacher_unnormalized --lr_mode 2 --epochs 100 --dataset CIFAR10
if __name__ == "__main__":
    # restrict GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # force reorder
    # TODO: add CUDA_VISIBLE_DEVICES into params config
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # parse command options
    args: Args = Args().parse_args()

    # load config file if provided
    # TODO: test load config file
    if args.config_file is not None:
        args.load(args.config_file)

    # decide device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # prepare model output dir path
    # TODO: change to parent folder
    cwd = Path.cwd()

    # TODO: add model output option
    model_dir_path = cwd / "_model"
    model_dataset_cat_dir_path = model_dir_path / f"{args.dataset}"
    model_complete_dir_path = model_dataset_cat_dir_path / f"model_{args.model_id}"

    # root = f"./ting_models/{args.dataset}"
    # model_dir = f"{root}/model_{args.model_id}"

    if args.concat:
        # model_dir += f"concat_{args.concat_factor}"
        model_complete_dir_path = model_dataset_cat_dir_path / f"model_{args.model_id}_concat_{args.concat_factor}"

    args.model_complete_dir_path = model_complete_dir_path

    # create folder if not exists
    model_complete_dir_path.mkdir(parents=True, exist_ok=True)

    # TODO: change type in Args class
    args.device = device

    # TODO: use set all seed
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    n_class = {
        "CIFAR10": 10,
        "CIFAR100": 100,
        "AFAD": 26,
        "SVHN": 10,
        "ImageNet": 1000,
        "CIFAR10-Cat-Dog": 2,
        "STL10-Cat-Dog": 2,
        "Kaggle-Cat-Dog": 2,
        "CIFAR10-STL10-Cat-Dog": 2,
        "CIFAR10-Kaggle-Cat-Dog": 2,
        "STL10-Kaggle-Cat-Dog": 2,
        "CIFAR10-STL10-Kaggle-Cat-Dog": 2,
    }
    args.num_classes = n_class[args.dataset]

    # save config
    args.save(path=model_complete_dir_path / "train_config.json")

    # setup output stream
    bothout = Unbuffered(file_path=model_complete_dir_path / "logs.txt")

    TIMESTAMP = (datetime.utcnow() + timedelta(hours=8)).strftime("%y-%m-%d %H:%M")
    print(TIMESTAMP, file=bothout)
    startTime = time.process_time()
    print("Model Directory:", model_complete_dir_path, file=bothout)
    trainer(args)
    endTime = time.process_time()
    print(f"Time taken: {endTime - startTime:.2f} s", file=bothout)
