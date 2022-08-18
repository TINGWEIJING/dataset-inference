from __future__ import absolute_import

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet, mobilenet_v3_large, resnet18

import params
from funcs import *
from models import *
from params import Args
from utils.logger import Unbuffered

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


def train_one_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
):
    '''
    ResNet normal training
    '''
    # constant
    train_size = len(train_dataloader.dataset)

    # return variable
    running_loss = 0  # total loss of one epoch
    avg_running_loss = 0
    avg_sole_batch_loss_list = []
    total_correct_num = 0  # num of correct pred
    acc = 0

    for batch_sample in tqdm(
        train_dataloader,
        desc="Batch training",
        leave=True,
        total=len(train_dataloader)
    ):
        inputs: torch.Tensor
        labels: torch.Tensor
        inputs, labels = batch_sample

        # move to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        preds: torch.Tensor = model(inputs)
        loss: torch.Tensor = loss_fn(preds, labels)  # return avg loss of current batch
        loss.backward()
        optimizer.step()

        # statistics
        avg_sole_batch_loss = loss.detach().item()
        sole_batch_loss = avg_sole_batch_loss * inputs.size(0)  # compute total single batch loss
        running_loss += sole_batch_loss
        correct_num = (preds.argmax(1) == labels).type(torch.float).sum().item()
        total_correct_num += correct_num

        avg_sole_batch_loss_list.append(avg_sole_batch_loss)

    avg_running_loss = running_loss / train_size
    acc = (total_correct_num / train_size) * 100

    return avg_running_loss, avg_sole_batch_loss_list, int(total_correct_num), acc


def val_test_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
        device: torch.device,
):
    '''
    ResNet normal testing
    '''
    # constant
    dataset_size = len(dataloader.dataset)

    # return variable
    running_loss = 0  # total loss of one epoch
    avg_running_loss = 0
    avg_sole_batch_loss_list = []
    total_correct_num = 0  # num of correct pred
    acc = 0

    with torch.no_grad():
        for batch_sample in tqdm(
            dataloader,
            desc="Batch testing",
            leave=True,
            total=len(dataloader)
        ):
            inputs: torch.Tensor
            labels: torch.Tensor
            inputs, labels = batch_sample

            # move to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            preds: torch.Tensor = model(inputs)
            loss: torch.Tensor = loss_fn(preds, labels)  # return avg loss of current batch

            # statistics
            avg_sole_batch_loss = loss.detach().item()
            sole_batch_loss = avg_sole_batch_loss * inputs.size(0)  # compute total single batch loss
            running_loss += sole_batch_loss
            correct_num = (preds.argmax(1) == labels).type(torch.float).sum().item()

            total_correct_num += correct_num

            avg_sole_batch_loss_list.append(avg_sole_batch_loss)

        avg_running_loss = running_loss / dataset_size
        acc = (total_correct_num / dataset_size) * 100

    return avg_running_loss, avg_sole_batch_loss_list, int(total_correct_num), acc


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
    elif args.opt_type == "Adam":
        opt = optim.Adam(student.parameters(), lr=args.lr_max)
    else:
        opt = optim.Adam(student.parameters(), lr=0.1)

    # learning rate scheduler
    if args.lr_mode == 5:
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(args.epochs * 0.3), int(args.epochs * 0.8)], gamma=0.1)
        loss_fn = torch.nn.CrossEntropyLoss().to(args.device)
    else:
        lr_schedule = lr_scheduler(args)
        loss_fn = None
    t_start = 0  # represent iteration/epoch starting number

    # TODO: test resume script
    if args.resume:
        location = f"{args.model_dir}/iter_{str(args.resume_iter)}.pt"
        t_start = args.resume_iter + 1
        student.load_state_dict(torch.load(location, map_location=device))

    # start training
    train_epoch_acc_list = []
    train_epoch_avg_loss_list = []
    train_mini_batch_avg_loss_list = []
    latest_train_correct_num = 0
    latest_train_acc = 0

    val_epoch_acc_list = []
    val_epoch_avg_loss_list = []
    val_mini_batch_avg_loss_list = []
    latest_val_correct_num = 0
    latest_val_acc = 0
    is_80_acc_model_saved = False
    is_85_acc_model_saved = False
    is_90_acc_model_saved = False
    for t in range(t_start, args.epochs):
        if args.lr_mode == 5:  # ResNet normal training and evaluation iteration
            # training
            result = train_one_epoch(
                model=student,
                train_dataloader=train_loader,
                optimizer=opt,
                loss_fn=loss_fn,
                device=args.device,
            )
            train_avg_running_loss, train_avg_sole_batch_loss_list, \
                train_total_correct_num, train_acc = result

            train_epoch_acc_list.append(train_acc)
            train_epoch_avg_loss_list.append(train_avg_running_loss)
            train_mini_batch_avg_loss_list.extend(train_avg_sole_batch_loss_list)
            latest_train_correct_num = train_total_correct_num
            latest_train_acc = train_acc

            # validation
            result = val_test_one_epoch(
                model=student,
                dataloader=test_loader,
                loss_fn=loss_fn,
                device=args.device,
            )
            val_avg_running_loss, val_avg_sole_batch_loss_list, \
                val_total_correct_num, val_acc = result

            val_epoch_acc_list.append(val_acc)
            val_epoch_avg_loss_list.append(val_avg_running_loss)
            val_mini_batch_avg_loss_list.extend(val_avg_sole_batch_loss_list)
            latest_val_correct_num = val_total_correct_num
            latest_val_acc = val_acc

            # learning rate schedulers
            lr_schedule.step()
            print(f'{t+1:02d} | Train Loss: {train_avg_running_loss:0.4f}, Train Acc: {train_acc:0.2f}',
                  file=bothout)
            print(f'{t+1:02d} | Test Loss: {val_avg_running_loss:0.4f}, Test Acc: {val_acc:0.2f}',
                  file=bothout)

            # save model
            if train_acc >= 90 and not is_90_acc_model_saved:
                model_file = f"{args.model_complete_dir_path}/iter_{t}_acc_90.pt"
                try:
                    torch.save(student.module.state_dict(), model_file)
                except Exception as e:
                    print(e, file=bothout)
                    torch.save(student.state_dict(), model_file)
                is_90_acc_model_saved = True
                is_85_acc_model_saved = True
                is_80_acc_model_saved = True
            elif train_acc >= 85 and not is_85_acc_model_saved:
                model_file = f"{args.model_complete_dir_path}/iter_{t}_acc_85.pt"
                try:
                    torch.save(student.module.state_dict(), model_file)
                except Exception as e:
                    print(e, file=bothout)
                    torch.save(student.state_dict(), model_file)
                is_85_acc_model_saved = True
                is_80_acc_model_saved = True
            elif train_acc >= 80 and not is_80_acc_model_saved:
                model_file = f"{args.model_complete_dir_path}/iter_{t}_acc_80.pt"
                try:
                    torch.save(student.module.state_dict(), model_file)
                except Exception as e:
                    print(e, file=bothout)
                    torch.save(student.state_dict(), model_file)
                is_80_acc_model_saved = True

        else:
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
            print(f'Epoch: {t}, Test Loss: {test_loss:.3f} Test Acc: {test_acc:.3f}',
                  file=bothout)

            # ! The paper didn't use MNIST datasets, this is for testing I guess
            # if args.dataset == "MNIST":
            #     torch.save(student.state_dict(), f"{args.model_complete_dir_path}/iter_{t}.pt")
            # elif (t+1) % 25 == 0:  # save every 25 iteration
            #     torch.save(student.state_dict(), f"{args.model_complete_dir_path}/iter_{t}.pt")

            # save model
            if train_acc >= 0.9 and not is_90_acc_model_saved:
                model_file = f"{args.model_complete_dir_path}/iter_{t}_acc_90.pt"
                try:
                    torch.save(student.module.state_dict(), model_file)
                except Exception as e:
                    print(e, file=bothout)
                    torch.save(student.state_dict(), model_file)
                is_90_acc_model_saved = True
                is_85_acc_model_saved = True
                is_80_acc_model_saved = True
            elif train_acc >= 0.85 and not is_85_acc_model_saved:
                model_file = f"{args.model_complete_dir_path}/iter_{t}_acc_85.pt"
                try:
                    torch.save(student.module.state_dict(), model_file)
                except Exception as e:
                    print(e, file=bothout)
                    torch.save(student.state_dict(), model_file)
                is_85_acc_model_saved = True
                is_80_acc_model_saved = True
            elif train_acc >= 0.80 and not is_80_acc_model_saved:
                model_file = f"{args.model_complete_dir_path}/iter_{t}_acc_80.pt"
                try:
                    torch.save(student.module.state_dict(), model_file)
                except Exception as e:
                    print(e, file=bothout)
                    torch.save(student.state_dict(), model_file)
                is_80_acc_model_saved = True

    try:
        torch.save(student.module.state_dict(), f"{args.model_complete_dir_path}/final.pt")
    except Exception as e:
        print(e, file=bothout)
        torch.save(student.state_dict(), f"{args.model_complete_dir_path}/final.pt")


def get_student_teacher(args: Args) -> Tuple[torch.nn.Module, torch.nn.Module]:
    '''
    Prepare student (threat) model architecture based on `args.dataset` and `args.mode` for training
    '''
    w_f = 2 if args.dataset == "CIFAR100" else 1
    net_mapper = {
        "CIFAR10": WideResNet,
        "CIFAR100": WideResNet,
        "AFAD": resnet34,
        "SVHN": ResNet_8x
    }
    Net_Arch: nn.Module = net_mapper.get(args.dataset, resnet18)  # ! Use ResNet18 otherwise
    mode = args.mode
    # ['zero-shot', 'fine-tune', 'extract-label', 'extract-logit', 'distillation', 'teacher']
    deep_full = 34 if args.dataset in ["SVHN", "AFAD"] else 28
    deep_half = 18 if args.dataset in ["SVHN", "AFAD"] else 16

    # loading teach model
    if mode in ["teacher", "independent", "pre-act-18"]:
        teacher = None
    else:
        deep = 34 if args.dataset in ["SVHN", "AFAD"] else 28
        # only for authors used model architecture
        if args.dataset in net_mapper.keys():
            teacher = Net_Arch(n_classes=args.num_classes,
                               depth=deep,
                               widen_factor=10,
                               normalize=args.normalize,
                               dropRate=0.3)
        else:  # for ResNet
            teacher = Net_Arch(
                pretrained=False,
                num_classes=args.num_classes,
            )
        # TODO: check parallel
        if args.dataset != "SVHN":
            if args.use_data_parallel and torch.cuda.device_count() > 1:
                teacher = nn.DataParallel(teacher)
        teacher.to(args.device)
        # teacher = nn.DataParallel(teacher).to(args.device) if args.dataset != "SVHN" else teacher.to(args.device)
        teacher_dir = "model_teacher_normalized" if args.normalize else "model_teacher_unnormalized"
        path = f"./_model/{args.dataset}/{teacher_dir}/final"
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
        path = f"./_model/{args.dataset}/{teacher_dir}/final"
        student = load(student, path)
        student.train()
        # assert(args.pseudo_labels)

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
        # assert(args.pseudo_labels)

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
        # only for authors used model architecture
        if "CIFAR-CINIC" in args.dataset:  # WideResNet
            student = WideResNet(n_classes=args.num_classes,
                                 depth=deep_full,
                                 widen_factor=10,
                                 normalize=args.normalize,
                                 dropRate=0.3)
            print("Is using WideResNet")
            # TODO: Remove this print
        elif args.dataset in net_mapper.keys():
            student = Net_Arch(n_classes=args.num_classes,
                               depth=deep_full,
                               widen_factor=10,
                               normalize=args.normalize,
                               dropRate=0.3)
        else:  # for ResNet
            student = Net_Arch(
                pretrained=False,
                num_classes=args.num_classes,
            )
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
        "CIFAR-CINIC-100-0": 10,
        "CIFAR-CINIC-90-10": 10,
        "CIFAR-CINIC-80-20": 10,
        "CIFAR-CINIC-60-40": 10,
        "CIFAR-CINIC-40-60": 10,
        "CIFAR-CINIC-20-80": 10,
        "CIFAR-CINIC-10-90": 10,
        "CIFAR-CINIC-0-100": 10,
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
