from __future__ import absolute_import

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import ipdb
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large

import params
from attacks import *
from funcs import *
from models import *
from params import Args
from train import epoch_test
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


def get_adversarial_vulnerability(args: Args,
                                  loader: DataLoader,
                                  model: torch.nn.Module,
                                  num_images=1000) -> Tuple[float, float, torch.Tensor]:
    '''
    '''
    train_loss, train_acc, train_n = 0, 0, 0
    batch_size = 100
    max_iter = num_images/batch_size
    full_dist = []
    ex_skipped = 0
    func = tqdm  # if stop == False else lambda x:x

    for i, batch in tqdm(enumerate(loader)):
        # NOTE: what is regressor_embed?
        if args.regressor_embed == 1:  # We need an extra set of `distinct images for training the confidence regressor
            if(ex_skipped < num_images):
                y = batch[1]
                ex_skipped += y.shape[0]
                continue

        for attack in [pgd_l1, pgd_l2, pgd_linf]:
            for target_i in range(10):
                X, y = batch[0].to(device), batch[1].to(device)
                # NOTE: check delta tensor shape
                delta = attack(model, X, y, args, target=y*0 + target_i) if attack is not None else 0
                # NOTE: check yp tensor shape
                yp = model(X+delta)
                yp = yp[0] if len(yp) == 4 else yp
                loss = nn.CrossEntropyLoss()(yp, y)
                train_loss += loss.item()*y.size(0)
                train_acc += (yp.max(1)[1] == y).sum().item()
                train_n += y.size(0)
                # calculate distance between boundaries
                distance_dict = {pgd_linf: norms_linf_squeezed,
                                 pgd_l1: norms_l1_squeezed,
                                 pgd_l2: norms_l2_squeezed}
                distances = distance_dict[attack](delta)
                full_dist.append(distances.cpu().detach())

        # safety loop break
        if i+1 >= max_iter:
            break

    # NOTE: check tensor shape
    full = [x.view(-1, 1) for x in full_dist]
    full_d = torch.cat(full, dim=1)

    return train_loss / train_n, train_acc / train_n, full_d


def get_random_label_only(args: Args, loader, model, num_images=1000):
    # output stream
    bothout = Unbuffered()
    print("Getting random attacks", file=bothout)
    batch_size = args.batch_size
    max_iter = num_images/batch_size
    lp_dist = [[], [], []]
    ex_skipped = 0
    for i, batch in enumerate(loader):
        print(f'======== iter: {i} ========', file=bothout)
        if args.regressor_embed == 1:  # We need an extra set of `distinct images for training the confidence regressor
            if(ex_skipped < num_images):
                y = batch[1]
                ex_skipped += y.shape[0]
                continue
        for j, distance in enumerate(["linf", "l2", "l1"]):
            print(f'==== distance: {distance} ====', file=bothout)
            temp_list = []
            for target_i in range(10):  # 5 random starts
                print(f'== target_i: {target_i} ==', file=bothout)
                X, y = batch[0].to(device), batch[1].to(device)
                args.distance = distance
                # args.lamb = 0.0001
                preds = model(X)
                targets = None
                delta = rand_steps(model, X, y, args, target=targets)
                yp = model(X+delta)
                distance_dict = {"linf": norms_linf_squeezed, "l1": norms_l1_squeezed, "l2": norms_l2_squeezed}
                distances = distance_dict[distance](delta)
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            # temp_dist = [batch_size, num_classes)]
            temp_dist = torch.cat(temp_list, dim=1)
            lp_dist[j].append(temp_dist)
        if i+1 >= max_iter:
            break
    # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
    lp_d = [torch.cat(lp_dist[i], dim=0).unsqueeze(-1) for i in range(3)]
    # full_d = [num_images, num_classes, num_attacks]
    full_d = torch.cat(lp_d, dim=-1)
    print(f"full_d.shape: {full_d.shape}", file=bothout)

    return full_d


def get_topgd_vulnerability(args: Args, loader, model, num_images=1000):
    batch_size = args.batch_size
    max_iter = num_images/batch_size
    lp_dist = [[], [], []]
    ex_skipped = 0
    for i, batch in enumerate(loader):
        if args.regressor_embed == 1:  # We need an extra set of `distinct images for training the confidence regressor
            if(ex_skipped < num_images):
                y = batch[1]
                ex_skipped += y.shape[0]
                continue
        for j, distance in enumerate(["linf", "l2", "l1"]):
            temp_list = []
            for target_i in range(10):
                X, y = batch[0].to(device), batch[1].to(device)
                args.distance = distance
                # args.lamb = 0.0001
                preds = model(X)
                tgt = target_i + 1 if args.dataset == "CIFAR100" else target_i
                targets = torch.argsort(preds, dim=-1, descending=True)[:, tgt]
                delta = mingd(model, X, y, args, target=targets)
                yp = model(X+delta)
                distance_dict = {"linf": norms_linf_squeezed,
                                 "l1": norms_l1_squeezed,
                                 "l2": norms_l2_squeezed}
                distances = distance_dict[distance](delta)
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            # temp_dist = [batch_size, num_classes)]
            temp_dist = torch.cat(temp_list, dim=1)
            lp_dist[j].append(temp_dist)
        if i+1 >= max_iter:
            break
    # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
    lp_d = [torch.cat(lp_dist[i], dim=0).unsqueeze(-1) for i in range(3)]
    # full_d = [num_images, num_classes, num_attacks]
    full_d = torch.cat(lp_d, dim=-1)
    print(f"full_d.shape: {full_d.shape}", file=bothout)

    return full_d


def get_mingd_vulnerability(args: Args,
                            loader: DataLoader,
                            model: torch.nn.Module,
                            num_images=1000):
    batch_size = args.batch_size
    max_iter = num_images/batch_size
    lp_dist = [[], [], []]
    ex_skipped = 0
    for i, batch in enumerate(loader):  # for each batch of data
        print(f'======== iter: {i} ========', file=bothout)
        if args.regressor_embed == 1:  # We need an extra set of `distinct images for training the confidence regressor
            if(ex_skipped < num_images):
                y = batch[1]
                ex_skipped += y.shape[0]
                continue
        for j, distance in enumerate(["linf", "l2", "l1"]):  # for each norm func
            print(f'==== distance: {distance} ====', file=bothout)
            temp_list = []
            for target_i in range(args.num_classes):  # for each class of data
                print(f'== target_i: {target_i} ==', file=bothout)
                X, y = batch[0].to(device), batch[1].to(device)
                args.distance = distance  # distance type in string
                # args.lamb = 0.0001 (unknown intention, maybe is zombie code left by authors)

                # get difference in distance between class boundaries
                # `y*0 + target_i` same as torch.full(y.shape, target_i)
                delta = mingd(model, X, y, args, target=y*0 + target_i)
                yp = model(X+delta)  # (unknown intention, maybe is zombie code left by authors)

                distance_dict = {"linf": norms_linf_squeezed,
                                 "l1": norms_l1_squeezed,
                                 "l2": norms_l2_squeezed}
                distances: torch.Tensor = distance_dict[distance](delta)
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            # temp_dist = [batch_size, num_classes)]
            temp_dist = torch.cat(temp_list, dim=1)
            lp_dist[j].append(temp_dist)
        if i+1 >= max_iter:
            break
    # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
    lp_d = [torch.cat(lp_dist[i], dim=0).unsqueeze(-1) for i in range(3)]
    # full_d = [num_images, num_classes, num_attacks]
    full_d = torch.cat(lp_d, dim=-1)
    print(f"full_d.shape: {full_d.shape}", file=bothout)

    return full_d


def feature_extractor(args: Args):
    '''
    '''
    # output stream
    bothout = Unbuffered()
    # load trained student model .pt file
    if args.dataset != "ImageNet":
        train_loader, test_loader = get_dataloaders(
            args.victim_dataset,
            args.batch_size,
            normalize=args.normalize,
            download=args.download_dataset,
            pseudo_labels=False,
            train_shuffle=False
        )
        student, _ = get_student_teacher(args)  # teacher is not needed # TODO: use Mobilenet
        location = f"{args.model_complete_dir_path}/final.pt"
        try:  # TODO: solve error
            student = student.to(args.device)
            student.load_state_dict(torch.load(location, map_location=args.device))
        except Exception as e:
            student = nn.DataParallel(student).to(args.device)
            student.load_state_dict(torch.load(location, map_location=args.device))
    else:
        train_loader, test_loader = get_dataloaders(
            args.dataset,
            args.batch_size,
            normalize=True,
            pseudo_labels=False,
            train_shuffle=False
        )
        import torchvision.models as models
        imagenet_arch = {"alexnet": models.alexnet,
                         "inception": models.inception_v3,
                         "wrn": models.wide_resnet50_2}
        I_Arch = imagenet_arch[args.imagenet_architecture]
        student = I_Arch(pretrained=True)
        student = nn.DataParallel(student).to(args.device)

    # no training
    student.eval()

    # _, train_acc  = epoch_test(args, train_loader, student)
    _, test_acc = epoch_test(args,
                             test_loader,
                             student,
                             stop=True)
    print(f'Model: {args.model_complete_dir_path}', file=bothout)
    print(f'Test Acc: {test_acc:.3f}', file=bothout)

    mapping = {'pgd': get_adversarial_vulnerability,
               'topgd': get_topgd_vulnerability,
               'mingd': get_mingd_vulnerability,
               'rand': get_random_label_only}

    func = mapping[args.feature_type]

    test_d = func(args, test_loader, student)
    torch.save(test_d, f"{args.feature_complete_dir_path}/test_{args.feature_type}_vulnerability_2.pt")

    train_d = func(args, train_loader, student)
    torch.save(train_d, f"{args.feature_complete_dir_path}/train_{args.feature_type}_vulnerability_2.pt")


def get_student_teacher(args: Args) -> Tuple[torch.nn.Module, None]:
    '''
    Prepare student (threat) model architecture based on `args.dataset` and `args.mode` for loading
    '''
    w_f = 2 if args.dataset == "CIFAR100" else 1
    net_mapper = {"CIFAR10": WideResNet,
                  "CIFAR100": WideResNet,
                  "AFAD": resnet34,
                  "SVHN": ResNet_8x}
    Net_Arch: Union[torch.nn.Module, ResNet] = net_mapper.get(args.dataset, WideResNet)  # ! Use WideResNet otherwise
    teacher = None
    mode = args.mode
    # ['zero-shot', 'prune', 'fine-tune', 'extract-label', 'extract-logit', 'distillation', 'teacher']
    deep_full = 34 if args.dataset in ["SVHN", "AFAD"] else 28
    deep_half = 18 if args.dataset in ["SVHN", "AFAD"] else 16

    if mode == 'zero-shot':
        student = Net_Arch(n_classes=args.num_classes,
                           depth=deep_half,
                           widen_factor=1,
                           normalize=args.normalize)

    elif mode == "prune":
        raise("Not handled")

    elif mode == "random":
        assert (args.dataset == "SVHN")
        # python generate_features.py --feature_type rand --dataset SVHN --batch_size 500 --mode random --normalize 1 --model_id random_normalized
        student = Net_Arch(n_classes=args.num_classes,
                           depth=deep_half,
                           widen_factor=8,
                           normalize=args.normalize)

    elif mode == "fine-tune":
        # python generate_features.py --batch_size 500 --mode fine-tune --normalize 0 --model_id fine-tune_unnormalized
        # python generate_features.py --batch_size 500 --mode fine-tune --normalize 1 --model_id fine-tune_normalized
        student = Net_Arch(n_classes=args.num_classes,
                           depth=deep_full,
                           widen_factor=10,
                           normalize=args.normalize)

    elif mode in ["extract-label", "extract-logit"]:
        # python generate_features.py --batch_size 500 --mode extract-label --normalize 0 --model_id extract-label_unnormalized
        # python generate_features.py --batch_size 500 --mode extract-label --normalize 1 --model_id extract-label_normalized
        student = Net_Arch(n_classes=args.num_classes,
                           depth=deep_half,
                           widen_factor=w_f,
                           normalize=args.normalize)

    elif mode in ["distillation", "independent"]:
        dR = 0.3 if mode == "independent" else 0.0
        # python generate_features.py --batch_size 500 --mode distillation --normalize 0 --model_id distillation_unnormalized
        # python generate_features.py --batch_size 500 --mode distillation --normalize 1 --model_id distillation_normalized
        student = Net_Arch(n_classes=args.num_classes,
                           depth=deep_half,
                           widen_factor=w_f,
                           normalize=args.normalize,
                           dropRate=dR)

    elif mode == "pre-act-18":
        student = PreActResNet18(num_classes=args.num_classes,
                                 normalize=args.normalize)

    else:  # teacher
        # python generate_features.py --feature_type rand --dataset SVHN --batch_size 500 --mode teacher --normalize 1 --model_id teacher_normalized
        # python generate_features.py --batch_size 500 --mode teacher --normalize 0 --model_id teacher_unnormalized --dataset CIFAR10
        # python generate_features.py --batch_size 500 --mode teacher --normalize 1 --model_id teacher_normalized --dataset CIFAR10
        # TODO: change back
        student = Net_Arch(n_classes=args.num_classes,
                           depth=deep_full,
                           widen_factor=10,
                           normalize=args.normalize,
                           dropRate=0.3)
        # student = mobilenet_v3_large(pretrained=False)
        # Alternate student models: [lr_max = 0.01, epochs = 100], [preactresnet], [dropRate]

    return student, teacher


if __name__ == "__main__":
    # restrict GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # parse command options
    args: Args = Args().parse_args()

    # load config file if provided
    if args.config_file is not None:
        args.load(args.config_file)

    # decide device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # prepare model input dir path
    # TODO: change to parent folder
    cwd = Path.cwd()

    # TODO: add model output option
    model_dir_path = cwd / "_model"
    model_dataset_cat_dir_path = model_dir_path / f"{args.dataset}"
    model_complete_dir_path = model_dataset_cat_dir_path / f"model_{args.model_id}"

    # prepare feature extration output dir path
    feature_dir_path = cwd / "_feature"
    feature_dataset_cat_dir_path = feature_dir_path / f"{args.victim_dataset}"  # ! use victim dataset
    feature_complete_dir_path = feature_dataset_cat_dir_path / f"model_{args.model_id}-{args.dataset}"
    if args.regressor_embed == 1:
        feature_complete_dir_path = feature_dataset_cat_dir_path / f"model_{args.model_id}_cr"

    args.model_complete_dir_path = model_complete_dir_path
    args.feature_complete_dir_path = feature_complete_dir_path

    # create folder if not exists
    feature_complete_dir_path.mkdir(parents=True, exist_ok=True)

    # TODO: change type in Args class
    args.device = device

    # TODO: use set all seed
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
    bothout = Unbuffered(file_path=feature_complete_dir_path / "logs.txt")

    TIMESTAMP = (datetime.utcnow() + timedelta(hours=8)).strftime("%y-%m-%d %H:%M")
    print(TIMESTAMP, file=bothout)
    startTime = time.process_time()
    print("Model Directory:", model_complete_dir_path, file=bothout)
    print("Feature Directory:", feature_complete_dir_path, file=bothout)
    feature_extractor(args)
    endTime = time.process_time()
    print(f"Time taken: {endTime - startTime:.2f} s", file=bothout)
