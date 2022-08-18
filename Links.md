# Resources Links
## TODO
- [X] Combine cat & dogs images
  - CIFAR 10
  - ImageNet
  - https://www.kaggle.com/c/dogs-vs-cats/data
- [X] Change CIFAR10 & ImageNet Label
- [X] Convert into 3 dataset methods
- [X] Calculate number of sample for train & test
- [X] Calculate normalize value
- [X] Check what is pseudolabel (for cifar10 only)
- [ ] Store loss & accuracy
- [X] Create download dataset flag
- [ ] Use mobilenet for extract
- [ ] Train a independent/parent model
  - [ ] CIFAR
  - [ ] STL
  - [ ] Kaggle
- [ ] Train labels extract
  - [ ] CIFAR
  - [ ] STL
  - [ ] Kaggle
- [ ] Train logits extract
  - [ ] CIFAR
  - [ ] STL
  - [ ] Kaggle
- [ ] Train fine tune
  - [ ] CIFAR - STL & Kaggle
  - [ ] STL - CIFAR & Kaggle
  - [ ] Kaggle - CIFAR & STL

## Model Threats Mode
- Teacher
  - Alone, no student
- Data Access
  - Model Distillation
    - pseudo_labels false
    - KLDivLoss
  - Diff Arch
    - not found
- Model Access
  - Fine Tuning
    - Use back teacher model as student model
    - pseudo_labels true
    - CrossEntropyLoss
  - Zero shot
    - Cannot do, dont have model
- Query Access
  - Labels
    - pseudo_labels true
    - CrossEntropyLoss
  - Logits
    - pseudo_labels true
    - KLDivLoss

## Combine Datasets
- [SO | Pytorch - Concatenating Datasets before using Dataloader](https://stackoverflow.com/questions/60840500/pytorch-concatenating-datasets-before-using-dataloader)
- [SO | How to merge two torch.utils.data dataloaders with a single operation](https://stackoverflow.com/questions/65621414/how-to-merge-two-torch-utils-data-dataloaders-with-a-single-operation)
- [SO | Combine Pytorch ImageFolder dataset with custom Pytorch dataset](https://stackoverflow.com/questions/62288855/combine-pytorch-imagefolder-dataset-with-custom-pytorch-dataset)
- [PyTorch Forum | How does ConcatDataset work?](https://discuss.pytorch.org/t/how-does-concatdataset-work/60083)
- [PyTorch Forum | Concat image datasets with different size and number of channels](https://discuss.pytorch.org/t/concat-image-datasets-with-different-size-and-number-of-channels/36362)
- [Medium | Manipulating Pytorch Datasets](https://medium.com/mlearning-ai/manipulating-pytorch-datasets-c58487ab113f#860e)

## Normalize Dataset
- [PyTorch Forum | Computing the mean and std of dataset](https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/23)
- [GitHub | Calculate mean and std](https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py)

## Parallel Training
- [PyTorch Doc | Optional: Data Parallelism](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)
  - Main script reference
- [PyTorch Doc | Multi-GPU Examples](https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html)
  - Good extra explanation on `DataParallel`
- [PyTorch Doc | Overview torch.nn.parallel.DistributedDataParallel](https://pytorch.org/tutorials/beginner/dist_overview.html#torch-nn-parallel-distributeddataparallel)
  - Read if got time
- [PyTorch Doc | Single-Machine Model Parallel Best Practices](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)
  - Hard to configure dynamically
  - Move layers to different gpu
- [PyTorch Doc | ]()
- []()

## Cat Dog Dataset
- [Kaggle | Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats/code?competitionId=3362&sortBy=voteCount)
- [Kaggle | [pytorch] cat vs dog](https://www.kaggle.com/code/jaeboklee/pytorch-cat-vs-dog/notebook)
- []()

## Mobilenet
- [SO | Fine Tuning Pretrained Model MobileNet_V3_Large PyTorch](https://stackoverflow.com/questions/69321848/fine-tuning-pretrained-model-mobilenet-v3-large-pytorch)
- []()
- []()