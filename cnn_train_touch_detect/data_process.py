import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image
import cv2
import os


def Myloader(path):
    return Image.open(path).convert('RGB')


# Get a list of paths and labels
def init_process(directory_name, class_num):
    data = []
    i = 0
    for filename in os.listdir(directory_name):
        data.append([directory_name + "/" + filename, class_num])
        i = i + 1
    return data


class MyDataset(Dataset):
    def __init__(self, data, transform, loader):
        self.data = data
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


def find_label(str):
    first, last = 0, 0
    for i in range(len(str) - 1, -1, -1):
        if str[i] == '%' and str[i - 1] == '_':
            last = i - 1
        if (str[i] == 'F' or str[i] == 'n' or str[i] == 'P') and str[i - 1] == '/':
            first = i
            break

    name = str[first:last]
    if name == 'Fully_capped':
        return 1
    elif name == 'no_capped':
        return 0
    # else:
    #     return 0


def load_data(batch_size):
    # percentage of training set to use as validation
    valid_size = 0.2

    # Data Augmentation
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.CenterCrop(224),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalization
    ])

    path1 = '/home/acldifstudent2/Code/Jiayun/PyTouch/img_anothersensor/train/class0'
    data1 = init_process(path1, 0)
    path2 = '/home/acldifstudent2/Code/Jiayun/PyTouch/img_anothersensor/train/class1'
    data2 = init_process(path2, 1)

    train_data = data1 + data2
    train = MyDataset(train_data, transform=train_tf, loader=Myloader)

    # obtain training indices that will be used for validation
    num_train = len(train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # load training data in batches
    train_loader = DataLoader(dataset=train, batch_size=batch_size, sampler=train_sampler,
                              num_workers=0, pin_memory=True)

    # load validation data in batches
    valid_loader = DataLoader(dataset=train, batch_size=batch_size, sampler=valid_sampler,
                              num_workers=0, pin_memory=True)

    # train_data = DataLoader(dataset=train, batch_size=5, shuffle=True, num_workers=0, pin_memory=True)
    # test_data = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    return train_loader, valid_loader


