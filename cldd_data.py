import os
import random

import torchvision
# from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
torch.manual_seed(1)
import matplotlib.pyplot as plt


def MSL_GEN(new_train_label, new_train_set, new_train_TF_label_index,  n):
    new_train_SL_label_index = [[0] * n] * new_train_set.size()[0]
    for i in range(new_train_set.size()[0]):
        r_m = random.sample(range(3), n)
        flag = 0
        for k in range(n):
            if new_train_label[i] == r_m[k]:
                new_train_TF_label_index[i] = r_m[k]
                flag = 1
                for j in range(n):
                    new_train_SL_label_index[i][j] = r_m[j]
        if flag == 0:
            new_train_TF_label_index[i] = 3
            for j in range(n):
                new_train_SL_label_index[i][j] = r_m[j]
    new_train_SL_label_index = torch.tensor(new_train_SL_label_index)
    return new_train_TF_label_index, new_train_SL_label_index

def read_data_cldd_TF_MSL(i_train_iter, n, conceal_label):
    random.seed(i_train_iter)
    P0 = 0
    P1 = 1
    P2 = 2
    train_datasets = 'D:/code/data/DDSMA/DDSMA/train'
    test_datasets = 'D:/code/data/DDSMA/DDSMA/test'
    transform_resize = transforms.Compose([
        transforms.Resize((156, 156)),
        transforms.CenterCrop((128, 128)),
        transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # transform_rotation = transforms.Compose([
    #     transforms.Resize((156, 156)),
    #     transforms.RandomCrop((128, 128)),
    #     # transforms.RandomErasing(p=1, scale=(0.02, 0.03), ratio=(0.3, 3.3), value=(1,1,1), inplace=False),
    #     # transforms.RandomRotation(20),
    #     transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])
    train_set_resize = torchvision.datasets.ImageFolder(
        train_datasets, transform=transform_resize,
        target_transform=None,
        is_valid_file=None)
    # train_set_rotation = torchvision.datasets.ImageFolder(
    #     train_datasets, transform=transform_rotation,
    #     target_transform=None,
    #     is_valid_file=None)
    # train_set = torch.utils.data.ConcatDataset([train_set_resize, train_set_rotation])
    train_set = train_set_resize
    test_set = torchvision.datasets.ImageFolder(
        test_datasets, transform=transform_resize,
        target_transform=None,
        is_valid_file=None)
    # print('train_set_rotation', train_set_rotation.classes)
    print('train_set_resize', train_set_resize.classes)
    # print(train_set.class_to_idx)
    # print(test_set.class_to_idx)
    # print(train_set.imgs)
    # print(train_set[1][0].tolist())
    train_label = []
    train_data = []
    test_label = []
    test_data = []
    for i in range(4080):
        train_label.append(train_set[i][1])
        train_data.append(train_set[i][0].tolist())

    for i in range(1020):
        test_label.append(test_set[i][1])
        test_data.append(test_set[i][0].tolist())
    # return 0
    # print(train_data)

    train_label = torch.tensor(train_label)
    test_label = torch.tensor(test_label)
    train_data = torch.tensor(train_data)
    test_data = torch.tensor(test_data)

    print(train_data.size())
    num_data_t = 0
    index = []
    for i in range(train_label.size()[0]):
        if train_label[i] == P0 or train_label[i] == P1 or train_label[i] == P2:
            index.append(i)
            num_data_t = num_data_t + 1

    index = torch.tensor(index)
    new_train_set = torch.index_select(train_data, dim=0, index=index)
    new_train_label = torch.index_select(train_label, dim=0, index=index)
    new_train_label_index = torch.index_select(train_label, dim=0, index=index)
    new_train_TF_label_index = torch.index_select(train_label, dim=0, index=index)


    for i in range(new_train_set.size()[0]):
        if new_train_label_index[i] == P0:
            new_train_label[i] = 0
        if new_train_label_index[i] == P1:
            new_train_label[i] = 1
        if new_train_label_index[i] == P2:
            new_train_label[i] = 2


    print("data N = ", n)
    new_train_TF_label_index, new_train_SL_label_index = MSL_GEN(new_train_label, new_train_set, new_train_TF_label_index, n)
    num_ALL = new_train_set.size()[0]
    num_t_label = torch.eq(new_train_TF_label_index, 2).sum()
    num_H0_label = torch.eq(new_train_TF_label_index, 0).sum()
    num_H1_label = torch.eq(new_train_TF_label_index, 1).sum()
    num_0_label = torch.eq(new_train_label, 0).sum()
    num_1_label = torch.eq(new_train_label, 1).sum()
    data_prior = num_t_label.float()/num_ALL
    data_prior_hide = (num_0_label+ num_1_label - num_H0_label-num_H1_label)/num_t_label.float()

    data_prior_i = torch.eq(new_train_label_index, P1).sum()/num_ALL/(torch.eq(new_train_TF_label_index, P1).sum()/num_ALL)

    print('data_prior', data_prior)
    print('data_prior_hide', data_prior_hide)
    # print(index)
    # print(new_train_label.size())
    index = []
    for i in range(test_data.size()[0]):
        if test_label[i] == P0 or test_label[i] == P1 or test_label[i] == P2:
            index.append(i)
    index = torch.tensor(index)
    new_test_set = torch.index_select(test_data, dim=0, index=index)
    new_test_label = torch.index_select(test_label, dim=0, index=index)
    new_test_label_index = torch.index_select(test_label, dim=0, index=index)
    for i in range(new_test_set.size()[0]):
        if new_test_label_index[i] == P0:
            new_test_label[i] = 0
        if new_test_label_index[i] == P1:
            new_test_label[i] = 1
        if new_test_label_index[i] == P2:
            new_test_label[i] = 2



    return new_train_set, new_train_TF_label_index, new_train_SL_label_index, new_test_set, new_test_label, data_prior, data_prior_hide


