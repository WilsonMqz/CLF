import os
import random

import torchvision.datasets.mnist as mnist
import torch
torch.manual_seed(1)


def read_data_minist(data_set, i_train_iter):
    num_data = 25000
    random.seed(i_train_iter)
    if data_set == 1:
        root="D:/code/data/minist/minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images.idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels.idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images.idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels.idx1-ubyte'))
        )
    elif data_set ==2:
        root = "D:/code/data/fashion_minist/fashion_minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )
    else:
        root = "D:/code/data/Kuzushiji_minist/Kuzushiji_minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )


    # index = []
    # for i in range(train_set[0].size()[0]):
    #     if train_set[1][i] == 6 or train_set[1][i] == 2 or train_set[1][i] == 3:
    #         index.append(i)
    # index = torch.tensor(index)
    # new_train_set = torch.index_select(train_set[0],dim=0, index = index)
    # new_train_label = torch.index_select(train_set[1],dim=0, index = index)
    # rand_index = random.sample(range(new_train_set.size()[0]), num_data)
    # for i in range(new_train_set.size()[0]):
    #     if new_train_label[i] == 6:
    #         new_train_label[i] = 0
    #     if new_train_label[i] == 2:
    #         new_train_label[i] = 1
    #     if new_train_label[i] == 3:
    #         new_train_label[i] = 2
    #     if i in rand_index[:int(num_data/2)] and new_train_label[i] == 1:
    #         new_train_label[i] = 0
    #     if i in rand_index[int(num_data/2):]:
    #         new_train_label[i] = 2
    # # print(index)
    # # print(new_train_label.size())
    #
    #
    # index = []
    # for i in range(test_set[0].size()[0]):
    #     if test_set[1][i] == 6 or test_set[1][i] == 2 or test_set[1][i] == 3:
    #         index.append(i)
    #
    # index = torch.tensor(index)
    # new_test_set = torch.index_select(test_set[0],dim=0, index = index)
    # new_test_label = torch.index_select(test_set[1],dim=0, index = index)
    # for i in range(new_test_set.size()[0]):
    #     if new_test_label[i] == 6:
    #         new_test_label[i] = 0
    #     if new_test_label[i] == 2:
    #         new_test_label[i] = 1
    #     if new_test_label[i] == 3:
    #         new_test_label[i] = 2
    # # print(new_test_label.size())
    # print('data set  Parameter： unknown set 3 known set 2 unlabel set 6 number of unknown and unlabel  {} {}'.format(
    #     num_data/2, num_data/2))
    # new_train_set = new_train_set.unsqueeze(1)
    # new_test_set = new_test_set.unsqueeze(1)
    # return new_train_set, new_train_label, new_test_set, new_test_label



    train_label = torch.tensor(train_set[1])
    test_label = torch.tensor(test_set[1])
    train_data = torch.tensor(train_set[0])
    test_data = torch.tensor(test_set[0])
    data_prior = [0,0,0,0,0,0]
    num_unlabel_known = 0
    num_unknown_known = 0
    num_unknown_unlabel = 0
    num_unlabel_unlabel = 0
    index = []
    for i in range(train_label.size()[0]):
        if train_label[i] == 6 or train_label[i] == 5 or train_label[i] == 7 or train_label[i] == 2 or train_label[i] == 3 or train_label[i] == 8:
            index.append(i)
    index = torch.tensor(index)
    new_train_set = torch.index_select(train_data, dim=0, index=index)
    new_train_label = torch.index_select(train_label, dim=0, index=index)
    rand_index = random.sample(range(new_train_set.size()[0]), num_data)
    for i in range(new_train_set.size()[0]):
        if new_train_label[i] == 6 or new_train_label[i] == 5:
            new_train_label[i] = 0
        if new_train_label[i] == 2 or new_train_label[i] == 7:
            new_train_label[i] = 1
        if new_train_label[i] == 3 or new_train_label[i] == 8:
            new_train_label[i] = 2
        if i in rand_index[:int(num_data / 2)] and (new_train_label[i] == 0 or new_train_label[i] == 1):
            if new_train_label[i] == 0:
                num_unlabel_known = num_unlabel_known + 1 # unlabel 中含有的标记样本个数
            elif new_train_label[i] == 1:
                num_unlabel_unlabel = num_unlabel_unlabel + 1
            new_train_label[i] = 1
        if i in rand_index[int(num_data / 2):]:
            if new_train_label[i] == 0:
                num_unknown_known = num_unknown_known + 1
            elif new_train_label[i] == 1:
                num_unknown_unlabel = num_unknown_unlabel + 1
            new_train_label[i] = 2
    num_unlabel = torch.eq(new_train_label, 1).sum()
    num_unknown = torch.eq(new_train_label, 2).sum()
    num_known_class = torch.eq(train_label, 6).sum() + torch.eq(train_label, 5).sum() # 真实类别 样本个数
    num_unlabel_class = torch.eq(train_label, 2).sum() + torch.eq(train_label, 7).sum()
    num_unknown_class = torch.eq(train_label, 3).sum() + torch.eq(train_label, 8).sum()

    num_unlabel_known = torch.tensor(num_unlabel_known)
    num_unknown_known = torch.tensor(num_unknown_known)
    num_unknown_unlabel = torch.tensor(num_unknown_unlabel)
    num_unlabel_unlabel = torch.tensor(num_unlabel_unlabel)

    data_prior[0] = torch.tensor(1)                                  # p_1_1
    data_prior[1] = num_unlabel_known.float() / num_unlabel      #p_2_1
    data_prior[2] = (num_unlabel_class - num_unknown_unlabel).float() / num_unlabel      #p_2_2
    data_prior[3] = num_unknown_known.float() / num_unknown    #p_3_1
    data_prior[4] = num_unknown_unlabel.float() / num_unknown    #p_3_2
    data_prior[5] = num_unknown_class.float() / num_unknown

    data_class_prior = [0,0,0]
    data_class_prior[0] = num_known_class.float()/new_train_set.size()[0]
    data_class_prior[1] = num_unlabel_class.float() / new_train_set.size()[0]
    data_class_prior[2] = num_unknown_class.float() / new_train_set.size()[0]
    print(data_prior, data_class_prior,num_known_class, num_unlabel_class, num_unknown_class)
    # print(index)
    # print(new_train_label.size())

    index = []
    for i in range(test_data.size()[0]):
        if test_label[i] == 6 or test_label[i] == 5 or test_label[i] == 7 or test_label[i] == 2 or test_label[i] == 8:
            index.append(i)


    index = torch.tensor(index)
    new_test_set = torch.index_select(test_data, dim=0, index=index)
    new_test_label = torch.index_select(test_label, dim=0, index=index)
    for i in range(new_test_set.size()[0]):
        if new_test_label[i] == 6 or new_test_label[i] == 5:
            new_test_label[i] = 0
        if new_test_label[i] == 2 or new_test_label[i] == 7:
            new_test_label[i] = 1
        if new_test_label[i] == 3 or new_test_label[i] == 8:
            new_test_label[i] = 2
    # print(new_test_label.size())
    print('data set  Parameter： unknown set 3 known set 2 unlabel set 6 number of unknown and unlabel  {} {}'.format(
        num_data / 2, num_data / 2))



    print('train_x shape:%s, train_y shape:%s' % (train_data.shape, train_label.shape))
    print('test_x shape:%s, test_y shape:%s' % (test_data.shape, test_label.shape))
    return new_train_set.unsqueeze(1), new_train_label, new_test_set.unsqueeze(1), new_test_label, data_prior, data_class_prior



def read_data_minist_single(data_set, i_train_iter):
    num_data = 6000
    random.seed(i_train_iter)
    C_u_p = 1
    C_a_p = 1
    C_a_n = 1
    P = 9
    N = 8
    A = 7
    A_d = 6
    if data_set == 1:
        root="D:/code/data/minist/minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images.idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels.idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images.idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels.idx1-ubyte'))
        )
    elif data_set ==2:
        root = "D:/code/data/fashion_minist/fashion_minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )
    else:
        root = "D:/code/data/Kuzushiji_minist/Kuzushiji_minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )



    train_label = torch.tensor(train_set[1])
    test_label = torch.tensor(test_set[1])
    train_data = torch.tensor(train_set[0])
    test_data = torch.tensor(test_set[0])
    data_prior = [0,0,0,0,0,0]
    num_unlabel_known = 0
    num_unknown_known = 0
    num_unknown_unlabel = 0
    num_unlabel_unlabel = 0
    index = []
    for i in range(train_label.size()[0]):
        if train_label[i] == P or train_label[i] == N or train_label[i] == A or train_label[i] == A_d:
            index.append(i)
    index = torch.tensor(index)
    new_train_set = torch.index_select(train_data, dim=0, index=index)
    new_train_label = torch.index_select(train_label, dim=0, index=index)
    new_train_label_index = torch.index_select(train_label, dim=0, index=index)
    rand_index = random.sample(range(new_train_set.size()[0]), num_data)
    for i in range(new_train_set.size()[0]):
        if new_train_label_index[i] == P:
            new_train_label[i] = 0
        if new_train_label_index[i] == N:
            new_train_label[i] = 1
        if new_train_label_index[i] == A or new_train_label_index[i] == A_d:
            new_train_label[i] = 2
        if i in rand_index[:int(num_data / 2)] and (new_train_label[i] == 0 or new_train_label[i] == 1):
            if new_train_label[i] == 0:
                num_unlabel_known = num_unlabel_known + 1 # unlabel 中含有的标记样本个数
            elif new_train_label[i] == 1:
                num_unlabel_unlabel = num_unlabel_unlabel + 1
            new_train_label[i] = 1
        if i in rand_index[int(num_data / 2):]:
            if new_train_label[i] == 0:
                num_unknown_known = num_unknown_known + 1
            elif new_train_label[i] == 1:
                num_unknown_unlabel = num_unknown_unlabel + 1
            new_train_label[i] = 2

    num_known = torch.eq(new_train_label, 0).sum()
    num_unlabel = torch.eq(new_train_label, 1).sum()
    num_unknown = torch.eq(new_train_label, 2).sum()
    num_known_class = torch.eq(train_label, P).sum()  # 真实类别 样本个数
    num_unlabel_class = torch.eq(train_label, N).sum()
    num_unknown_class = torch.eq(train_label, A).sum() + torch.eq(train_label, A_d).sum()

    num_unlabel_known = torch.tensor(num_unlabel_known)
    num_unknown_known = torch.tensor(num_unknown_known)
    num_unknown_unlabel = torch.tensor(num_unknown_unlabel)
    num_unlabel_unlabel = torch.tensor(num_unlabel_unlabel)

    data_prior[0] = torch.tensor(1)                                  # p_1_1
    data_prior[1] = num_unlabel_known.float() / num_unlabel * C_u_p      #p_2_1
    data_prior[2] = (num_unlabel_class - num_unknown_unlabel).float() / num_unlabel      #p_2_2
    data_prior[3] = num_unknown_known.float() / num_unknown * C_a_p   #p_3_1
    data_prior[4] = num_unknown_unlabel.float() / num_unknown * C_a_n  #p_3_2
    data_prior[5] = num_unknown_class.float() / num_unknown

    data_class_prior = [0,0,0]
    data_class_prior[0] = num_known_class.float()/new_train_set.size()[0]
    data_class_prior[1] = num_unlabel_class.float() / new_train_set.size()[0]
    data_class_prior[2] = num_unknown_class.float() / new_train_set.size()[0]
    print(data_prior, data_class_prior,num_known_class, num_unlabel_class, num_unknown_class)
    # print(index)
    # print(new_train_label.size())
    index = []
    for i in range(test_data.size()[0]):
        if test_label[i] == P or test_label[i] == N or test_label[i] == A or test_label[i] == A_d:
            index.append(i)
    index = torch.tensor(index)
    new_test_set = torch.index_select(test_data, dim=0, index=index)
    new_test_label = torch.index_select(test_label, dim=0, index=index)
    new_test_label_index = torch.index_select(test_label, dim=0, index=index)
    for i in range(new_test_set.size()[0]):
        if new_test_label_index[i] == P:
            new_test_label[i] = 0
        if new_test_label_index[i] == N:
            new_test_label[i] = 1
        if new_test_label_index[i] == A or new_test_label_index[i] == A_d:
            new_test_label[i] = 2
    # print(new_test_label.size())
    # print('data set  Parameter： unknown set 3 known set 2 unlabel set 6 number of unknown and unlabel  {} {}'.format(
    #     num_data / 2, num_data / 2))

    test_num_unknown = torch.eq(new_test_label, 2).sum()
    print('A class:{}, N class: {}, A class: {}'.format(P, N, A))
    print('PU number:{}, AC number: {}'.format(num_known+num_unlabel, num_unknown))
    print('Test number:{}, test AC number: {}'.format(new_test_label.size()[0], test_num_unknown))

    return new_train_set.unsqueeze(1), new_train_label, new_test_set.unsqueeze(1), new_test_label, data_prior, data_class_prior


def read_data_minist_single_shift(data_set, i_train_iter):
    num_data = 6000
    random.seed(i_train_iter)
    C_u_p = 1
    C_a_p = 1
    C_a_n = 1
    P = 3
    N = 5
    A = 1
    if data_set == 1:
        root="D:/code/data/minist/minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images.idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels.idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images.idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels.idx1-ubyte'))
        )
    elif data_set ==2:
        root = "D:/code/data/fashion_minist/fashion_minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )
    else:
        root = "D:/code/data/Kuzushiji_minist/Kuzushiji_minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )



    train_label = torch.tensor(train_set[1])
    test_label = torch.tensor(test_set[1])
    train_data = torch.tensor(train_set[0])
    test_data = torch.tensor(test_set[0])
    data_prior = [0,0,0,0,0,0]
    num_unlabel_known = 0
    num_unknown_known = 0
    num_unknown_unlabel = 0
    num_unlabel_unlabel = 0
    index = []
    for i in range(train_label.size()[0]):
        if train_label[i] == P or train_label[i] == N or train_label[i] == A:
            index.append(i)
    index = torch.tensor(index)
    new_train_set = torch.index_select(train_data, dim=0, index=index)
    new_train_label = torch.index_select(train_label, dim=0, index=index)
    new_train_label_index = torch.index_select(train_label, dim=0, index=index)
    rand_index = random.sample(range(new_train_set.size()[0]), num_data)
    for i in range(new_train_set.size()[0]):
        if new_train_label_index[i] == P:
            new_train_label[i] = 0
        if new_train_label_index[i] == N:
            new_train_label[i] = 1
        if new_train_label_index[i] == A:
            new_train_label[i] = 2
        if i in rand_index[:int(num_data / 2)] and (new_train_label[i] == 0 or new_train_label[i] == 1):
            if new_train_label[i] == 0:
                num_unlabel_known = num_unlabel_known + 1 # unlabel 中含有的标记样本个数
            elif new_train_label[i] == 1:
                num_unlabel_unlabel = num_unlabel_unlabel + 1
            new_train_label[i] = 1
        if i in rand_index[int(num_data / 2):]:
            if new_train_label[i] == 0:
                num_unknown_known = num_unknown_known + 1
            elif new_train_label[i] == 1:
                num_unknown_unlabel = num_unknown_unlabel + 1
            new_train_label[i] = 2

    num_known = torch.eq(new_train_label, 0).sum()
    num_unlabel = torch.eq(new_train_label, 1).sum()
    num_unknown = torch.eq(new_train_label, 2).sum()
    num_known_class = torch.eq(train_label, P).sum()  # 真实类别 样本个数
    num_unlabel_class = torch.eq(train_label, N).sum()
    num_unknown_class = torch.eq(train_label, A).sum()

    num_unlabel_known = torch.tensor(num_unlabel_known)
    num_unknown_known = torch.tensor(num_unknown_known)
    num_unknown_unlabel = torch.tensor(num_unknown_unlabel)
    num_unlabel_unlabel = torch.tensor(num_unlabel_unlabel)

    data_prior[0] = torch.tensor(1)                                  # p_1_1
    data_prior[1] = num_unlabel_known.float() / num_unlabel * C_u_p      #p_2_1
    data_prior[2] = (num_unlabel_class - num_unknown_unlabel).float() / num_unlabel      #p_2_2
    data_prior[3] = num_unknown_known.float() / num_unknown * C_a_p   #p_3_1
    data_prior[4] = num_unknown_unlabel.float() / num_unknown * C_a_n  #p_3_2
    data_prior[5] = num_unknown_class.float() / num_unknown

    data_class_prior = [0,0,0]
    data_class_prior[0] = num_known_class.float()/new_train_set.size()[0]
    data_class_prior[1] = num_unlabel_class.float() / new_train_set.size()[0]
    data_class_prior[2] = num_unknown_class.float() / new_train_set.size()[0]
    print(data_prior, data_class_prior,num_known_class, num_unlabel_class, num_unknown_class)
    # print(index)
    # print(new_train_label.size())
    N_p = 1
    N_n = 1
    N_a = 1
    num_p = torch.eq(test_label, P).sum()/2
    num_n = torch.eq(test_label, N).sum()/2
    num_a = torch.eq(test_label, A).sum()/2
    f_p = 0
    f_n = 0
    f_a = 0
    index = []
    for i in range(test_data.size()[0]):
        if test_label[i] == P and (f_p <= num_p*N_p):
            f_p = f_p + 1
            index.append(i)
        if test_label[i] == N and (f_n <= num_n*N_n):
            f_p = f_p + 1
            index.append(i)
        if test_label[i] == A and (f_a <= num_a*N_a):
            f_p = f_p + 1
            index.append(i)



    index = torch.tensor(index)
    new_test_set = torch.index_select(test_data, dim=0, index=index)
    new_test_label = torch.index_select(test_label, dim=0, index=index)
    new_test_label_index = torch.index_select(test_label, dim=0, index=index)
    for i in range(new_test_set.size()[0]):
        if new_test_label_index[i] == P:
            new_test_label[i] = 0
        if new_test_label_index[i] == N:
            new_test_label[i] = 1
        if new_test_label_index[i] == A:
            new_test_label[i] = 2
    # print(new_test_label.size())
    # print('data set  Parameter： unknown set 3 known set 2 unlabel set 6 number of unknown and unlabel  {} {}'.format(
    #     num_data / 2, num_data / 2))

    test_num_unknown = torch.eq(new_test_label, 2).sum()
    print('A class:{}, N class: {}, A class: {}'.format(P, N, A))
    print('PU number:{}, AC number: {}'.format(num_known+num_unlabel, num_unknown))
    print('Test number:{}, test AC number: {}'.format(new_test_label.size()[0], test_num_unknown))

    return new_train_set.unsqueeze(1), new_train_label, new_test_set.unsqueeze(1), new_test_label, data_prior, data_class_prior


def read_data_minist_CL_rato(data_set, i_train_iter, rato):
    num_data = 20000
    random.seed(i_train_iter)
    P1 = 4
    P2 = 5
    P3 = 6
    A = 0
    if data_set == 1:
        root="D:/code/data/minist/minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images.idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels.idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images.idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels.idx1-ubyte'))
        )
    elif data_set ==2:
        root = "D:/code/data/fashion_minist/fashion_minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )
    else:
        root = "D:/code/data/Kuzushiji_minist/Kuzushiji_minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )



    train_label = torch.tensor(train_set[1])
    test_label = torch.tensor(test_set[1])
    train_data = torch.tensor(train_set[0])
    test_data = torch.tensor(test_set[0])
    data_prior = [0,0,0]
    num_p1_unlabel = 0
    num_p2_unlabel = 0
    num_p3_unlabel = 0
    index = []
    for i in range(train_label.size()[0]):
        if train_label[i] == P1 or train_label[i] == P2 or train_label[i] == P3 or train_label[i] == A:
            index.append(i)
    index = torch.tensor(index)
    new_train_set = torch.index_select(train_data, dim=0, index=index)
    new_train_label = torch.index_select(train_label, dim=0, index=index)
    new_train_label_index = torch.index_select(train_label, dim=0, index=index)
    new_train_CL_label_index = torch.index_select(train_label, dim=0, index=index)
    rand_index = random.sample(range(new_train_set.size()[0]), num_data)
    for i in range(new_train_set.size()[0]):
        if new_train_label_index[i] == P1:
            new_train_label[i] = 0
        if new_train_label_index[i] == P2:
            new_train_label[i] = 1
        if new_train_label_index[i] == P3:
            new_train_label[i] = 2
        if new_train_label_index[i] == A:
            new_train_label[i] = 3
        if i in rand_index[:int(num_data / 2)] and (new_train_label[i] == 0 or new_train_label[i] == 1 or new_train_label[i] == 2):
            if new_train_label[i] == 0:
                num_p1_unlabel = num_p1_unlabel + 1 # unlabel 中含有的标记样本个数
            elif new_train_label[i] == 1:
                num_p2_unlabel = num_p2_unlabel + 1
            elif new_train_label[i] == 2:
                num_p3_unlabel = num_p3_unlabel + 1
            new_train_label[i] = 3

    for i in range(new_train_set.size()[0]):
        if new_train_label[i] == 0:
            r = random.randint(1, 2)
            new_train_CL_label_index[i] = r
        if new_train_label[i] == 1:
            r = random.randint(1, 2)
            if r == 1:
                new_train_CL_label_index[i] = 0
            elif r == 2:
                new_train_CL_label_index[i] = 2
        if new_train_label[i] == 2:
            r = random.randint(1, 2)
            if r == 1:
                new_train_CL_label_index[i] = 0
            elif r == 2:
                new_train_CL_label_index[i] = 1
        if new_train_label[i] == 3:
            new_train_CL_label_index[i] = 3



    num_ALL = new_train_set.size()[0]
    num_P1 = torch.eq(new_train_label_index, P1).sum()
    num_P2 = torch.eq(new_train_label_index, P2).sum()
    num_P3 = torch.eq(new_train_label_index, P3).sum()
    num_A = torch.eq(new_train_CL_label_index, 3).sum()
    num_p1_unlabel = torch.tensor(num_p1_unlabel)
    num_p2_unlabel = torch.tensor(num_p2_unlabel)
    num_p3_unlabel = torch.tensor(num_p3_unlabel)

    # data_prior[0] = num_P1.float() / num_ALL                                 # p_1_1
    # data_prior[1] = num_P2.float() / num_ALL      #p_2_1
    # data_prior[2] = num_P3.float() / num_ALL     #p_2_2
    #
    data_prior[0] = num_p1_unlabel.float() / num_A                                 # p_1_1
    data_prior[1] = num_p2_unlabel.float() / num_A      #p_2_1
    data_prior[2] = num_p3_unlabel.float() / num_A     #p_2_2

    data_sum_prior = data_prior[0] + data_prior[1] + data_prior[2]
    number = 8000
    train_num = number * rato * data_sum_prior/3
    train_num_a = number - number *  data_sum_prior
    print('data_prior',data_prior)
    # print(index)
    # print(new_train_label.size())
    index = []
    i_1 = 0
    i_2 = 0
    i_3 = 0
    i_4 = 0
    for i in range(test_data.size()[0]):
        if test_label[i] == P1 or test_label[i] == P2 or test_label[i] == P3 or test_label[i] == A:
            if i_1 < train_num and test_label[i] == P1:
                index.append(i)
                i_1 = i_1+1
            elif i_2 < train_num and test_label[i] == P2:
                index.append(i)
                i_2 = i_2 + 1
            elif i_3 < train_num and test_label[i] == P3:
                index.append(i)
                i_3 = i_3 + 1
            elif i_4 < train_num_a and test_label[i] == A:
                index.append(i)
                i_4 = i_4 + 1

    index = torch.tensor(index)
    new_test_set = torch.index_select(test_data, dim=0, index=index)
    new_test_label = torch.index_select(test_label, dim=0, index=index)
    new_test_label_index = torch.index_select(test_label, dim=0, index=index)
    for i in range(new_test_set.size()[0]):
        if new_test_label_index[i] == P1:
            new_test_label[i] = 0
        if new_test_label_index[i] == P2:
            new_test_label[i] = 1
        if new_test_label_index[i] == P3:
            new_test_label[i] = 2
        if new_test_label_index[i] == A:
            new_test_label[i] = 3

    return new_train_set.unsqueeze(1), new_train_CL_label_index, new_test_set.unsqueeze(1), new_test_label, data_prior


def read_data_minist_CL(data_set, i_train_iter):
    num_data = 20000
    random.seed(i_train_iter)
    P1 = 7
    P2 = 8
    P3 = 9
    A = 1
    if data_set == 1:
        root="D:/code/data/minist/minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images.idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels.idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images.idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels.idx1-ubyte'))
        )
    elif data_set ==2:
        root = "D:/code/data/fashion_minist/fashion_minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )
    else:
        root = "D:/code/data/Kuzushiji_minist/Kuzushiji_minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )



    train_label = torch.tensor(train_set[1])
    test_label = torch.tensor(test_set[1])
    train_data = torch.tensor(train_set[0])
    test_data = torch.tensor(test_set[0])
    data_prior = [0,0,0]
    num_p1_unlabel = 0
    num_p2_unlabel = 0
    num_p3_unlabel = 0
    index = []
    for i in range(train_label.size()[0]):
        if train_label[i] == P1 or train_label[i] == P2 or train_label[i] == P3 or train_label[i] == A:
            index.append(i)
    index = torch.tensor(index)
    new_train_set = torch.index_select(train_data, dim=0, index=index)
    new_train_label = torch.index_select(train_label, dim=0, index=index)
    new_train_label_index = torch.index_select(train_label, dim=0, index=index)
    new_train_CL_label_index = torch.index_select(train_label, dim=0, index=index)
    rand_index = random.sample(range(new_train_set.size()[0]), num_data)
    for i in range(new_train_set.size()[0]):
        if new_train_label_index[i] == P1:
            new_train_label[i] = 0
        if new_train_label_index[i] == P2:
            new_train_label[i] = 1
        if new_train_label_index[i] == P3:
            new_train_label[i] = 2
        if new_train_label_index[i] == A:
            new_train_label[i] = 3
        if i in rand_index[:int(num_data / 2)] and (new_train_label[i] == 0 or new_train_label[i] == 1 or new_train_label[i] == 2):
            if new_train_label[i] == 0:
                num_p1_unlabel = num_p1_unlabel + 1 # unlabel 中含有的标记样本个数
            elif new_train_label[i] == 1:
                num_p2_unlabel = num_p2_unlabel + 1
            elif new_train_label[i] == 2:
                num_p3_unlabel = num_p3_unlabel + 1
            new_train_label[i] = 3

    for i in range(new_train_set.size()[0]):
        if new_train_label[i] == 0:
            r = random.randint(1, 2)
            new_train_CL_label_index[i] = r
        if new_train_label[i] == 1:
            r = random.randint(1, 2)
            if r == 1:
                new_train_CL_label_index[i] = 0
            elif r == 2:
                new_train_CL_label_index[i] = 2
        if new_train_label[i] == 2:
            r = random.randint(1, 2)
            if r == 1:
                new_train_CL_label_index[i] = 0
            elif r == 2:
                new_train_CL_label_index[i] = 1
        if new_train_label[i] == 3:
            new_train_CL_label_index[i] = 3



    num_ALL = new_train_set.size()[0]
    num_P1 = torch.eq(new_train_label_index, P1).sum()
    num_P2 = torch.eq(new_train_label_index, P2).sum()
    num_P3 = torch.eq(new_train_label_index, P3).sum()
    num_A = torch.eq(new_train_CL_label_index, 3).sum()
    num_p1_unlabel = torch.tensor(num_p1_unlabel)
    num_p2_unlabel = torch.tensor(num_p2_unlabel)
    num_p3_unlabel = torch.tensor(num_p3_unlabel)

    # data_prior[0] = num_P1.float() / num_ALL                                 # p_1_1
    # data_prior[1] = num_P2.float() / num_ALL      #p_2_1
    # data_prior[2] = num_P3.float() / num_ALL     #p_2_2
    #
    data_prior[0] = num_p1_unlabel.float() / num_A                                 # p_1_1
    data_prior[1] = num_p2_unlabel.float() / num_A      #p_2_1
    data_prior[2] = num_p3_unlabel.float() / num_A     #p_2_2

    print('data_prior',data_prior)
    # print(index)
    # print(new_train_label.size())
    index = []
    for i in range(test_data.size()[0]):
        if test_label[i] == P1 or test_label[i] == P2 or test_label[i] == P3 or test_label[i] == A:
            index.append(i)
    index = torch.tensor(index)
    new_test_set = torch.index_select(test_data, dim=0, index=index)
    new_test_label = torch.index_select(test_label, dim=0, index=index)
    new_test_label_index = torch.index_select(test_label, dim=0, index=index)
    for i in range(new_test_set.size()[0]):
        if new_test_label_index[i] == P1:
            new_test_label[i] = 0
        if new_test_label_index[i] == P2:
            new_test_label[i] = 1
        if new_test_label_index[i] == P3:
            new_test_label[i] = 2
        if new_test_label_index[i] == A:
            new_test_label[i] = 3

    return new_train_set.unsqueeze(1), new_train_CL_label_index, new_test_set.unsqueeze(1), new_test_label, data_prior


def read_data_minist_CL_m(data_set, i_train_iter):
    num_data = 20000
    random.seed(i_train_iter)
    P1 = 7
    P2 = 8
    P3 = 9
    A_1 = 1
    A_2 = 3
    if data_set == 1:
        root="D:/code/data/minist/minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images.idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels.idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images.idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels.idx1-ubyte'))
        )
    elif data_set ==2:
        root = "D:/code/data/fashion_minist/fashion_minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )
    else:
        root = "D:/code/data/Kuzushiji_minist/Kuzushiji_minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )



    train_label = torch.tensor(train_set[1])
    test_label = torch.tensor(test_set[1])
    train_data = torch.tensor(train_set[0])
    test_data = torch.tensor(test_set[0])
    data_prior = [0,0,0]
    num_p1_unlabel = 0
    num_p2_unlabel = 0
    num_p3_unlabel = 0
    index = []
    for i in range(train_label.size()[0]):
        if train_label[i] == P1 or train_label[i] == P2 or train_label[i] == P3 or train_label[i] == A_1 or train_label[i] == A_2:
            index.append(i)
    index = torch.tensor(index)
    new_train_set = torch.index_select(train_data, dim=0, index=index)
    new_train_label = torch.index_select(train_label, dim=0, index=index)
    new_train_label_index = torch.index_select(train_label, dim=0, index=index)
    new_train_CL_label_index = torch.index_select(train_label, dim=0, index=index)
    rand_index = random.sample(range(new_train_set.size()[0]), num_data)
    for i in range(new_train_set.size()[0]):
        if new_train_label_index[i] == P1:
            new_train_label[i] = 0
        if new_train_label_index[i] == P2:
            new_train_label[i] = 1
        if new_train_label_index[i] == P3:
            new_train_label[i] = 2
        if new_train_label_index[i] == A_1:
            new_train_label[i] = 3
        if new_train_label_index[i] == A_2:
            new_train_label[i] = 3
        if i in rand_index[:int(num_data / 2)] and (new_train_label[i] == 0 or new_train_label[i] == 1 or new_train_label[i] == 2):
            if new_train_label[i] == 0:
                num_p1_unlabel = num_p1_unlabel + 1 # unlabel 中含有的标记样本个数
            elif new_train_label[i] == 1:
                num_p2_unlabel = num_p2_unlabel + 1
            elif new_train_label[i] == 2:
                num_p3_unlabel = num_p3_unlabel + 1
            new_train_label[i] = 3

    for i in range(new_train_set.size()[0]):
        if new_train_label[i] == 0:
            r = random.randint(1, 2)
            new_train_CL_label_index[i] = r
        if new_train_label[i] == 1:
            r = random.randint(1, 2)
            if r == 1:
                new_train_CL_label_index[i] = 0
            elif r == 2:
                new_train_CL_label_index[i] = 2
        if new_train_label[i] == 2:
            r = random.randint(1, 2)
            if r == 1:
                new_train_CL_label_index[i] = 0
            elif r == 2:
                new_train_CL_label_index[i] = 1
        if new_train_label[i] == 3:
            new_train_CL_label_index[i] = 3



    num_ALL = new_train_set.size()[0]
    num_P1 = torch.eq(new_train_label_index, P1).sum()
    num_P2 = torch.eq(new_train_label_index, P2).sum()
    num_P3 = torch.eq(new_train_label_index, P3).sum()
    num_A = torch.eq(new_train_CL_label_index, 3).sum()
    num_p1_unlabel = torch.tensor(num_p1_unlabel)
    num_p2_unlabel = torch.tensor(num_p2_unlabel)
    num_p3_unlabel = torch.tensor(num_p3_unlabel)

    # data_prior[0] = num_P1.float() / num_ALL                                 # p_1_1
    # data_prior[1] = num_P2.float() / num_ALL      #p_2_1
    # data_prior[2] = num_P3.float() / num_ALL     #p_2_2
    #
    data_prior[0] = num_p1_unlabel.float() / num_A                                 # p_1_1
    data_prior[1] = num_p2_unlabel.float() / num_A      #p_2_1
    data_prior[2] = num_p3_unlabel.float() / num_A     #p_2_2

    print('data_prior',data_prior)
    # print(index)
    # print(new_train_label.size())
    index = []
    for i in range(test_data.size()[0]):
        if test_label[i] == P1 or test_label[i] == P2 or test_label[i] == P3 or test_label[i] == A_1 or test_label[i] == A_2:
            index.append(i)
    index = torch.tensor(index)
    new_test_set = torch.index_select(test_data, dim=0, index=index)
    new_test_label = torch.index_select(test_label, dim=0, index=index)
    new_test_label_index = torch.index_select(test_label, dim=0, index=index)
    for i in range(new_test_set.size()[0]):
        if new_test_label_index[i] == P1:
            new_test_label[i] = 0
        if new_test_label_index[i] == P2:
            new_test_label[i] = 1
        if new_test_label_index[i] == P3:
            new_test_label[i] = 2
        if new_test_label_index[i] == A_1:
            new_test_label[i] = 3
        if new_test_label_index[i] == A_2:
            new_test_label[i] = 3
    print(new_test_label.size())

    return new_train_set.unsqueeze(1), new_train_CL_label_index, new_test_set.unsqueeze(1), new_test_label, data_prior

def read_data_minist_CL_mc(data_set, i_train_iter):
    num_data = 20000
    random.seed(i_train_iter)
    P1 = 1
    P2 = 2
    P3 = 3
    P4 = 4
    A = 0
    if data_set == 1:
        root="D:/code/data/minist/minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images.idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels.idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images.idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels.idx1-ubyte'))
        )
    elif data_set ==2:
        root = "D:/code/data/fashion_minist/fashion_minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )
    else:
        root = "D:/code/data/Kuzushiji_minist/Kuzushiji_minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )



    train_label = torch.tensor(train_set[1])
    test_label = torch.tensor(test_set[1])
    train_data = torch.tensor(train_set[0])
    test_data = torch.tensor(test_set[0])
    data_prior = [0,0,0,0]
    num_p1_unlabel = 0
    num_p2_unlabel = 0
    num_p3_unlabel = 0
    num_p4_unlabel = 0
    index = []
    for i in range(train_label.size()[0]):
        if train_label[i] == P1 or train_label[i] == P2 or train_label[i] == P3 or train_label[i] == P4 or train_label[i] == A:
            index.append(i)
    index = torch.tensor(index)
    new_train_set = torch.index_select(train_data, dim=0, index=index)
    new_train_label = torch.index_select(train_label, dim=0, index=index)
    new_train_label_index = torch.index_select(train_label, dim=0, index=index)
    new_train_CL_label_index = torch.index_select(train_label, dim=0, index=index)
    rand_index = random.sample(range(new_train_set.size()[0]), num_data)
    for i in range(new_train_set.size()[0]):
        if new_train_label_index[i] == P1:
            new_train_label[i] = 0
        if new_train_label_index[i] == P2:
            new_train_label[i] = 1
        if new_train_label_index[i] == P3:
            new_train_label[i] = 2
        if new_train_label_index[i] == P4:
            new_train_label[i] = 3
        if new_train_label_index[i] == A:
            new_train_label[i] = 4
        if i in rand_index[:int(num_data / 2)] and (new_train_label[i] == 0 or new_train_label[i] == 1 or new_train_label[i] == 2 or new_train_label[i] == 3):
            if new_train_label[i] == 0:
                num_p1_unlabel = num_p1_unlabel + 1 # unlabel 中含有的标记样本个数
            elif new_train_label[i] == 1:
                num_p2_unlabel = num_p2_unlabel + 1
            elif new_train_label[i] == 2:
                num_p3_unlabel = num_p3_unlabel + 1
            elif new_train_label[i] == 3:
                num_p4_unlabel = num_p4_unlabel + 1
            new_train_label[i] = 4

    for i in range(new_train_set.size()[0]):
        if new_train_label[i] == 0:
            r = random.randint(1, 3)
            new_train_CL_label_index[i] = r
        if new_train_label[i] == 1:
            r = random.randint(1, 3)
            if r == 1:
                new_train_CL_label_index[i] = 0
            elif r == 2:
                new_train_CL_label_index[i] = 2
            elif r == 3:
                new_train_CL_label_index[i] = 3
        if new_train_label[i] == 2:
            r = random.randint(1, 3)
            if r == 1:
                new_train_CL_label_index[i] = 0
            elif r == 2:
                new_train_CL_label_index[i] = 1
            elif r == 3:
                new_train_CL_label_index[i] = 3
        if new_train_label[i] == 3:
            r = random.randint(1, 3)
            if r == 1:
                new_train_CL_label_index[i] = 0
            elif r == 2:
                new_train_CL_label_index[i] = 1
            elif r == 3:
                new_train_CL_label_index[i] = 2
        if new_train_label[i] == 4:
            new_train_CL_label_index[i] = 4



    num_ALL = new_train_set.size()[0]
    num_P1 = torch.eq(new_train_label_index, P1).sum()
    num_P2 = torch.eq(new_train_label_index, P2).sum()
    num_P3 = torch.eq(new_train_label_index, P3).sum()
    num_A = torch.eq(new_train_CL_label_index, 4).sum()
    num_p1_unlabel = torch.tensor(num_p1_unlabel)
    num_p2_unlabel = torch.tensor(num_p2_unlabel)
    num_p3_unlabel = torch.tensor(num_p3_unlabel)
    num_p4_unlabel = torch.tensor(num_p4_unlabel)

    # data_prior[0] = num_P1.float() / num_ALL                                 # p_1_1
    # data_prior[1] = num_P2.float() / num_ALL      #p_2_1
    # data_prior[2] = num_P3.float() / num_ALL     #p_2_2
    #
    data_prior[0] = num_p1_unlabel.float() / num_A                                 # p_1_1
    data_prior[1] = num_p2_unlabel.float() / num_A      #p_2_1
    data_prior[2] = num_p3_unlabel.float() / num_A     #p_2_2
    data_prior[3] = num_p4_unlabel.float() / num_A
    print('data_prior', data_prior)
    # print(index)
    # print(new_train_label.size())
    index = []
    for i in range(test_data.size()[0]):
        if test_label[i] == P1 or test_label[i] == P2 or test_label[i] == P3 or test_label[i] == A:
            index.append(i)
    index = torch.tensor(index)
    new_test_set = torch.index_select(test_data, dim=0, index=index)
    new_test_label = torch.index_select(test_label, dim=0, index=index)
    new_test_label_index = torch.index_select(test_label, dim=0, index=index)
    for i in range(new_test_set.size()[0]):
        if new_test_label_index[i] == P1:
            new_test_label[i] = 0
        if new_test_label_index[i] == P2:
            new_test_label[i] = 1
        if new_test_label_index[i] == P3:
            new_test_label[i] = 2
        if new_test_label_index[i] == P4:
            new_test_label[i] = 3
        if new_test_label_index[i] == A:
            new_test_label[i] = 4

    return new_train_set.unsqueeze(1), new_train_CL_label_index, new_test_set.unsqueeze(1), new_test_label, data_prior




def read_data_minist_TF(data_set, i_train_iter):
    random.seed(i_train_iter)
    P0 = 0
    P1 = 1
    P2 = 2
    P3 = 3
    P4 = 4
    P5 = 5
    P6 = 6
    P7 = 7
    P8 = 8
    P9 = 9
    if data_set == 1:
        root="D:/code/data/minist/minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images.idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels.idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images.idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels.idx1-ubyte'))
        )
    elif data_set ==2:
        root = "D:/code/data/fashion_minist/fashion_minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )
    else:
        root = "D:/code/data/Kuzushiji_minist/Kuzushiji_minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )



    train_label = torch.tensor(train_set[1])
    test_label = torch.tensor(test_set[1])
    train_data = torch.tensor(train_set[0])
    test_data = torch.tensor(test_set[0])
    num_data_t = 0
    index = []
    for i in range(train_label.size()[0]):
        if train_label[i] == P0 or train_label[i] == P1 or train_label[i] == P2 or train_label[i] == P3 or train_label[i] == P4 or train_label[i] == P5 or train_label[i] == P6 or train_label[i] == P7 or train_label[i] == P8 or train_label[i] == P9:
            index.append(i)
            num_data_t = num_data_t + 1

    index = torch.tensor(index)
    new_train_set = torch.index_select(train_data, dim=0, index=index)
    new_train_label = torch.index_select(train_label, dim=0, index=index)
    new_train_label_index = torch.index_select(train_label, dim=0, index=index)
    new_train_TF_label_index = torch.index_select(train_label, dim=0, index=index)
    new_train_SL_label_index_0 = torch.index_select(train_label, dim=0, index=index)
    new_train_SL_label_index_1 = torch.index_select(train_label, dim=0, index=index)

    for i in range(new_train_set.size()[0]):
        if new_train_label_index[i] == P0:
            new_train_label[i] = 0
        if new_train_label_index[i] == P1:
            new_train_label[i] = 1
        if new_train_label_index[i] == P2:
            new_train_label[i] = 2
        if new_train_label_index[i] == P3:
            new_train_label[i] = 3
        if new_train_label_index[i] == P4:
            new_train_label[i] = 4
        if new_train_label_index[i] == P5:
            new_train_label[i] = 5
        if new_train_label_index[i] == P6:
            new_train_label[i] = 6
        if new_train_label_index[i] == P7:
            new_train_label[i] = 7
        if new_train_label_index[i] == P8:
            new_train_label[i] = 8
        if new_train_label_index[i] == P9:
            new_train_label[i] = 9


    for i in range(new_train_set.size()[0]):
        r_m = random.sample(range(9), 2)
        # print(r_m)
        # r = random.randint(0, 3)
        # print('r = ', r)
        if new_train_label[i] == r_m[0]:
            new_train_TF_label_index[i] = r_m[0]
            new_train_SL_label_index_0[i] = r_m[0]
            new_train_SL_label_index_1[i] = r_m[1]
        elif new_train_label[i] == r_m[1]:
            new_train_TF_label_index[i] = r_m[1]
            new_train_SL_label_index_0[i] = r_m[0]
            new_train_SL_label_index_1[i] = r_m[1]
        else:
            new_train_TF_label_index[i] = 2
            new_train_SL_label_index_0[i] = r_m[0]
            new_train_SL_label_index_1[i] = r_m[1]
        # print('new_train_label = ', new_train_label[i])
        # print('new_train_SL_label_index = ', new_train_SL_label_index[i])
        # print('new_train_TF_label_index = ', new_train_TF_label_index[i])


    num_ALL = new_train_set.size()[0]
    num_t_label = torch.eq(new_train_TF_label_index, 2).sum()
    data_prior = num_t_label.float()/num_ALL

    print('data_prior', data_prior)
    # print(index)
    # print(new_train_label.size())
    index = []
    for i in range(test_data.size()[0]):
        if test_label[i] == P0 or test_label[i] == P1 or test_label[i] == P2 or test_label[i] == P3 or test_label[i] == P4 or test_label[i] == P5 or test_label[i] == P6 or test_label[i] == P7 or test_label[i] == P8 or test_label[i] == P9:
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
        if new_test_label_index[i] == P3:
            new_test_label[i] = 3
        if new_test_label_index[i] == P4:
            new_test_label[i] = 4
        if new_test_label_index[i] == P5:
            new_test_label[i] = 5
        if new_test_label_index[i] == P6:
            new_test_label[i] = 6
        if new_test_label_index[i] == P7:
            new_test_label[i] = 7
        if new_test_label_index[i] == P8:
            new_test_label[i] = 8
        if new_test_label_index[i] == P9:
            new_test_label[i] = 9

    return new_train_set.unsqueeze(1), new_train_TF_label_index, new_train_SL_label_index_0, new_train_SL_label_index_1, new_test_set.unsqueeze(1), new_test_label, data_prior



def MSL_GEN(new_train_label, new_train_set, new_train_TF_label_index,  n):
    new_train_SL_label_index = [[0] * n] * new_train_set.size()[0]
    for i in range(new_train_set.size()[0]):
        r_m = random.sample(range(9), n)
        flag = 0
        for k in range(n):
            if new_train_label[i] == r_m[k]:
                new_train_TF_label_index[i] = r_m[k]
                flag = 1
                for j in range(n):
                    new_train_SL_label_index[i][j] = r_m[j]
        if flag == 0:
            new_train_TF_label_index[i] = 9
            for j in range(n):
                new_train_SL_label_index[i][j] = r_m[j]
    new_train_SL_label_index = torch.tensor(new_train_SL_label_index)
    return new_train_TF_label_index, new_train_SL_label_index

def read_data_minist_TF_MSL(data_set, i_train_iter, n, conceal_label):
    random.seed(i_train_iter)
    P = []
    for i in [0,1,2,3,4,5,6,7,8,9]:
        if i!= conceal_label:
            P.append(i)
    P.append(conceal_label)
    if data_set == 1:
        root="D:/code/data/minist/minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images.idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels.idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images.idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels.idx1-ubyte'))
        )
    elif data_set ==2:
        root = "D:/code/data/fashion_minist/fashion_minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )
    else:
        root = "D:/code/data/Kuzushiji_minist/Kuzushiji_minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )



    train_label = torch.tensor(train_set[1])
    test_label = torch.tensor(test_set[1])
    train_data = torch.tensor(train_set[0])
    test_data = torch.tensor(test_set[0])
    num_data_t = 0
    index = []
    for i in range(train_label.size()[0]):
        if train_label[i] == P[1] or train_label[i] == P[2] or train_label[i] == P[3] or train_label[i] == P[4] or train_label[i] == P[5] or train_label[i] == P[6] or train_label[i] == P[7] or train_label[i] == P[8] or train_label[i] == P[9] or train_label[i] == P[0]:
            index.append(i)
            num_data_t = num_data_t + 1

    index = torch.tensor(index)
    new_train_set = torch.index_select(train_data, dim=0, index=index)
    new_train_label = torch.index_select(train_label, dim=0, index=index)
    new_train_label_index = torch.index_select(train_label, dim=0, index=index)
    new_train_TF_label_index = torch.index_select(train_label, dim=0, index=index)


    for i in range(new_train_set.size()[0]):
        if new_train_label_index[i] == P[0]:
            new_train_label[i] = 0
        if new_train_label_index[i] == P[1]:
            new_train_label[i] = 1
        if new_train_label_index[i] == P[2]:
            new_train_label[i] = 2
        if new_train_label_index[i] == P[3]:
            new_train_label[i] = 3
        if new_train_label_index[i] == P[4]:
            new_train_label[i] = 4
        if new_train_label_index[i] == P[5]:
            new_train_label[i] = 5
        if new_train_label_index[i] == P[6]:
            new_train_label[i] = 6
        if new_train_label_index[i] == P[7]:
            new_train_label[i] = 7
        if new_train_label_index[i] == P[8]:
            new_train_label[i] = 8
        if new_train_label_index[i] == P[9]:
            new_train_label[i] = 9

        # print('new_train_label = ', new_train_label[i])
        # print('new_train_SL_label_index = ', new_train_SL_label_index[i])
        # print('new_train_TF_label_index = ', new_train_TF_label_index[i])
    print("data N = ", n)
    new_train_TF_label_index, new_train_SL_label_index = MSL_GEN(new_train_label, new_train_set, new_train_TF_label_index, n)
    num_ALL = new_train_set.size()[0]
    num_t_label = torch.eq(new_train_TF_label_index, 9).sum()
    data_prior = num_t_label.float()/num_ALL
    num_H0_label = torch.eq(new_train_TF_label_index, 0).sum()
    num_H1_label = torch.eq(new_train_TF_label_index, 1).sum()
    num_H2_label = torch.eq(new_train_TF_label_index, 2).sum()
    num_H3_label = torch.eq(new_train_TF_label_index, 3).sum()
    num_H4_label = torch.eq(new_train_TF_label_index, 4).sum()
    num_H5_label = torch.eq(new_train_TF_label_index, 5).sum()
    num_H6_label = torch.eq(new_train_TF_label_index, 6).sum()
    num_H7_label = torch.eq(new_train_TF_label_index, 7).sum()
    num_H8_label = torch.eq(new_train_TF_label_index, 8).sum()

    num_0_label = torch.eq(new_train_label, 0).sum()
    num_1_label = torch.eq(new_train_label, 1).sum()
    num_2_label = torch.eq(new_train_label, 2).sum()
    num_3_label = torch.eq(new_train_label, 3).sum()
    num_4_label = torch.eq(new_train_label, 4).sum()
    num_5_label = torch.eq(new_train_label, 5).sum()
    num_6_label = torch.eq(new_train_label, 6).sum()
    num_7_label = torch.eq(new_train_label, 7).sum()
    num_8_label = torch.eq(new_train_label, 8).sum()

    data_prior_hide = (num_0_label + num_1_label + num_2_label + num_3_label+num_2_label+num_4_label+num_5_label+num_6_label+num_7_label+num_8_label- num_H0_label - num_H1_label - num_H2_label\
                       - num_H3_label- num_H4_label- num_H5_label- num_H6_label- num_H7_label- num_H8_label) / num_t_label.float()

    data_prior_i = torch.eq(new_train_label_index, P[1]).sum()/num_ALL/(torch.eq(new_train_TF_label_index, P[1]).sum()/num_ALL)

    print('data_prior', data_prior)
    print('data_prior_i', data_prior_i)
    # print(index)
    # print(new_train_label.size())
    index = []
    for i in range(test_data.size()[0]):
        if test_label[i] == P[0] or test_label[i] == P[1] or test_label[i] == P[2] or test_label[i] == P[3] or test_label[i] == P[4] or test_label[i] == P[5] or test_label[i] == P[6] or test_label[i] == P[7] or test_label[i] == P[8] or test_label[i] == P[9]:
            index.append(i)
    index = torch.tensor(index)
    new_test_set = torch.index_select(test_data, dim=0, index=index)
    new_test_label = torch.index_select(test_label, dim=0, index=index)
    new_test_label_index = torch.index_select(test_label, dim=0, index=index)
    for i in range(new_test_set.size()[0]):
        if new_test_label_index[i] == P[0]:
            new_test_label[i] = 0
        if new_test_label_index[i] == P[1]:
            new_test_label[i] = 1
        if new_test_label_index[i] == P[2]:
            new_test_label[i] = 2
        if new_test_label_index[i] == P[3]:
            new_test_label[i] = 3
        if new_test_label_index[i] == P[4]:
            new_test_label[i] = 4
        if new_test_label_index[i] == P[5]:
            new_test_label[i] = 5
        if new_test_label_index[i] == P[6]:
            new_test_label[i] = 6
        if new_test_label_index[i] == P[7]:
            new_test_label[i] = 7
        if new_test_label_index[i] == P[8]:
            new_test_label[i] = 8
        if new_test_label_index[i] == P[9]:
            new_test_label[i] = 9

    return new_train_set.unsqueeze(1), new_train_TF_label_index, new_train_SL_label_index, new_test_set.unsqueeze(1), new_test_label, data_prior, data_prior_hide