import numpy as np
import random
import pickle
import platform
import os
import torch

#加载序列文件
def load_pickle(f):
    version=platform.python_version_tuple()#判断python的版本
    if version[0]== '2':
        return pickle.load(f)
    elif version[0]== '3':
        return pickle.load(f,encoding='latin1')
    raise ValueError("invalid python version:{}".format(version))
#处理原数据
def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict=load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
        #reshape()是在不改变矩阵的数值的前提下修改矩阵的形状,transpose()对矩阵进行转置
        Y = np.array(Y)
        return X, Y

#返回可以直接使用的数据集
def load_CIFAR10(ROOT):
    xs=[]
    ys=[]
    for b in range(1,6):
        f=os.path.join(ROOT,'data_batch_%d'%(b,))#os.path.join()将多个路径组合后返回
        X,Y=load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)#这个函数用于将多个数组进行连接
    Ytr = np.concatenate(ys)
    del X,Y
    Xte,Yte=load_CIFAR_batch(os.path.join(ROOT,'test_batch'))
    return Xtr, Ytr, Xte, Yte

def read_data_cifar10():
    num_data = 10000
    random.seed(1)
    P1 = 6
    P2 = 8
    P3 = 9
    A = 1
    datasets = 'D:/code/data/cifar/cifar-10-python/cifar-10-batches-py'
    train_data, train_label, test_data, test_label = load_CIFAR10(datasets)

    train_data = np.swapaxes(train_data, 1, 3)
    train_data = np.swapaxes(train_data, 2, 3)

    test_data = np.swapaxes(test_data, 1, 3)
    test_data = np.swapaxes(test_data, 2, 3)

    train_label = torch.tensor(train_label)
    test_label = torch.tensor(test_label)
    train_data = torch.tensor(train_data)
    test_data = torch.tensor(test_data)
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
    new_train_label_index = new_train_label.clone()
    rand_index = random.sample(range(new_train_set.size()[0]), num_data)
    new_train_CL_label_index = torch.index_select(train_label, dim=0, index=index)
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

    # index = []
    # for i in range(test_data.size()[0]):
    #     if test_label[i] == P  or test_label[i] == N or test_label[i] == A:
    #         index.append(i)
    #
    #
    # index = torch.tensor(index)
    # new_test_set = torch.index_select(test_data, dim=0, index=index)
    # new_test_label = torch.index_select(test_label, dim=0, index=index)
    # new_test_label_index = torch.index_select(test_label, dim=0, index=index)
    # for i in range(new_test_set.size()[0]):
    #     if new_test_label_index[i] == P:
    #         new_test_label[i] = 0
    #     if new_test_label_index[i] == N:
    #         new_test_label[i] = 1
    #     if new_test_label_index[i] == A:
    #         new_test_label[i] = 2
    # # print(new_test_label.size())
    # print('data set  Parameter： unknown set 3 known set 2 unlabel set 6 number of unknown and unlabel  {} {}'.format(
    #     num_data / 2, num_data / 2))



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

    print('train_x shape:%s, train_y shape:%s' % (train_data.shape, train_label.shape))
    print('test_x shape:%s, test_y shape:%s' % (test_data.shape, test_label.shape))
    return new_train_set, new_train_CL_label_index, new_test_set, new_test_label, data_prior


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

def read_data_CIFAR10_TF_MSL(i_train_iter, n):
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
    datasets = 'D:/code/data/cifar/cifar-10-python/cifar-10-batches-py'
    train_data, train_label, test_data, test_label = load_CIFAR10(datasets)

    train_data = np.swapaxes(train_data, 1, 3)
    train_data = np.swapaxes(train_data, 2, 3)

    test_data = np.swapaxes(test_data, 1, 3)
    test_data = np.swapaxes(test_data, 2, 3)

    train_label = torch.tensor(train_label)
    test_label = torch.tensor(test_label)
    train_data = torch.tensor(train_data)
    test_data = torch.tensor(test_data)

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

        # print('new_train_label = ', new_train_label[i])
        # print('new_train_SL_label_index = ', new_train_SL_label_index[i])
        # print('new_train_TF_label_index = ', new_train_TF_label_index[i])
    print("data N = ", n)
    new_train_TF_label_index, new_train_SL_label_index = MSL_GEN(new_train_label, new_train_set, new_train_TF_label_index, n)
    num_ALL = new_train_set.size()[0]
    num_t_label = torch.eq(new_train_TF_label_index, 9).sum()
    data_prior = num_t_label.float()/num_ALL

    print('data_prior', data_prior)
    data_prior_i = torch.eq(new_train_label_index, P2).sum() / num_ALL / (
                torch.eq(new_train_TF_label_index, P2).sum() / num_ALL)

    print('data_prior_i', data_prior_i)
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

    return new_train_set, new_train_TF_label_index, new_train_SL_label_index, new_test_set, new_test_label, data_prior


def read_data_CIFAR10_H_MSL(i_train_iter, n, conceal_label):
    random.seed(i_train_iter)
    P = []
    for i in [0,1,2,3,4,5,6,7,8,9]:
        if i!= conceal_label:
            P.append(i)
    P.append(conceal_label)
    datasets = 'D:/code/data/cifar/cifar-10-python/cifar-10-batches-py'
    train_data, train_label, test_data, test_label = load_CIFAR10(datasets)

    train_data = np.swapaxes(train_data, 1, 3)
    train_data = np.swapaxes(train_data, 2, 3)

    test_data = np.swapaxes(test_data, 1, 3)
    test_data = np.swapaxes(test_data, 2, 3)

    train_label = torch.tensor(train_label)
    test_label = torch.tensor(test_label)
    train_data = torch.tensor(train_data)
    test_data = torch.tensor(test_data)

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
    data_prior_hide = (num_0_label + num_1_label + num_2_label + num_3_label + num_2_label + num_4_label + num_5_label + num_6_label + num_7_label + num_8_label - num_H0_label - num_H1_label - num_H2_label \
                    - num_H3_label - num_H4_label - num_H5_label - num_H6_label - num_H7_label - num_H8_label) / num_t_label.float()
    print('data_prior', data_prior)
    data_prior_i = torch.eq(new_train_label_index, P[2]).sum() / num_ALL / (
                torch.eq(new_train_TF_label_index, P[2]).sum() / num_ALL)

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

    return new_train_set, new_train_TF_label_index, new_train_SL_label_index, new_test_set, new_test_label, data_prior, data_prior_hide