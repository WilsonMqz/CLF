import argparse
import time

import numpy
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader

import Network
import Resnet
import abs_train
import cifar_10_data
import cldd_data
import minist_data
import smoking_data
import test
import train


def data_gen(new_train_set, new_train_TF_label, new_train_SL_label_0, new_train_SL_label_1, new_test_set,
             new_test_label):
    train_dataset = TensorDataset(new_train_set, new_train_TF_label, new_train_SL_label_0, new_train_SL_label_1)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=512,
                              shuffle=True)

    test_dataset = TensorDataset(new_test_set, new_test_label)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=256,
                             shuffle=True)
    return train_loader, test_loader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
    rato = 0.4
    # Training settings
    parser = argparse.ArgumentParser(description='Pytorch MNIST Example')
    parser.add_argument('--bs', type=int, default=1024, metavar='N',
                        help='input batch size for training (default:64)')
    parser.add_argument('--tbs', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default:1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of eopchs to train （default:14)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default:1)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--N', type=int, default=1, metavar='N',
                        help='the number of sample classes')
    parser.add_argument('--data_set', type=int, default=1, metavar='N',
                        help='minist or fashion minist or Kuzushiji minist')
    parser.add_argument('--data_choice', type=int, default=1, metavar='N',
                        help='minist=1 or SVHN or cifar=3 or smoking =5 or cldd=6')
    parser.add_argument('--trials_choice', type=int, default=3, metavar='N',
                        help='number of iter')

    args = parser.parse_args()

    def data_gen_MSL(new_train_set, new_train_TF_label, new_train_SL_label, new_test_set, new_test_label):
        train_dataset = TensorDataset(new_train_set, new_train_TF_label, new_train_SL_label)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.bs,
                                  shuffle=True, num_workers=4, pin_memory=True)

        test_dataset = TensorDataset(new_test_set, new_test_label)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=args.bs,
                                 shuffle=True, num_workers=4, pin_memory=True)
        return train_loader, test_loader

    for rate in [1]:
        for sample_classes in [1]:
            for conceal_label in [1, 3, 5, 7, 9]:
                for method_i in range(2, 4):
                    # method_2 and method_3 denote the two types of losses respectively.
                    max_test_acc = torch.zeros(args.trials_choice)
                    max_test_tpr = torch.zeros(args.trials_choice)
                    for i_train_iter in range(args.trials_choice):
                        if args.data_choice == 1:
                            # new_train_set, new_train_TF_label, new_train_SL_label_0, new_train_SL_label_1,  new_test_set, new_test_label, data_prior = minist_data.read_data_minist_TF(args.data_set, i_train_iter) #read data 1:minist 2:fashion_minist
                            new_train_set, new_train_TF_label, new_train_SL_label, new_test_set, new_test_label, data_prior, data_prior_hide = minist_data.read_data_minist_TF_MSL(
                                args.data_set, i_train_iter, sample_classes, conceal_label)
                            print("main N = ", sample_classes)

                            if method_i == 2:
                                model = Network.CNN_Net_MNIST().to(device)
                                if i_train_iter == 0:
                                    lr_known = 2e-1
                                    known_gamma = 0.5
                                    step_size = 15

                            elif method_i == 3:
                                model = Network.linear()
                                if i_train_iter == 0:
                                    lr_known = 2e-2
                                    known_gamma = 0.5
                                    step_size = 10

                        elif args.data_choice == 3:
                            new_train_set, new_train_TF_label, new_train_SL_label, new_test_set, new_test_label, data_prior, data_prior_hide = cifar_10_data.read_data_CIFAR10_H_MSL(
                                i_train_iter, sample_classes, conceal_label)

                            if method_i == 2:
                                model = Network.CNN_Net_cifar().to(device)
                                if i_train_iter == 0:
                                    lr_known = 8e-1
                                    known_gamma = 0.5
                                    step_size = 10

                            elif method_i == 3:
                                model = Resnet.resnet34().to(device)
                                if i_train_iter == 0:
                                    lr_known = 4e-1
                                    known_gamma = 0.5
                                    step_size = 20

                        elif args.data_choice == 5:
                            new_train_set, new_train_TF_label, new_train_SL_label, new_test_set, new_test_label, data_prior, data_prior_hide = smoking_data.read_data_smoking_TF_MSL(
                                i_train_iter, sample_classes)
                            if method_i == 2:
                                model = Network.CNN_Net_smoking()
                                if i_train_iter == 0:
                                    parser.add_argument('--lr_known', type=float, default=1.5e-1, metavar='LR_m',
                                                        # metavar用在help信息的输出中
                                                        help='learning rate for known(default:1.0)')
                                    parser.add_argument('--known_gamma', type=float, default=0.5, metavar='M',
                                                        help='Learning rate step gamma (default:0.5)')
                                    parser.add_argument('--step_size', type=int, default=20, metavar='N',
                                                        help='size')
                            elif method_i == 3:
                                model = Network.CNN_Net_smoking()
                                if i_train_iter == 0:
                                    parser.add_argument('--lr_known', type=float, default=0.6e-1, metavar='LR_m',
                                                        # metavar用在help信息的输出中
                                                        help='learning rate for known(default:1.0)')
                                    parser.add_argument('--known_gamma', type=float, default=0.5, metavar='M',
                                                        help='Learning rate step gamma (default:0.5)')
                                    parser.add_argument('--step_size', type=int, default=20, metavar='N',
                                                        help='size')

                        elif args.data_choice == 6:
                            new_train_set, new_train_TF_label, new_train_SL_label, new_test_set, new_test_label, data_prior, data_prior_hide = cldd_data.read_data_cldd_TF_MSL(
                                i_train_iter, sample_classes, conceal_label)
                            print("main N = ", sample_classes)

                            if method_i == 2:
                                model = Network.CNN_Net_ccld()
                                if i_train_iter == 0:
                                    lr_known = 6e-2  # fashion-mnist 8e-2, Ku 8e-1
                                    known_gamma = 0.5
                                    step_size = 10

                            elif method_i == 3:
                                model = Network.CNN_Net_ccld()
                                if i_train_iter == 0:
                                    lr_known = 1.5e-1
                                    known_gamma = 0.1
                                    step_size = 10

                        args = parser.parse_args()
                        numpy.random.seed(i_train_iter)
                        # print(model)
                        optimizer_known = optim.Adadelta(model.parameters(), lr=lr_known)
                        scheduler_known = StepLR(optimizer_known, step_size=step_size, gamma=known_gamma)  ####0.98

                        print(
                            'data choice: {}\n 1:minist 2:SVHN 3:cifar \n mnist data set: {} \n 1: minist 2: fashion minsit 3: Kuzushiji_minist\n Train Parameter：step size {}, known gamma {}, known lr {}\n model chioce: {}'.format(
                                args.data_choice, args.data_set, step_size, known_gamma, lr_known, method_i))

                        # train_loader, test_loader = data_gen(new_train_set, new_train_TF_label,  new_train_SL_label_0, new_train_SL_label_1,new_test_set, new_test_label)
                        train_loader, test_loader = data_gen_MSL(new_train_set, new_train_TF_label, new_train_SL_label,
                                                                 new_test_set, new_test_label)

                        test_acc = torch.zeros(args.epochs)
                        test_tpr = torch.zeros(args.epochs)
                        for epoch in range(1, args.epochs + 1):
                            start = time.perf_counter()
                            if method_i == 2:
                                train.train_multi_split_MSL(args, model, train_loader, optimizer_known, epoch,
                                                            sample_classes, data_prior, rate)
                            elif method_i == 3:
                                abs_train.train_multi_split_MSL(args, model, train_loader, optimizer_known, epoch,
                                                                sample_classes, data_prior)
                            scheduler_known.step()
                            test_acc[epoch - 1], test_tpr[epoch - 1] = test.test_multi(model, test_loader)
                            end = time.perf_counter()
                            print('Running time: %s Seconds' % (end - start))
                        max_test_acc[i_train_iter] = torch.max(test_acc, 0)[0]

                        max_test_tpr[i_train_iter] = test_tpr[torch.max(test_acc, 0)[1]]

                    acc_mean = torch.mean(max_test_acc)
                    acc_std = torch.std(max_test_acc)
                    tpr_mean = torch.mean(max_test_tpr)
                    tpr_std = torch.std(max_test_tpr)
                    print(max_test_acc, max_test_tpr)
                    print(acc_mean, acc_std, tpr_mean, tpr_std)
                    print("\033[1;31m cl = \033[0m", conceal_label)


if __name__ == '__main__':
    main()
