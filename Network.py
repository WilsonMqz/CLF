import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)


class Net_add(nn.Module):  # torch.nn.Module所有神经网络模块的基类
    def __init__(self):
        super(Net, self).__init__()  # 对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
        # 输入(N, C_in, H, W) 输出(N, C_out, H_out, W_out)
        # MNIST图像：28 * 28
        # print('net', data_set)
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1)
        # self.conv3 = nn.Conv2d(16, 16, 3, 1)
        # self.conv4 = nn.Conv2d(16, 32, 3, 1)
        self.dropout1 = nn.Dropout2d(0.05)
        self.dropout2 = nn.Dropout2d(0.05)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 10)  # 10个数字，10类

    def forward(self, x):
        # 28 * 28
        # print(x.size())
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        # 26 * 26
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # 24 * 24
        # x = self.conv3(x)
        x = F.max_pool2d(x, 2)  # 池化
        # 12 * 12
        x = self.dropout1(x)

        # x = self.conv4(x)
        # x = F.max_pool2d(x, 2)  # 池化
        # 12 * 12
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # 9216 * 1
        # print(x.size())
        x = self.fc1(x)
        # 128 * 1
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # 10 * 1
        # output = F.log_softmax(x, dim=1)  # (N, C, H, W)  转换成概率分布的形式，并且取对数
        return x



class linear(nn.Module):  # torch.nn.Module所有神经网络模块的基类
    def __init__(self):
        super(linear, self).__init__()  # 对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
        # 输入(N, C_in, H, W) 输出(N, C_out, H_out, W_out)
        # MNIST图像：28 * 28
        # print('net', data_set)



        self.fc = nn.Linear(784, 10)  # 10个数字，10类

    def forward(self, x):
        # 28 * 28
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Net(nn.Module):  # torch.nn.Module所有神经网络模块的基类
    def __init__(self):
        super(Net, self).__init__()  # 对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
        # 输入(N, C_in, H, W) 输出(N, C_out, H_out, W_out)
        # MNIST图像：28 * 28
        # print('net', data_set)
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1)
        self.dropout1 = nn.Dropout2d(0.05)
        self.dropout2 = nn.Dropout2d(0.05)

        self.fc1 = nn.Linear(2304, 128)
        self.fc2 = nn.Linear(128, 10)  # 10个数字，10类

    def forward(self, x):
        # 28 * 28
        # print(x.size())
        x = self.conv1(x)
        # 26 * 26
        x = F.relu(x)
        x = self.conv2(x)
        # 24 * 24
        x = F.max_pool2d(x, 2)  # 池化
        # 12 * 12
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # 9216 * 1
        # print(x.size())
        x = self.fc1(x)
        # 128 * 1
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # 10 * 1
        # output = F.log_softmax(x, dim=1)  # (N, C, H, W)  转换成概率分布的形式，并且取对数
        return x


class CNN_Net_MNIST(nn.Module):  # torch.nn.Module所有神经网络模块的基类
    def __init__(self):
        super(CNN_Net_MNIST, self).__init__()  # 对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
        # 输入(N, C_in, H, W) 输出(N, C_out, H_out, W_out)
        # MNIST图像：28 * 28
        self.conv1 = nn.Conv2d(1, 16, 3, 1)  # 二维卷积层：输入通道数，输出通道数，kernel_size，stride
        self.conv1_1 = nn.Conv2d(16, 32, 1, 1)
        # self.conv1_1_1 = nn.Conv2d(128, 128, 1, 1)
        # self.dropout1_1 = nn.Dropout2d(0.5)
        # self.conv1_1_1 = nn.Conv2d(64, 64, 1, 1)
        # self.avg_pool_1 = nn.AdaptiveAvgPool2d(1)
        # self.conv1_2 = nn.Conv2d(32, 32, 3, 1)
        # self.conv1_se_2 = nn.Conv2d(32, 32, 1, 1)
        # self.avg_pool_2 = nn.AdaptiveAvgPool2d(1)
        self.BN1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 128, 3, 1)
        # self.conv2_1 = nn.Conv2d(64, 64, 1, 1)

        self.conv3 = nn.Conv2d(128, 1024, 1, 1)
        self.BN2 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout2d(0.01)
        self.dropout2 = nn.Dropout2d(0.01)
        self.dropout3 = nn.Dropout2d(0.01)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 10)  # 10个数字，10类

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 4)
        x = F.relu(x)
        # x_se_1 = self.avg_pool_1(x)
        x = self.conv1_1(x)
        # x = F.max_pool2d(x, 2)
        x = self.BN1(x)
        x = F.relu(x)

        # x = self.conv1_1_1(x)
        # # x = F.max_pool2d(x, 2)
        # # x = self.BN1(x)
        # x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.BN2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2)  # 池化
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        # print(x.size())
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)  # (N, C, H, W)  转换成概率分布的形式，并且取对数
        return x


class CNN_Net_ccld(nn.Module):  # torch.nn.Module所有神经网络模块的基类
    def __init__(self):
        super(CNN_Net_ccld, self).__init__()  # 对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
        self.conv1 = nn.Conv2d(3, 32, 7, 1)  # 二维卷积层：输入通道数，输出通道数，kernel_size，stride
        self.conv1_1 = nn.Conv2d(32, 128, 1, 1)
        self.conv1_1_1 = nn.Conv2d(128, 128, 1, 1)
        # self.dropout1_1 = nn.Dropout2d(0.5)
        # self.conv1_1_1 = nn.Conv2d(64, 64, 1, 1)
        # self.avg_pool_1 = nn.AdaptiveAvgPool2d(1)
        # self.conv1_2 = nn.Conv2d(32, 32, 3, 1)
        # self.conv1_se_2 = nn.Conv2d(32, 32, 1, 1)
        # self.avg_pool_2 = nn.AdaptiveAvgPool2d(1)
        self.BN1 = nn.BatchNorm2d(128)
        self.BN1_1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 128, 3, 1)
        # self.conv2_1 = nn.Conv2d(64, 64, 1, 1)

        self.conv3 = nn.Conv2d(128, 64, 1, 1)
        self.BN2 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout2d(0.05)
        self.dropout2 = nn.Dropout2d(0.05)
        self.dropout3 = nn.Dropout2d(0.05)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 3)  # 10个数字，10类

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 4)
        x = F.relu(x)
        # x_se_1 = self.avg_pool_1(x)
        x = self.conv1_1(x)
        # x = F.max_pool2d(x, 2)
        x = self.BN1(x)
        x = F.relu(x)

        x = self.conv1_1_1(x)
        x = F.max_pool2d(x, 4)
        x = self.BN1_1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.BN2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2)  # 池化
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        # print(x.size())
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)  # (N, C, H, W)  转换成概率分布的形式，并且取对数
        return x


class CNN_Net_cifar(nn.Module):  # torch.nn.Module所有神经网络模块的基类
    def __init__(self):
        super(CNN_Net_cifar, self).__init__()  # 对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
        # 输入(N, C_in, H, W) 输出(N, C_out, H_out, W_out)
        # MNIST图像：28 * 28
        self.conv1 = nn.Conv2d(3, 32, 3, 1)  # 二维卷积层：输入通道数，输出通道数，kernel_size，stride
        self.conv1_1 = nn.Conv2d(32, 64, 1, 1)
        # self.conv1_1_1 = nn.Conv2d(128, 128, 1, 1)
        # self.dropout1_1 = nn.Dropout2d(0.5)
        # self.conv1_1_1 = nn.Conv2d(64, 64, 1, 1)
        # self.avg_pool_1 = nn.AdaptiveAvgPool2d(1)
        # self.conv1_2 = nn.Conv2d(32, 32, 3, 1)
        # self.conv1_se_2 = nn.Conv2d(32, 32, 1, 1)
        # self.avg_pool_2 = nn.AdaptiveAvgPool2d(1)
        self.BN1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        # self.conv2_1 = nn.Conv2d(64, 64, 1, 1)

        self.conv3 = nn.Conv2d(128, 64, 1, 1)
        self.BN2 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout2d(0.01)
        self.dropout3 = nn.Dropout2d(0.01)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 10)  # 10个数字，10类

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 3)
        x = F.relu(x)
        # x_se_1 = self.avg_pool_1(x)
        x = self.conv1_1(x)
        # x = F.max_pool2d(x, 2)
        x = self.BN1(x)
        x = F.relu(x)

        # x = self.conv1_1_1(x)
        # # x = F.max_pool2d(x, 2)
        # # x = self.BN1(x)
        # x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.BN2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2)  # 池化
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        # print(x.size())
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)  # (N, C, H, W)  转换成概率分布的形式，并且取对数
        return x


class CNN_Net_smoking(nn.Module):  # torch.nn.Module所有神经网络模块的基类
    def __init__(self):
        super(CNN_Net_smoking, self).__init__()  # 对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
        # 输入(N, C_in, H, W) 输出(N, C_out, H_out, W_out)
        # MNIST图像：28 * 28
        self.conv1 = nn.Conv2d(3, 32, 7, 1)  # 二维卷积层：输入通道数，输出通道数，kernel_size，stride
        self.conv1_1 = nn.Conv2d(32, 128, 1, 1)
        self.conv1_1_1 = nn.Conv2d(128, 128, 1, 1)
        # self.dropout1_1 = nn.Dropout2d(0.5)
        # self.conv1_1_1 = nn.Conv2d(64, 64, 1, 1)
        # self.avg_pool_1 = nn.AdaptiveAvgPool2d(1)
        # self.conv1_2 = nn.Conv2d(32, 32, 3, 1)
        # self.conv1_se_2 = nn.Conv2d(32, 32, 1, 1)
        # self.avg_pool_2 = nn.AdaptiveAvgPool2d(1)
        self.BN1 = nn.BatchNorm2d(128)
        self.BN1_1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 128, 3, 1)
        # self.conv2_1 = nn.Conv2d(64, 64, 1, 1)

        self.conv3 = nn.Conv2d(128, 64, 1, 1)
        self.BN2 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout2d(0.05)
        self.dropout2 = nn.Dropout2d(0.05)
        self.dropout3 = nn.Dropout2d(0.05)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 4)  # 10个数字，10类

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 4)
        x = F.relu(x)
        # x_se_1 = self.avg_pool_1(x)
        x = self.conv1_1(x)
        # x = F.max_pool2d(x, 2)
        x = self.BN1(x)
        x = F.relu(x)

        x = self.conv1_1_1(x)
        x = F.max_pool2d(x, 4)
        x = self.BN1_1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.BN2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2)  # 池化
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        # print(x.size())
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)  # (N, C, H, W)  转换成概率分布的形式，并且取对数
        return x



class CNN_Net_SVHN(nn.Module):  # torch.nn.Module所有神经网络模块的基类
    def __init__(self):
        super(CNN_Net_SVHN, self).__init__()  # 对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
        # 输入(N, C_in, H, W) 输出(N, C_out, H_out, W_out)
        # MNIST图像：28 * 28
        self.conv1 = nn.Conv2d(3, 64, 3, 1)  # 二维卷积层：输入通道数，输出通道数，kernel_size，stride
        self.conv1_1 = nn.Conv2d(64, 128, 1, 1)
        # self.conv1_1_1 = nn.Conv2d(128, 128, 1, 1)
        # self.dropout1_1 = nn.Dropout2d(0.5)
        # self.conv1_1_1 = nn.Conv2d(64, 64, 1, 1)
        # self.avg_pool_1 = nn.AdaptiveAvgPool2d(1)
        # self.conv1_2 = nn.Conv2d(32, 32, 3, 1)
        # self.conv1_se_2 = nn.Conv2d(32, 32, 1, 1)
        # self.avg_pool_2 = nn.AdaptiveAvgPool2d(1)
        self.BN1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 64, 3, 1)
        # self.conv2_1 = nn.Conv2d(64, 64, 1, 1)

        self.conv3 = nn.Conv2d(64, 64, 1, 1)
        self.BN2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout2d(0.5)
        self.dropout3 = nn.Dropout2d(0.8)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 3)  # 10个数字，10类

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 3)
        x = F.relu(x)
        # x_se_1 = self.avg_pool_1(x)
        x = self.conv1_1(x)
        # x = F.max_pool2d(x, 2)
        x = self.BN1(x)
        x = F.relu(x)

        # x = self.conv1_1_1(x)
        # # x = F.max_pool2d(x, 2)
        # # x = self.BN1(x)
        # x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.BN2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2)  # 池化
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        # print(x.size())
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)  # (N, C, H, W)  转换成概率分布的形式，并且取对数
        return x

