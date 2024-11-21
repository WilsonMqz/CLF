import torch
import torch.nn.functional as F
import train
import math
import itertools
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# def test(args, model, device, test_loader):
#     model.eval()  # 测试模式
#     test_loss_1 = 0
#     correct_1 = 0
#     with torch.no_grad():  # 数据不需要计算梯度，也不会进行反向传播
#         for data, target in test_loader:
#             # data, target = data.float.to(device), target.to(device)
#             data = data.unsqueeze(1)
#             data = torch.tensor(data, dtype=torch.float32)
#             output = model(data)
#             test_loss_1 += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct_1 += pred.eq(target.view_as(pred)).sum().item()
#             # view_as:返回被视作与给定的tensor相同大小的原tensor
#
#
#         test_loss_2 = 0
#         correct_2 = 0
#         for data, target in test_loader:
#             # data, target = data.float.to(device), target.to(device)
#             data = data.unsqueeze(1)
#             data = torch.tensor(data, dtype=torch.float32)
#             output = model(data)
#             for i in range (target.size()[0]):
#                 if target[i] == 2:
#                     target[i] = 3
#                 else:
#                     if target[i] == 3:
#                         target[i] = 2
#             test_loss_2 += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct_2 += pred.eq(target.view_as(pred)).sum().item()
#
#         test_loss_3 = 0
#         correct_3 = 0
#         for data, target in test_loader:
#             # data, target = data.float.to(device), target.to(device)
#             data = data.unsqueeze(1)
#             data = torch.tensor(data, dtype=torch.float32)
#             output = model(data)
#             for i in range(target.size()[0]):
#                 if target[i] == 2:
#                     target[i] = 4
#                 else:
#                     if target[i] == 4:
#                         target[i] = 2
#             test_loss_3 += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct_3 += pred.eq(target.view_as(pred)).sum().item()
#
#         test_loss_4 = 0
#         correct_4 = 0
#         for data, target in test_loader:
#             # data, target = data.float.to(device), target.to(device)
#             data = data.unsqueeze(1)
#             data = torch.tensor(data, dtype=torch.float32)
#             output = model(data)
#             for i in range(target.size()[0]):
#                 if target[i] == 3:
#                     target[i] = 4
#                 else:
#                     if target[i] == 4:
#                         target[i] = 3
#             test_loss_4 += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct_4 += pred.eq(target.view_as(pred)).sum().item()
#
#
#         # view_as:返回被视作与给定的tensor相同大小的原tensor
#     # test_loss是累计了所有batch的loss， 所以取平均
#     test_loss_1 = test_loss_1 / len(test_loader)
#     test_loss_2 = test_loss_2 / len(test_loader)
#     test_loss_3 = test_loss_3 / len(test_loader)
#     test_loss_4 = test_loss_4 / len(test_loader)
#     correct = max(correct_2,correct_3,correct_3,correct_4)
#     print(test_loss_1, test_loss_2,test_loss_3,test_loss_4)
#     print(correct_2,correct_1,correct_3,correct_4)
#     test_loss = min (test_loss_1, test_loss_2,test_loss_3,test_loss_4)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))




def test_multi( model,  test_loader):
    model.eval()  # 测试模式
    test_loss = 0# torch.zeros([6, 1])
    correct = 0# torch.zeros([6, 1])
    tp = 0
    new_class = 0
    # j = 0
    with torch.no_grad():  # 数据不需要计算梯度，也不会进行反向传播
        # nums = [0, 1, 2]
        # for num in itertools.permutations(nums):
            # print(num)
        for data, target in test_loader:
            # data, target = data.float.to(device), target.to(device)
            # data = data.unsqueeze(1)
            data = torch.tensor(data, dtype=torch.float32).to(device)
            output = model(data)
            test_loss += train.c_loss(output, target.long()).item()  # sum up batch loss
            # test_loss += torch.mean(train.c_loss(output, target.long()))
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for i in range(target.to(device).view_as(pred).size()[0]):
                if target.view_as(pred)[i] == 9:
                    new_class = new_class + 1
                    if pred[i] == 9:
                        tp = tp + 1
                # print('target.view_as(pred), pred', target.view_as(pred)[i], pred[i])
            # print('pred', pred)
            # tp = 1
            # new_class = 1
            correct += pred.eq(target.to(device).view_as(pred)).sum().item()

        test_loss = test_loss / len(test_loader)


    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%), TPR: ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), 100.*tp / new_class))
    # print('TPR = ', 100.*tp/new_class)
    return 100. * correct / len(test_loader.dataset), 100.*tp / new_class



def c_loss(output, label):
    if output.size()[0] == 0:
        output = 0
    else:
        loss = nn.MSELoss(size_average=True)
        class_mun = 10
        one_hot = F.one_hot(label.to(torch.int64), class_mun) * 2 - 1
        sig_out = output * one_hot
        y_label = torch.ones(sig_out.size())
        output = loss(sig_out, y_label)
    return output

def TF_risk_MSL(output, target_TF, target_SL,   data_prior, n):
    class_mun = 10
    index_T_0 = []
    index_T_1 = []
    index_T_2 = []
    index_T_3 = []
    index_T_4 = []
    index_T_5 = []
    index_T_6 = []
    index_T_7 = []
    index_T_8 = []
    index_F = []
    # print(n)

    for i in range(target_TF.size()[0]):
        if target_TF[i] == 0:
            index_T_0.append(i)
        elif target_TF[i] == 1:
            index_T_1.append(i)
        elif target_TF[i] == 2:
            index_T_2.append(i)
        elif target_TF[i] == 3:
            index_T_3.append(i)
        elif target_TF[i] == 4:
            index_T_4.append(i)
        elif target_TF[i] == 5:
            index_T_5.append(i)
        elif target_TF[i] == 6:
            index_T_6.append(i)
        elif target_TF[i] == 7:
            index_T_7.append(i)
        elif target_TF[i] == 8:
            index_T_8.append(i)
        elif target_TF[i] == 9:
            index_F.append(i)
    index_T_0 = torch.tensor(index_T_0).long()
    index_T_1 = torch.tensor(index_T_1).long()
    index_T_2 = torch.tensor(index_T_2).long()
    index_T_3 = torch.tensor(index_T_3).long()
    index_T_4 = torch.tensor(index_T_4).long()
    index_T_5 = torch.tensor(index_T_5).long()
    index_T_6 = torch.tensor(index_T_6).long()
    index_T_7 = torch.tensor(index_T_7).long()
    index_T_8 = torch.tensor(index_T_8).long()
    index_F = torch.tensor(index_F).long()
    # target_SL_T_0 = torch.index_select(target_SL, dim=0, index=index_T)

    # target_SL_F = np.zeros((index_F.size()[0], n))
    target_SL_F = torch.index_select(target_SL, dim=0, index=index_F)




    target_T_0 = torch.index_select(target_TF, dim=0, index=index_T_0)
    target_T_1 = torch.index_select(target_TF, dim=0, index=index_T_1)
    target_T_2 = torch.index_select(target_TF, dim=0, index=index_T_2)
    target_T_3 = torch.index_select(target_TF, dim=0, index=index_T_3)
    target_T_4 = torch.index_select(target_TF, dim=0, index=index_T_4)
    target_T_5 = torch.index_select(target_TF, dim=0, index=index_T_5)
    target_T_6 = torch.index_select(target_TF, dim=0, index=index_T_6)
    target_T_7 = torch.index_select(target_TF, dim=0, index=index_T_7)
    target_T_8 = torch.index_select(target_TF, dim=0, index=index_T_8)


    output_T_0 = torch.index_select(output, dim=0, index=index_T_0)
    output_T_1 = torch.index_select(output, dim=0, index=index_T_1)
    output_T_2 = torch.index_select(output, dim=0, index=index_T_2)
    output_T_3 = torch.index_select(output, dim=0, index=index_T_3)
    output_T_4 = torch.index_select(output, dim=0, index=index_T_4)
    output_T_5 = torch.index_select(output, dim=0, index=index_T_5)
    output_T_6 = torch.index_select(output, dim=0, index=index_T_6)
    output_T_7 = torch.index_select(output, dim=0, index=index_T_7)
    output_T_8 = torch.index_select(output, dim=0, index=index_T_8)
    output_F = torch.index_select(output, dim=0, index=index_F)
    num_sl = target_SL_F.size()[0]
    label_9 = torch.ones(num_sl) * 9
    label_1 = torch.ones(output_F.size()[0])
    label_1_a = torch.ones(output.size()[0])

    loss = (c_loss(output_T_0, target_T_0) + c_loss(output_T_1, target_T_1) + c_loss(output_T_2, target_T_2)  + c_loss(output_T_3, target_T_3)  + c_loss(output_T_4, target_T_4) + c_loss(output_T_5, target_T_5) + c_loss(output_T_6, target_T_6) + c_loss(output_T_7, target_T_7) + c_loss(output_T_8, target_T_8)) \
           + 1/class_mun*(c_loss(output_F, label_1*0)+ c_loss(output_F, label_1*1) + c_loss(output_F, label_1*2)+ c_loss(output_F, label_1*3)+ c_loss(output_F, label_1*4)+ c_loss(output_F, label_1*5)+ c_loss(output_F, label_1*6)+ c_loss(output_F, label_1*7)+ c_loss(output_F, label_1*8)) \
           - 1/(class_mun+1)/class_mun * (c_loss(output, label_1_a*0)+ c_loss(output, label_1_a*1) + c_loss(output, label_1_a*2) + c_loss(output, label_1_a*3)+ c_loss(output, label_1_a*4)+ c_loss(output, label_1_a*5)+ c_loss(output, label_1_a*6)+ c_loss(output, label_1_a*7)+ c_loss(output, label_1_a*8))\
           + class_mun / n * (c_loss(output_F, label_1*9))\
           -(class_mun - n)/n * c_loss(output, label_1_a*9)

    return loss


def train_multi(model,  train_loader, N,  data_prior):
    model.eval()  # 测试模式
    train_loss = 0# torch.zeros([6, 1])
    correct = 0# torch.zeros([6, 1])
    tp = 0
    new_class = 0
    # j = 0
    with torch.no_grad():  # 数据不需要计算梯度，也不会进行反向传播
        # nums = [0, 1, 2]
        # for num in itertools.permutations(nums):
            # print(num)
        for batch_idx, (data, target_TF, target_SL) in enumerate(train_loader):
            # data, target = data.float.to(device), target.to(device)
            # data = data.unsqueeze(1)
            data = torch.tensor(data, dtype=torch.float32)
            output = model(data)
            train_loss += TF_risk_MSL(output, target_TF, target_SL, data_prior, N).item()  # sum up batch loss
            # test_loss += torch.mean(train.c_loss(output, target.long()))
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # for i in range(target.view_as(pred).size()[0]):
            #     if target.view_as(pred)[i] == 9:
            #         new_class = new_class + 1
            #         if pred[i] == 9:
            #             tp = tp + 1
                # print('target.view_as(pred), pred', target.view_as(pred)[i], pred[i])
            # print('pred', pred)
            # tp = 1
            # new_class = 1
            # correct += pred.eq(target.view_as(pred)).sum().item()

        # test_loss = test_loss / len(test_loader)


    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%), TPR: ({:.3f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset), 100.*tp / new_class))
    # print('TPR = ', 100.*tp/new_class)
    return train_loss/ batch_idx

