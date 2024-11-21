import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def c_loss(output, label):
    if output.size()[0] == 0:
        output = 0
    else:
        loss = nn.MSELoss(size_average=True).to(device)
        class_mun = 10
        one_hot = F.one_hot(label.to(torch.int64), class_mun) * 2 - 1
        sig_out = output * one_hot.to(device)
        y_label = torch.ones(sig_out.size()).to(device)
        output = loss(sig_out, y_label)
    return output


def ConL_risk_MSL(output, target_TF, target_SL, data_prior, n, rate):
    class_mun = 9
    index_S = []
    index_F = []
    # print(n)

    for i in range(target_TF.size()[0]):
        if target_TF[i] == 9:
            index_F.append(i)
        else:
            index_S.append(i)
    index_S = torch.tensor(index_S).long().to(device)
    index_F = torch.tensor(index_F).long().to(device)

    target_S = torch.index_select(target_TF, dim=0, index=index_S).to(device)

    output_S = torch.index_select(output, dim=0, index=index_S).to(device)
    output_F = torch.index_select(output, dim=0, index=index_F).to(device)

    label_1 = torch.ones(output_F.size()[0]).to(device)
    label_1_a = torch.ones(output.size()[0]).to(device)
    loss = (class_mun / n) * c_loss(output_S, target_S) + (class_mun / n) * c_loss(output_F, label_1 * 9) - (
                class_mun - n) / n * c_loss(output, label_1_a * 9)
    return loss


def train_multi_split_MSL(args, model, train_loader, optimizer_kn, epoch, N, data_prior, rate):
    model.train()  # 针对在网络train和eval时采用不同方式的情况，比如 BatchNormalization 和 Dropout
    for batch_idx, (data, target_TF, target_SL) in enumerate(train_loader):
        data = torch.tensor(data, dtype=torch.float32).to(device)

        optimizer_kn.zero_grad()
        output = model(data)
        # loss = CLAC_loss_fourclass(output, target, data_prior)
        # loss = TF_risk(output, target_TF, target_SL, data_prior)
        # print("train N = ", N)
        # data_prior = data_prior * rate
        # loss = TF_risk_MSL(output, target_TF.to(device), target_SL.to(device), data_prior.to(device), N, rate)
        loss = ConL_risk_MSL(output, target_TF.to(device), target_SL.to(device), data_prior.to(device), N, rate)
        # loss = TF_smoking_risk_MSL(output, target_TF, target_SL, data_prior, N)
        # loss = TF_cldd_MSL(output, target_TF, target_SL, data_prior, N, rate)
        # print('target_SL', target_SL.size())
        # loss = CLAC_loss(output, target, data_prior)
        # loss = CLAC_loss_sevi(output, target, data_prior)
        loss.backward()
        optimizer_kn.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch： {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))  # item获得一个元素tensor的value
