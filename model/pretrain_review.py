import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch as th
import dgl
import sys
from model import MLP

sys.path.append("..")

class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
def cons_loss(x, x_aug):
    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss
def pretrain_review_feat(train_feat, test_feat, rating_train, rating_test, device='cuda:2'):

    train_feat = np.array([x.cpu().detach().numpy() for x in train_feat])
    test_feat = np.array([x.cpu().detach().numpy() for x in test_feat])
    rating_train = np.array(rating_train)
    rating_test = np.array(rating_test)
    train_data = GetLoader(train_feat, rating_train)
    test_data = GetLoader(test_feat, rating_test)

    train_loader = torch.utils.data.DataLoader(train_data,batch_size=2048, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=2048, shuffle=False)

    mlp = MLP(64, 1).to(device)

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adagrad(mlp.parameters(), lr=0.005)
    trained_feat = None
    for epoch in range(300):
        mlp.train()
        loss_sum, loss_ssl_sum = 0, 0
        for data, y in train_loader:
            data = torch.tensor(data, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)
            z, pred = mlp(data)
            loss = loss_func(pred, y).mean()
            loss_ssl = cons_loss(z, data)
            # loss += 0.005*loss_ssl
            loss_sum += loss.item()*(data.shape[0])
            loss_ssl_sum += loss_ssl.item()*(data.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('loss_train:', loss_sum/train_feat.shape[0], loss_ssl_sum/train_feat.shape[0])
        mlp.eval()
        loss_sum = 0
        for data, y in test_loader:
            data = torch.tensor(data, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)
            z, pred = mlp(data)
            loss = loss_func(pred, y).mean()
            loss_sum += loss.item()*data.shape[0]
        print('loss:', loss_sum / test_feat.shape[0])
    with torch.no_grad():
        for data, y in train_loader:
            data = torch.tensor(data, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)
            z, pred = mlp(data)
            if trained_feat is None:
                trained_feat = z
            else:
                trained_feat = torch.cat([trained_feat, z], dim=0)
        return trained_feat.to('cpu')

