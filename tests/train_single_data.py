#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from gnot.models.MLP import MLP, FourierMLP
from gnot.utils import UnitTransformer,MinMaxTransformer

from gnot.tests.siren_demo import Siren

# data_path = './../data/inductor2d_nomask_1100_test.pkl'
data_path = './../data/inductor2d_bosch_test.pkl'
# data_path = './../data/ns2d_nomask_4ball_2200_test.pkl'
dataset = pickle.load(open(data_path,'rb'))

idx = 18
component = 1
X, Y, theta, _ = dataset[idx]


### remove duplicated points
def find_duplicate_or_close_points(points, threshold=1e-5):
    n = points.shape[0]

    # 计算所有点对之间的距离
    dist_matrix = np.sqrt(np.sum((points[:, np.newaxis] - points[np.newaxis, :]) ** 2, axis=2))

    # 将对角线上的值设为一个很大的数，以避免错误地认为同一个点是重复的
    np.fill_diagonal(dist_matrix, np.inf)

    # 寻找距离小于等于阈值的点对
    duplicate_or_close_points = np.argwhere(dist_matrix <= threshold)

    return duplicate_or_close_points



duplicate_ids = find_duplicate_or_close_points(X, 1e-10)

indices_to_keep = list(set(range(len(X))) - set(duplicate_ids[:, 0])-set(duplicate_ids[:,1]))

print('delete points {} / {}'.format(len(X)-len(indices_to_keep),len(X)))
X, Y = X[indices_to_keep], Y[indices_to_keep]


X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y[:,component:component + 1]).float()




normalizer = UnitTransformer(Y)
X = (X - X.mean())/X.std()




X = torch.tanh(X)
Y = normalizer.transform(Y,inverse=False)
# Y = (Y-Y.mean())/Y.std()
# net = MLP(2, 64, 1, 5, 'relu')
net = FourierMLP(2, n_hidden=128, output_size=1, n_layers=5, act='relu',fourier_dim=128,sigma=10,type='exp')
# net = Siren(in_features=2, out_features=1, hidden_layers=4,hidden_features=128,outermost_linear=True)


# 假设 net, X, Y 已经定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cm = plt.cm.get_cmap('rainbow')

net = net.to(device)
X = X.to(device)
Y = Y.to(device)
normalizer = normalizer.to(device)

ymin, ymax = Y.min(), Y.max()

criterion = nn.MSELoss()
lp_rel_err = lambda x,y,p: ((np.abs(x-y)**p).sum()/(np.abs(y)**p).sum())**(1/p)
optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9,0.99))

num_epochs = 10000
plot_interval = num_epochs //5
losses = []

for epoch in range(num_epochs):
    # 梯度清零
    optimizer.zero_grad()

    # 前向传播
    outputs = net(X)

    # 计算损失
    loss = criterion(outputs, Y)

    # 反向传播
    loss.backward()

    # 更新权重
    optimizer.step()

    losses.append(loss.item())

    # 每隔 plot_interval 个 epoch 绘制散点图
    if (epoch + 1) % plot_interval == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # compute rel error
        Y_np_orig, outputs_np_orig = normalizer.transform(Y,inverse=True), normalizer.transform(outputs, inverse=True)

        X_np = X.detach().cpu().numpy()
        Y_np, Y_np_orig = Y.detach().cpu().numpy(), Y_np_orig.detach().cpu().numpy()
        outputs_np, outputs_np_orig = outputs.cpu().detach().numpy(), outputs_np_orig.cpu().detach().numpy()
        err1, err2 = lp_rel_err(outputs_np, Y_np, 1), lp_rel_err(outputs_np, Y_np, 2)
        err1_orig, err2_orig = lp_rel_err(outputs_np_orig, Y_np_orig, 1), lp_rel_err(outputs_np_orig, Y_np_orig, 2)
        print('epoch {} L1 rel error {} L2 rel error {} Orig space L1 {} L2 {}'.format(epoch, err1, err2, err1_orig, err2_orig))

        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        cbar_ax = plt.subplot(gs[2])
        scatters = []

        sc1 = ax0.scatter(X_np[:, 0], X_np[:, 1], c=Y_np, cmap=cm, s=2)
        ax0.set_title('True')
        scatters.append(sc1)

        # sc2 = ax1.scatter(X_np[:, 0], X_np[:, 1], c=outputs_np, cmap=cm, s=2)
        # ax1.set_title('Predicted')
        # scatters.append(sc2)

        sc2 = ax1.scatter(X_np[:, 0], X_np[:, 1], c=outputs_np - Y_np, cmap=cm, s=2)
        ax1.set_title('Error')
        scatters.append(sc2)

        norm = matplotlib.colors.Normalize(vmin=ymin, vmax=ymax)
        for sc in scatters:
            sc.set_norm(norm)

        # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(scatters[0], cax=cbar_ax)

        plt.suptitle(f'Epoch {epoch + 1}')
        plt.show()



plt.figure()
plt.semilogy(np.arange(0,len(losses)), np.array(losses))
plt.show()

