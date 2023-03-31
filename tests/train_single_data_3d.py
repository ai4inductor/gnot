#!/usr/bin/env python
#-*- coding:utf-8 _*-
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib
from scipy.spatial import cKDTree
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from gnot.models.MLP import MLP, FourierMLP
from gnot.utils import UnitTransformer,MinMaxTransformer

from gnot.tests.siren_demo import Siren


# data = np.load('OneCase3D_2a.npz')
# x,y,z, v = data['xxa'], data['yya'], data['zza'], data['vva']
#
# # data = np.load('OneCase3D_2b.npz')
# # x,y,z, v = data['xxb'], data['yyb'], data['zzb'], data['vvb']
# X = np.stack([x,y,z],axis=-1)
# Y = v[...,None]

data_path = './../data/inductor3d_A1_test.pkl'
dataset = pickle.load(open(data_path,'rb'))

idx = 0
component = 1
X, Y, theta, _ = dataset[idx]



class PostAct(nn.Module):
    def __init__(self, act='exp'):
        super(nn.Module,self).__init__()
        if act == 'exp':
            self.act = torch.exp
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.act(x)


def fast_find_duplicate_or_close_points(points, threshold=1e-5):
    kdtree = cKDTree(points)

    # 查询 k-d 树以找到距离小于阈值的点对
    pairs = kdtree.query_pairs(threshold)

    # 提取要剔除的点的索引
    indices_to_remove = set()
    for pair in pairs:
        indices_to_remove.add(pair[0])
        indices_to_remove.add(pair[1])

    return list(indices_to_remove)


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



duplicate_ids = fast_find_duplicate_or_close_points(X, 1e-10)

indices_to_keep = list(set(range(len(X))) - set(duplicate_ids))

print('delete points {} / {}'.format(len(X)-len(indices_to_keep),len(X)))
X, Y = X[indices_to_keep], Y[indices_to_keep]


X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()

# Y = 10**Y





normalizer = UnitTransformer(Y)
X = (X - X.mean())/X.std()




# X = torch.tanh(X)
Y = normalizer.transform(Y,inverse=False)
# Y = (Y-Y.mean())/Y.std()


## random throw some points
train_idxs = torch.randperm(Y.shape[0])[:int(1*Y.shape[0])]
X_all , Y_all = X.clone(), Y.clone()
X = X[train_idxs]
Y = Y[train_idxs]
print('Using {} / {} downsampling'.format(len(train_idxs), Y_all.shape[0]))

# net = MLP(3, 256, 1, 5, 'gelu')
# net = FourierMLP(3, n_hidden=256, output_size=1, n_layers=4, act='relu',fourier_dim=128,sigma=32)
net = Siren(in_features=3, out_features=1, hidden_layers=4,hidden_features=256,outermost_linear=True)


# post_act = PostAct('exp')
# net = nn.Sequential(net, post_act)

# 假设 net, X, Y 已经定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cm = plt.cm.get_cmap('rainbow')

net = net.to(device)
X = X.to(device)
Y = Y.to(device)
X_all = X_all.to(device)
Y_all = Y_all.to(device)

normalizer = normalizer.to(device)

ymin, ymax = Y.min(), Y.max()

criterion = nn.MSELoss()
lp_rel_err = lambda x,y,p: ((np.abs(x-y)**p).sum()/(np.abs(y)**p).sum())**(1/p)
optimizer = optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9,0.999))    # 3d problem, 1e-3 does not converge good for Siren

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

        outputs = net(X_all)
        # compute rel error
        Y_np_orig, outputs_np_orig = normalizer.transform(Y_all,inverse=True), normalizer.transform(outputs, inverse=True)

        X_np = X_all.detach().cpu().numpy()
        Y_np, Y_np_orig = Y_all.detach().cpu().numpy(), Y_np_orig.detach().cpu().numpy()
        outputs_np, outputs_np_orig = outputs.cpu().detach().numpy(), outputs_np_orig.cpu().detach().numpy()
        err1, err2 = lp_rel_err(outputs_np, Y_np, 1), lp_rel_err(outputs_np, Y_np, 2)
        err1_orig, err2_orig = lp_rel_err(outputs_np_orig, Y_np_orig, 1), lp_rel_err(outputs_np_orig, Y_np_orig, 2)
        print('epoch {} L1 rel error {} L2 rel error {} Orig space L1 {} L2 {} Max Y {} Max err {} mean {}'.format(epoch, err1, err2, err1_orig, err2_orig, np.abs(Y_np_orig).max(),np.abs(Y_np_orig-outputs_np_orig).max(),np.abs(Y_np_orig-outputs_np_orig).mean()))


        # fig = plt.figure(figsize=(12, 6))
        # gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])
        # ax0 = fig.add_subplot(gs[0], projection='3d')
        # ax1 = fig.add_subplot(gs[1], projection='3d')
        # cbar_ax = plt.subplot(gs[2])
        # scatters = []
        #
        # sc1 = ax0.scatter(X_np[:, 0], X_np[:, 1], X_np[:, 2], c=Y_np, cmap=cm, s=2)
        # ax0.set_title('True')
        # scatters.append(sc1)
        #
        # sc2 = ax1.scatter(X_np[:, 0], X_np[:, 1], X_np[:, 2], c=outputs_np - Y_np, cmap=cm, s=2)
        # ax1.set_title('Error')
        # scatters.append(sc2)
        #
        # ymin = np.min(Y_np)  # 请确保您已设置 ymin 和 ymax
        # ymax = np.max(Y_np)
        #
        # norm = matplotlib.colors.Normalize(vmin=ymin, vmax=ymax)
        # for sc in scatters:
        #     sc.set_norm(norm)
        #
        # fig.colorbar(scatters[0], cax=cbar_ax)
        #
        # plt.suptitle(f'Epoch {epoch + 1}')
        # plt.show()


        def plot_scatter_3d(X_np, Y_np, outputs_np, shared_colorbar=True, epoch=0):

            err = outputs_np - Y_np
            fig = plt.figure(figsize=(12, 6))
            gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])
            ax0 = fig.add_subplot(gs[0], projection='3d')
            ax1 = fig.add_subplot(gs[1], projection='3d')
            if shared_colorbar:
                cbar_ax = plt.subplot(gs[2])
            scatters = []

            sc1 = ax0.scatter(X_np[:, 0], X_np[:, 1], X_np[:, 2], c=Y_np, cmap=cm, s=2)
            ax0.set_title('True')
            scatters.append(sc1)

            sc2 = ax1.scatter(X_np[:, 0], X_np[:, 1], X_np[:, 2], c=err, cmap=cm, s=2)
            ax1.set_title('Error')
            scatters.append(sc2)

            ymin = np.min(Y_np)  # 请确保您已设置 ymin 和 ymax
            ymax = np.max(Y_np)

            norm = matplotlib.colors.Normalize(vmin=ymin, vmax=ymax)
            # print(ymin, ymax)

            if shared_colorbar:
                for sc in scatters:
                    sc.set_norm(norm)
                fig.colorbar(scatters[0], cax=cbar_ax)
            else:
                sc2.set_norm(matplotlib.colors.Normalize(vmin=err.min(), vmax=err.max()))
                fig.colorbar(sc1, ax=ax0)
                fig.colorbar(sc2, ax=ax1)

            plt.suptitle(f'Epoch {epoch + 1}')
            plt.show()

        ### show large values error
        # idxs = (Y_np_orig>0.9*Y_np_orig.max()).squeeze()
        # X_np, Y_np_orig, outputs_np_orig = X_np[idxs], Y_np_orig[idxs], outputs_np_orig[idxs]

        plot_scatter_3d(X_np, Y_np_orig, outputs_np_orig, shared_colorbar=False,epoch=epoch)
        # plot_scatter_3d(X_np, Y_np, outputs_np, shared_colorbar=False,epoch=epoch)

plt.figure()
plt.semilogy(np.arange(0,len(losses)), np.array(losses))
plt.show()


