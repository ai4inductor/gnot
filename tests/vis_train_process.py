#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import pickle
import torch
import numpy as np
import torch.nn as nn
import dgl
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from scipy import interpolate
from matplotlib.animation import FuncAnimation

from matplotlib.animation import ArtistAnimation



#
# def scatter_save_video(x, pred, err, xlim, ylim, vlim):
#
#     # 生成一些测试数据
#     N = x.shape[0]  # 数据点个数
#     M = pred.shape[1]  # 视频帧数
#
#     # 初始化图形
#     fig, ax = plt.subplots(figsize=(6, 6))
#     fig = plt.figure(figsize=(10, 5))
#     gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])
#
#
#     ax.set_xlim(xlim[0], xlim[1])
#     ax.set_ylim(ylim[0], ylim[1])
#
#     scatter = ax.scatter(x[:, 0], x[:, 1], c=pred[:, 0], cmap='Spectral')
#     norm = matplotlib.colors.Normalize(vmin=v_lim[0], vmax=v_lim[1])
#
#     fig.colorbar(scatter)
#
#
#     def update(i):
#         # scatter.set_offsets(x)
#         scatter.set_array(pred[:, i])
#         scatter.set_norm(norm)
#
#         print(i)
#         return scatter,
#
#     # plt.figure()
#     # plt.colorbar()
#     anim = FuncAnimation(fig, update, frames=M, blit=True,)
#
#     # 显示动画
#     # plt.show()
#     return anim


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import matplotlib.colors


def scatter_save_video(x, pred, err, xlim, ylim, vlim):
    N = x.shape[0]  # 数据点个数
    M = pred.shape[1]  # 视频帧数

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])

    ax1 = plt.subplot(gs[0])
    ax1.set_xlim(xlim[0], xlim[1])
    ax1.set_ylim(ylim[0], ylim[1])
    ax1.set_title('Pred')

    ax2 = plt.subplot(gs[1])
    ax2.set_xlim(xlim[0], xlim[1])
    ax2.set_ylim(ylim[0], ylim[1])
    ax2.set_title('Error')

    norm = matplotlib.colors.Normalize(vmin=vlim[0], vmax=vlim[1])

    scatter_pred = ax1.scatter(x[:, 0], x[:, 1], c=pred[:, 0], cmap='Spectral', norm=norm)
    scatter_err = ax2.scatter(x[:, 0], x[:, 1], c=err[:, 0], cmap='Spectral', norm=norm)

    cax = plt.subplot(gs[2:])
    fig.colorbar(scatter_pred, cax=cax)

    def update(i):
        scatter_pred.set_array(pred[:, i])
        scatter_err.set_array(err[:, i])

        print(i)
        return scatter_pred, scatter_err,

    anim = FuncAnimation(fig, update, frames=M, blit=True, )
    return anim


data = pickle.load(open('./../data/data_vis_test.pkl','rb'))[::2]
x_lim = [0,1]
y_lim = [0, 1]

component = 1
v_lim = [data[0][2][:,component].min(), data[0][2][:, component].max()]

x_data = data[0][0]
pred_data = np.stack([data[i][1][:,component] for i in range(len(data))],axis=1)

gt_data = np.stack([data[i][2][:, component] for i in range(len(data))],axis=1)

err = np.abs(pred_data - gt_data)

anim = scatter_save_video(x_data, pred_data, err, x_lim, y_lim, v_lim)

anim.save('ns2d_vis_test.mp4')