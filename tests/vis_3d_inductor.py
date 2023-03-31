#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import matplotlib.pyplot as plt
import numpy as np
import pickle


# data = np.load('OneCase3D_2a.npz')
# x,y,z, v = data['xxa'], data['yya'], data['zza'], data['vva']

# data = np.load('OneCase3D_2b.npz')
# x,y,z, v = data['xxb'], data['yyb'], data['zzb'], data['vvb']


idx =8
data = pickle.load(open('./../data/inductor3d_B1_train.pkl','rb'))[idx]
x, y, z, v = data[0][:,0], data[0][:,1], data[0][:,2], data[1]


def plot_3d_scatter_subplot(ax, x, y, z, v, elev=30, azim=30):
    scatter = ax.scatter(x, y, z, c=v, cmap='viridis', marker='o')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置视角
    ax.view_init(elev=elev, azim=azim)

    return scatter

def plot_3d_scatter_multiple_views(x, y, z, v,):
    fig = plt.figure(figsize=(18, 12))
    angles = [(30, 0), (30, 90), (30, 30), (30, 270), (60, 30), (0, 0)]

    for i, angle in enumerate(angles):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        scatter = plot_3d_scatter_subplot(ax, x, y, z, v, elev=angle[0], azim=angle[1])
        fig.colorbar(scatter)
    # 添加 colorbar
    # fig.colorbar(scatter, ax=fig.get_axes())

    plt.show()


plot_3d_scatter_multiple_views(x, y, z, v)
print(x.shape)


### plot extreme values
idxs = (v>0.9*v.max()).squeeze()
x, y, z, v = x[idxs], y[idxs], z[idxs], v[idxs]
plot_3d_scatter_multiple_views(x, y, z, v)
print(x.shape)