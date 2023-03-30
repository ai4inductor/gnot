import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
from scipy.spatial import cKDTree

import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from gnot.models.MLP import MLP, FourierMLP
from gnot.utils import UnitTransformer,MinMaxTransformer, timing

from gnot.tests.siren_demo import Siren

# data_path = './../data/inductor2d_nomask_1100_test.pkl'
# data_path = './../data/ns2d_nomask_4ball_2200_test.pkl'
data_path = './../data/inductor2d_bosch_train.pkl'
dataset = pickle.load(open(data_path,'rb'))

idx = 18
component = 1
X, Y, theta, _ = dataset[idx]
Y = Y[:,component]
# X = torch.from_numpy(X).float()
# Y = torch.from_numpy(Y[:,component:component + 1]).float()


def find_duplicate_or_close_points(points, threshold=1e-5):
    n = points.shape[0]

    # 计算所有点对之间的距离
    dist_matrix = np.sqrt(np.sum((points[:, np.newaxis] - points[np.newaxis, :]) ** 2, axis=2))

    # 将对角线上的值设为一个很大的数，以避免错误地认为同一个点是重复的
    np.fill_diagonal(dist_matrix, np.inf)

    # 寻找距离小于等于阈值的点对
    duplicate_or_close_points = np.argwhere(dist_matrix <= threshold)

    return duplicate_or_close_points


@timing
def fast_duplicate_or_close_points(points, threshold=1e-5):
    kdtree = cKDTree(points)

    # 查询 k-d 树以找到距离小于阈值的点对
    pairs = kdtree.query_pairs(threshold)

    # 提取要剔除的点的索引
    indices_to_remove = set()
    for pair in pairs:
        indices_to_remove.add(pair[0])
        indices_to_remove.add(pair[1])

    return indices_to_remove

result = find_duplicate_or_close_points(X, 1e-5)
# result = list(fast_duplicate_or_close_points(X, 1e-5))
# result = np.stack([result,result],axis=0).T

idxs1 = result[:,0]
idxs2 = result[:,1]
print(result.shape, X.shape[0])

cm = plt.cm.get_cmap('rainbow')

plt.figure()
plt.scatter(X[idxs2,0],X[idxs2,1],s=20, c=Y[idxs2],cmap=cm)
plt.scatter(X[idxs1,0],X[idxs1,1],s=5,c=Y[idxs1],cmap=cm)
plt.colorbar()
plt.show()
