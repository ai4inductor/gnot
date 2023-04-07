#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import os
import sys
sys.path.append('./../')
import numpy as np
import torch
import time
import pickle

from scipy.spatial import cKDTree


def find_duplicate_or_close_points(points, threshold=1e-5,device='cpu'):
    n = points.shape[0]
    if device != 'cpu':
        # 计算所有点对之间的距离
        dist_matrix = np.sqrt(np.sum((points[:, np.newaxis] - points[np.newaxis, :]) ** 2, axis=2))

        # 将对角线上的值设为一个很大的数，以避免错误地认为同一个点是重复的
        np.fill_diagonal(dist_matrix, np.inf)
        duplicate_or_close_points = np.argwhere(dist_matrix <= threshold)

    else:
        device = torch.device(device)
        points = torch.tensor(points).to(device)

        dist_matrix = torch.sqrt(torch.sum((points[:,None] - points[None, :])**2,dim=2))
        dist_matrix[:,:] += torch.diag_embed(torch.full((n,),float('inf')))

        duplicate_or_close_points = np.argwhere(dist_matrix.numpy() <= threshold)

        duplicate_or_close_points = duplicate_or_close_points

    return duplicate_or_close_points



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

def del_duplicate_data(path,device):
    data = pickle.load(open(path, 'rb'))

    for i in range(len(data)):
        time0 = time.time()
        X, Y, theta, inputs = data[i]
        # duplicate_ids = find_duplicate_or_close_points(X, 1e-4)[:,0]
        duplicate_ids = fast_find_duplicate_or_close_points(X, 1e-4)
        indices_to_keep = list(set(range(len(X))) - set(duplicate_ids))


        print('process @ {} time {} delete points {} / {}'.format(i, time.time()-time0 ,len(duplicate_ids), len(X)))

        X, Y = X[indices_to_keep], Y[indices_to_keep]

        data[i][0], data[i][1] = X, Y

    return data


def del_duplicate_inductor2d(device):
    train_path ='./../data/inductor2d_nomask_1100_train.pkl'
    test_path = './../data/inductor2d_nomask_1100_test.pkl'
    train_save_path = './../data/inductor2d_nodup_1100_train.pkl'
    test_save_path = './../data/inductor2d_nodup_1100_test.pkl'

    data_train_del = del_duplicate_data(train_path,device)
    data_test_del = del_duplicate_data(test_path,device)


    pickle.dump(data_train_del,open(train_save_path,'wb'))
    pickle.dump(data_test_del, open(test_save_path, "wb"))


def del_duplicate_inductor2d_bosch():
    train_path ='./../data/inductor2d_bosch_train.pkl'  ### moved to gnot released version
    test_path = './../data/inductor2d_bosch_test.pkl'
    train_save_path = './../data/inductor2d_bosch_nodup_train.pkl'
    test_save_path = './../data/inductor2d_bosch_nodup_test.pkl'
    data_train_del = del_duplicate_data(train_path,device='cpu')
    data_test_del = del_duplicate_data(test_path, device="cpu")

    pickle.dump(data_train_del, open(train_save_path, 'wb'))
    pickle.dump(data_test_del, open(test_save_path, "wb"))

if __name__ == "__main__":
    # del_duplicate_inductor2d('cuda:7')
    del_duplicate_inductor2d_bosch()







