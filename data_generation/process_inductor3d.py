#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os
import numpy as np
import pickle
from scipy.spatial import cKDTree


def del_duplicate_points_data(X, Y, threshold=1e-9):
    kdtree = cKDTree(X)

    # 查询 k-d 树以找到距离小于阈值的点对
    pairs = kdtree.query_pairs(threshold)

    # 提取要剔除的点的索引
    indices_to_remove = set()
    for pair in pairs:
        indices_to_remove.add(pair[0])
        indices_to_remove.add(pair[1])

    duplicate_ids = list(indices_to_remove)
    indices_to_keep = list(set(range(len(X))) - set(duplicate_ids))

    print(' delete points {} / {}'.format(len(duplicate_ids), len(X)))

    return X[indices_to_keep], Y[indices_to_keep]


def extract_inductor3d():
    folder_path = './../data/inductor3d/'
    param_path = './../data/inductor3d/ParametricSetupResult1.csv'
    params = np.loadtxt(param_path, delimiter=",", skiprows=1)[:, 1:]

    data_list_a = []
    data_list_b = []
    for i in range(len(params)):
        print('Loading @ {} / {}'.format(i, len(params)))
        data = np.load(folder_path + 'Inductor3D{}a.npz'.format(i))

        xa = data['xxa']
        ya = data['yya']
        za = data['zza']
        va = data['vva'][...,None]
        data = np.load(folder_path + 'Inductor3D{}b.npz'.format(i))

        xb = data['xxb']
        yb = data['yyb']
        zb = data['zzb']
        vb = data['vvb'][...,None]

        Xa = np.stack((xa, ya, za), axis=-1)
        Xb = np.stack((xb, yb, zb), axis=-1)


        Xa, ya = del_duplicate_points_data(Xa, va, 1e-9)
        Xb, yb = del_duplicate_points_data(Xb, vb, 1e-9)



        data_list_a.append((Xa, ya,  params[i],  None))
        data_list_b.append((Xb, yb,  params[i], None))

    test_num = int(0.2*len(params))
    train_num = len(params) - test_num
    pickle.dump(data_list_a[:train_num], open('./../data/inductor3d_A1_train.pkl','wb'))
    pickle.dump(data_list_a[train_num:], open('./../data/inductor3d_A1_test.pkl','wb'))
    pickle.dump(data_list_b[:train_num], open('./../data/inductor3d_B1_train.pkl','wb'))
    pickle.dump(data_list_b[train_num:], open('./../data/inductor3d_B1_test.pkl','wb'))




if __name__ == "__main__":
    extract_inductor3d()
