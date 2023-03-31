#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import pickle
import numpy as np

# data = pickle.load(open('./../data/inductor3d_A1_train.pkl','rb'))
data = pickle.load(open('./../data/inductor3d_A1_test.pkl','rb'))


def rel_with_mean_pred(data):
    return np.array([((data[i][1]-data[i][1].mean())**2).sum()**0.5/((data[i][1]**2).sum())**0.5 for i in range(len(data))])

def mean_stats(data):
    return np.array([y[1].mean() for y in data])


err_with_mean = rel_with_mean_pred(data)
print(err_with_mean)
print(err_with_mean.mean())

means = mean_stats(data)
print(means, means.mean())
