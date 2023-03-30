#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import numpy as np
import pickle
import matplotlib.pyplot as plt


data_test = np.load('./../data/2023_01_30__173916__DoE_2d_inductor_DoE_200test.npz')
data_train = np.load('./../data/2023_01_30__153711__DoE_2d_inductor_DoE_1000train.npz')

params_test = np.loadtxt('./../data/2023_01_30_inductor2d_bosch_test_params.txt',delimiter=",",skiprows=1)
params_train =  np.loadtxt('./../data/2023_01_30_inductor2d_bosch_train_params.txt',delimiter=",",skiprows=1)


def visualize():
    z = data_test['doe_1']

    x, y = z[:,0], z[:,1]
    # x, y = (x-x.mean())/x.std(), (y-y.mean())/y.std()
    # x, y = np.tanh(x), np.tanh(y)
    u = z[:,0]

    cm = plt.get_cmap('rainbow')
    plt.figure()
    plt.scatter(x, y, c=u, s=2, cmap=cm)
    plt.colorbar()
    plt.show()

def process_bosch_inductor_data(data, params, save_path):
    data_list = []
    for i, key in enumerate(data.keys()):
        z = data[key]
        x = z[:,:2]
        y = np.stack([z[:,3],z[:,4],z[:,2]],axis=-1) # Bx, By, Az
        theta = params[i]

        data_list.append([x, y, theta, None])

        print('process @ {} num nodes {}'.format(i, x.shape[0]))

    pickle.dump(data_list, open(save_path,'wb'))








if __name__ == "__main__":
    # visualize()
    process_bosch_inductor_data(data_test, params_test, './../data/inductor2d_bosch_test.pkl') # 174
    process_bosch_inductor_data(data_train, params_train, './../data/inductor2d_bosch_train.pkl')   # 879

