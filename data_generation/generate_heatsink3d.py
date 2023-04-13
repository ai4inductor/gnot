
import os
import sys
import pickle
sys.path.extend(['..','../..','../../..'])
# os.environ['PATH'] += ':/home/zhongkai/.local'
os.environ['PATH'] += ':/home/zhongkai/comsol/comsol60/multiphysics/bin'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import mph
# import gnot.data_generation.MPh.mph as mph
import meshio
import networkx as nx
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import jpype
from scipy.spatial import cKDTree
pi = np.pi




parser = argparse.ArgumentParser()

parser.add_argument("--init",default=0,type=int)
parser.add_argument("--end",default=1,type=int)
# parser.add_argument("--np",default=4,type=int)

args = parser.parse_args()



def generate_params(N=1200):
    u = np.random.rand(N,8)
    u[:,0] = 6 + 3*u[:,0]
    u[:,1] = 3 + 2*u[:,1]
    u[:,2] = 1.2+2.8*u[:,2]
    u[:,3] = 0.5 + 1.5*u[:,3]
    u[:,4] = 0.5 + 1.5*u[:,4]
    u[:,5] = 0.5 + 2.5*u[:,5]
    u[:,6] = 3 + 12*u[:,6]
    u[:,7] = 0.5 + 2.5*u[:,7]
    print(np.max(u,axis=0),np.min(u,axis=0))
    np.savetxt('heatsink3d_params.txt',u)



def generate_new_doe(pymodel, u):
    pymodel.parameter('L_channel',str(u[0])+'[cm]')
    pymodel.parameter('W_channel',str(u[1]) + "[cm]")
    pymodel.parameter('H_channel',str(u[2])+'[cm]')
    pymodel.parameter('L_chip',str(u[3])+'[cm]')
    pymodel.parameter('W_chip',str(u[4])+'[cm]')
    pymodel.parameter('H_chip',str(u[5])+'[mm]')
    pymodel.parameter('U0',str(u[6])+'[cm/s]')
    pymodel.parameter('P0',str(u[7])+'[W]')

    return pymodel


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






def generate_heatsink3d_data(init, end):
    time0 = time.time()
    File_Path = os.path.dirname(os.path.abspath(__file__))
    data_path_prefix = File_Path + '/../data/heatsink3d/'
    #### load parameters
    client = mph.start(cores=4)
    pymodel = client.load(File_Path + '/heatsink3d.mph')
    model = pymodel.java
    print('Load model finished {} s'.format(time.time() - time0))

    params = np.loadtxt(File_Path +'/heatsink3d_params.txt')

    # data_random_str = str(np.random.randint(1000))
    # data_path_in = './heatsink_temp/' + 'u_in_' + data_random_str + '.txt'

    ### sequential solver
    # dataset = []
    for i in range(init, end):
        time_i = time.time()

        pymodel = generate_new_doe(pymodel, params[i])

        #### clear cached history
        if i % 5 == 0:
            pymodel.clear()
            pymodel.reset()

        try:
            pymodel.solve()
        except Exception:
            print('Task @ {} failed'.format(i + 1))
            continue
        time_i_solved = time.time()
        pymodel.java.result().export("data1").set("filename", data_path_prefix + "heatsink3d_{}.vtu".format(i + 1))
        pymodel.java.result().export("data1").run()


        print('Task @ {} solved with {}s saved with {}s'.format(i + 1, time_i_solved - time_i,
                                                                time.time() - time_i_solved))


    # pickle.dump(dataset, open('ns2d_1ball_dataset.pkl','wb'))
    print('Total time {}'.format(time.time() - time0))
    return


def merge_data_heatsink(N, with_p=False):
    data_prefix = './../data/'
    ### set random u_p
    # u_p = np.zeros_like(u_p)
    dataset = []

    params = np.loadtxt('./heatsink3d_params.txt')

    for i in range(N):
        time_i = time.time()
        if os.path.exists(data_prefix + 'heatsink3d/heatsink3d_{}.vtu'.format(i+1)):
            data_all = meshio.read(data_prefix + 'heatsink3d/heatsink3d_{}.vtu'.format(i+1))
        else:
            print('Task @ {} failded'.format(i+1))
            continue

        x_all = data_all.points[:,:3]
        T_all, u_all, v_all, w_all, p_all = data_all.point_data['T'], data_all.point_data["u" ], data_all.point_data['v'], data_all.point_data['w'], data_all.point_data['p']
        if with_p:
            y_all = np.stack([T_all, u_all, v_all, w_all, p_all],axis=-1)
        else:
            y_all = np.stack([T_all, u_all, v_all, w_all],axis=-1)
        ### clear NaN values
        y_all[np.isnan(y_all)] = 0

        u_p = params[i]

        n_orig = x_all.shape[0]
        if with_p:
            #### delete duplicate points
            x_all, y_all = del_duplicate_points_data(x_all, y_all, 1e-7)
            n_del = n_orig - x_all.shape[0]
        else:
            n_del = 0


        # dataset.append([x_all, y_all, u_p, (x_b, x_in, x_d1, x_d2, x_d3)])
        dataset.append([x_all, y_all, u_p, None])


        print('task @ {} num nodes {} del points {}/{}  with {}s'.format(i+1, x_all.shape[0], n_del,n_orig, time.time()-time_i))

    print('Total {} samples'.format(len(dataset)))
    pickle.dump(dataset[:int(10/11*len(dataset))], open(data_prefix +'heatsink3d_{}_train.pkl'.format(N),'wb'))
    pickle.dump(dataset[int(10/11*len(dataset)):], open(data_prefix + 'heatsink3d_{}_test.pkl'.format(N), "wb"))
    return



if __name__ == "__main__":
    # generate_params(1100)
    # generate_heatsink3d_data(args.init,args.end)
    merge_data_heatsink(1100)