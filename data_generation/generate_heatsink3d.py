
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


def merge_data_heatsink(N):
    data_prefix = './../data/'
    ### set random u_p
    # u_p = np.zeros_like(u_p)
    dataset = []

    for i in range(N):
        time_i = time.time()
        if os.path.exists(data_prefix + 'heatsink/heatsink_{}.vtu'.format(i+1)):
            data_all = meshio.read(data_prefix + 'heatsink/heatsink_{}.vtu'.format(i+1))
            data_in = meshio.read(data_prefix + 'heatsink/heatsink_{}_b.vtu'.format(i+1))
        else:
            print('Task @ {} failded'.format(i+1))
            continue

        x_all = data_all.points[:,:3]
        T_all, u_all, v_all, w_all, p_all = data_all.point_data['temp'], data_all.point_data["ux" ], data_all.point_data['uy'], data_all.point_data['uz'], data_all.point_data['p']
        y_all = np.stack([T_all, u_all, v_all, w_all, p_all],axis=-1)

        ### clear NaN values
        y_all[np.isnan(y_all)] = 0

        x_in = data_in.points[:,:3]
        y_in = data_in.point_data["ux"][...,None]
        x_in = np.concatenate([x_in, y_in],axis=1)


        x_in[np.isnan(x_in)] = 0

        #### comment these two lines for clean dataset
        if x_in.shape[0] == 85:
            print('{} uncleared'.format(i))
            continue
        u_p = np.zeros(1)



        #### buid nx graph
        g = nx.Graph()
        g.add_nodes_from(range(x_all.shape[0]))

        ###### Do Not add edges
        #### get edges, now only supports triangle mesh
        # edges = vtk_data.cells_dict['triangle']
        # g.add_edges_from(edges[:, [0, 1]])
        # g.add_edges_from(edges[:, [0, 2]])
        # g.add_edges_from(edges[:, [1, 2]])
        # g.add_edges_from([[_, _] for _ in range(x_2d.shape[0])])   #### add self loop

        #### current data fields are u, v and p

        #### load boundary data, omit u,v,p, only save boundary coordinates

        # dataset.append([x_all, y_all, u_p, (x_b, x_in, x_d1, x_d2, x_d3)])
        dataset.append([x_all, y_all, u_p, (x_in,)])

        print('task @ {} num nodes {} edges {} with {}s'.format(i+1, x_all.shape[0], g.number_of_edges(), time.time()-time_i))

    print('Total {} samples'.format(len(dataset)))
    pickle.dump(dataset[:int(0.8*len(dataset))], open(data_prefix +'heatsink_{}_train.pkl'.format(N),'wb'))
    pickle.dump(dataset[int(0.8*len(dataset)):], open(data_prefix + 'heatsink_{}_test.pkl'.format(N), "wb"))
    return



if __name__ == "__main__":
    # generate_params(1100)
    generate_heatsink3d_data(args.init,args.end)