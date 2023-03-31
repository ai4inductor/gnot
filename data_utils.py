#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import os
import torch
import numpy as np
import networkx as nx
import tqdm
import time
import pickle
import gc
import dgl

from sklearn.preprocessing import QuantileTransformer

from dgl.data import DGLDataset
from dgl.nn.pytorch import SumPooling, AvgPooling
from dgl.ops.segment import segment_reduce

from scipy import interpolate
from scipy.io import loadmat
from scipy.sparse import csr_matrix, diags
from torch.utils.data import Dataset
from torch.nn.modules.loss import _WeightedLoss
from torch.nn.utils.rnn import pad_sequence



from gnot.utils import TorchQuantileTransformer, UnitTransformer, PointWiseUnitTransformer, MultipleTensors, MinMaxTransformer
from gnot.models.cgpt import CGPTNO
from gnot.models.mmgpt import GNOT
from gnot.models.MLP import MLPNO, MLPNOScatter, FourierMLP



def get_dataset(args):
    if args.dataset == "ns2d_4ball":
        train_path = './data/ns2d_nomask_4ball_2200_train.pkl'
        test_path = './data/ns2d_nomask_4ball_2200_test.pkl'

    elif args.dataset == "inductor2d":
        # train_path = './data/inductor2d_nomask_1100_train.pkl'
        # test_path = './data/inductor2d_nomask_1100_test.pkl'
        train_path = './data/inductor2d_nodup_1100_train.pkl'
        test_path = './data/inductor2d_nodup_1100_test.pkl'
    elif args.dataset == "inductor2d_b":
        train_path = "./data/inductor2d_bosch_nodup_train.pkl"
        test_path = './data/inductor2d_bosch_nodup_test.pkl'

    elif args.dataset in  ["inductor3d_A1","inductor3d_B1"]:
        if args.dataset == "inductor3d_A1":
            train_path = "./data/inductor3d_A1_train.pkl"
            test_path = "./data/inductor3d_A1_test.pkl"
        elif args.dataset == "inductor3d_B1":
            train_path = "./data/inductor3d_B1_train.pkl"
            test_path = "./data/inductor3d_B1_test.pkl"

    else:
        raise NotImplementedError

    args.train_num = int(args.train_num) if args.train_num not in ['all', 'none'] else args.train_num
    args.test_num = int(args.test_num) if args.test_num not in ['all', 'none'] else args.test_num

    train_dataset = MIODataset(train_path, name=args.dataset, train=True, train_num=args.train_num,sort_data=args.sort_data,
                               normalize_y=args.use_normalizer,
                               normalize_x=args.normalize_x)
    test_dataset = MIODataset(test_path, name=args.dataset, train=False, test_num=args.test_num,sort_data=args.sort_data,
                              normalize_y=args.use_normalizer,
                              normalize_x=args.normalize_x, y_normalizer=train_dataset.y_normalizer,x_normalizer=train_dataset.x_normalizer,up_normalizer=train_dataset.up_normalizer)


    args.dataset_config = train_dataset.config

    return train_dataset, test_dataset


def get_scatter_dataset(args):
    if args.dataset == "ns2d_4ball":
        train_path = './data/ns2d_nomask_4ball_2200_train.pkl'
        test_path = './data/ns2d_nomask_4ball_2200_test.pkl'

    elif args.dataset == "inductor2d":
        # train_path = './data/inductor2d_nomask_1100_train.pkl'
        # test_path = './data/inductor2d_nomask_1100_test.pkl'
        train_path = './data/inductor2d_nodup_1100_train.pkl'
        test_path = './data/inductor2d_nodup_1100_test.pkl'
    elif args.dataset == "inductor2d_b":
        train_path = "./data/inductor2d_bosch_nodup_train.pkl"
        test_path = './data/inductor2d_bosch_nodup_test.pkl'

    elif args.dataset in ["inductor3d_A1", "inductor3d_B1"]:
        if args.dataset == "inductor3d_A1":
            train_path = "./data/inductor3d_A1_train.pkl"
            test_path = "./data/inductor3d_A1_test.pkl"
        elif args.dataset == "inductor3d_B1":
            train_path = "./data/inductor3d_B1_train.pkl"
            test_path = "./data/inductor3d_B1_test.pkl"

    else:
        raise NotImplementedError


    args.train_num = int(args.train_num) if args.train_num not in ['all','none'] else args.train_num
    args.test_num = int(args.test_num) if args.test_num not in ['all','none'] else args.test_num


    train_dataset = MIOScatterDataset(train_path, name=args.dataset, train=True, train_num=args.train_num,merge_all_inputs=args.merge_inputs,
                               normalize_y=args.use_normalizer,
                               normalize_x=args.normalize_x)
    test_dataset = MIOScatterDataset(test_path, name=args.dataset, train=False, test_num=args.test_num,merge_all_inputs=args.merge_inputs,
                              normalize_y=args.use_normalizer,
                              normalize_x=args.normalize_x, y_normalizer=train_dataset.y_normalizer,x_normalizer=train_dataset.x_normalizer,up_normalizer=train_dataset.up_normalizer)


    args.dataset_config = train_dataset.config

    return train_dataset, test_dataset


def get_model(args):
    # if args.dataset[:4] == 'ns2d':
    #
    #     space_dim = 2
    #     g_u_dim = 0
    #     if args.dataset == "ns2d_4ball" or "ns2d_4ball_rd" or 'ns2d_large':
    #         u_p_dim = 12
    #     else:
    #         raise NotImplementedError
    #     out_size = 3 if args.component in ['all','all-reduce'] else 1
    # else:
    #     raise NotImplementedError

    trunk_size, theta_size, output_size = args.dataset_config['input_dim'], args.dataset_config['theta_dim'], args.dataset_config['output_dim']
    output_size = args.dataset_config['output_dim'] if args.component in ['all', 'all_reduce'] else 1

    ### full batch training
    if args.model_name == "CGPT":
        # trunk_size, branch_size, output_size = space_dim + u_p_dim, space_dim + g_u_dim, out_size

        return CGPTNO(trunk_size=trunk_size + theta_size ,branch_sizes=args.branch_sizes, output_size=output_size,n_layers=args.n_layers, n_hidden=args.n_hidden, n_head=args.n_head,attn_type=args.attn_type, ffn_dropout=args.ffn_dropout, attn_dropout=args.attn_dropout, mlp_layers=args.mlp_layers, act=args.act,horiz_fourier_dim=args.hfourier_dim)



    elif args.model_name == "GNOT":

        return GNOT(trunk_size=trunk_size + theta_size,branch_sizes=args.branch_sizes, output_size=output_size,n_layers=args.n_layers, n_hidden=args.n_hidden, n_head=args.n_head,attn_type=args.attn_type, ffn_dropout=args.ffn_dropout, attn_dropout=args.attn_dropout, mlp_layers=args.mlp_layers, act=args.act,horiz_fourier_dim=args.hfourier_dim,space_dim=args.space_dim,n_experts=args.n_experts, n_inner=args.n_inner)


    elif args.model_name == 'MLP':
        return MLPNO(input_size=trunk_size + theta_size,n_hidden=args.n_hidden, output_size=output_size, n_layers=args.n_layers)


    ### scatter training
    elif args.model_name == 'MLP_s':
        return MLPNOScatter(input_size=trunk_size + theta_size,n_hidden=args.n_hidden, output_size=output_size, n_layers=args.n_layers)

    elif args.model_name == "FourierMLP":
        return FourierMLP(space_dim=args.space_dim, theta_dim= theta_size,n_hidden=args.n_hidden, output_size=output_size, n_layers=args.n_layers, fourier_dim=args.hfourier_dim, sigma=args.sigma)



    else:
        raise NotImplementedError

def get_loss_func(name, args, **kwargs):
    if name == 'rel2':
        return WeightedLpRelLoss(p=2,component=args.component, normalizer=kwargs['normalizer'])
    elif name == "rel1":
        return WeightedLpRelLoss(p=1,component=args.component, normalizer=kwargs['normalizer'])
    elif name == 'l2':
        return WeightedLpLoss(p=2, component=args.component, normalizer=kwargs["normalizer" ])
    elif name == "l1":
        return WeightedLpLoss(p=1, component=args.component, normalizer=kwargs["normalizer" ])
    else:
        raise NotImplementedError







'''
    A simple interface for processing FNO dataset,
    1. Data might be 1d, 2d, 3d
    2. X: concat of [pos, a], , we directly reshape them into a B*N*C array
    2. We could use pointwise normalizer since dimension of data is the same
    3. Building graphs for FNO dataset is fast since there is no edge info, we do not use cache
    4. for FNO dataset, we augment g_u = g and set u_p = 0
    
'''
class FNODataset(DGLDataset):
    def __init__(self, X, Y, name=' ',train=True,test=False, normalize_y=False, y_normalizer=None, normalize_x = False):
        self.normalize_y = normalize_y
        self.normalize_x = normalize_x
        self.y_normalizer = y_normalizer

        self.x_data = torch.from_numpy(X)
        self.y_data = torch.from_numpy(Y)


        ####  debug timing


        super(FNODataset, self).__init__(name)   #### invoke super method after read data



    def process(self):

        self.data_len = len(self.x_data)
        self.n_dim = self.x_data.shape[1]
        self.graphs = []
        self.graphs_u = []
        self.u_p = []
        for i in range(len(self)):
            x_t, y_t = self.x_data[i].float(), self.y_data[i].float()
            g = dgl.DGLGraph()
            g.add_nodes(self.n_dim)
            g.ndata['x'] = x_t
            g.ndata['y'] = y_t
            up = torch.zeros([1])
            u = torch.zeros([1])
            u_flag = torch.zeros(g.number_of_nodes(),1)
            g.ndata['u_flag'] = u_flag
            self.graphs.append(g)
            self.u_p.append(up) # global input parameters
            g_u = dgl.DGLGraph()
            g_u.add_nodes(self.n_dim)
            g_u.ndata['x'] = x_t
            g_u.ndata['u'] = torch.zeros(g_u.number_of_nodes(), 1)

            self.graphs_u.append(g_u)

            # print('processing {}'.format(i))

        self.u_p = torch.stack(self.u_p)


        #### normalize_y
        if self.normalize_y:
            self.__normalize_y()
        if self.normalize_x:
            self.__normalize_x()

        return

    def __normalize_y(self):
        if self.y_normalizer is None:

            self.y_normalizer = PointWiseUnitTransformer(self.y_data)
            # print('point wise normalizer shape',self.y_normalizer.mean.shape, self.y_normalizer.std.shape)

            # y_feats_all = torch.cat([g.ndata['y'] for g in self.graphs],dim=0)
            # self.y_normalizer = UnitTransformer(y_feats_all)


        for g in self.graphs:
            g.ndata['y'] = self.y_normalizer.transform(g.ndata["y"], inverse=False)  # a torch quantile transformer

        print('Target features are normalized using pointwise unit normalizer')
        # print('Target features are normalized using unit transformer')

    ### TODO: use train X normalizer since test data is not available
    def __normalize_x(self):
        x_feats_all = torch.cat([g.ndata["x"] for g in self.graphs],dim=0)

        self.x_normalizer = UnitTransformer(x_feats_all)

        # for g in self.graphs:
        #     g.ndata['x'] = self.x_normalizer.transform(g.ndata['x'], inverse=False)

        # if self.graphs_u[0].number_of_nodes() > 0:
        #     for g in self.graphs_u:
        #         g.ndata['x'] = self.x_normalizer.transform(g.ndata['x'], inverse=False)

        self.up_normalizer = UnitTransformer(self.u_p)
        self.u_p = self.up_normalizer.transform(self.u_p, inverse=False)


        print('Input features are normalized using unit transformer')


    def __len__(self):
        return self.data_len


    def __getitem__(self, idx):
        return self.graphs[idx], self.u_p[idx], self.graphs_u[idx]



def collate_op(items):
    transposed = zip(*items)
    batched = []
    for sample in transposed:
        if isinstance(sample[0], dgl.DGLGraph):
            batched.append(dgl.batch(list(sample)))
        elif isinstance(sample[0], torch.Tensor):
            batched.append(torch.stack(sample))
        elif isinstance(sample[0], MultipleTensors):
            sample_ = MultipleTensors([pad_sequence([sample[i][j] for i in range(len(sample))]).permute(1,0,2) for j in range(len(sample[0]))])
            batched.append(sample_)
        else:
            raise NotImplementedError
    return batched




''' 
    Dataset format:
    [X, Y, theta, (f1, f2, ...)], input functions could be None
'''
class MIODataset(DGLDataset):
    def __init__(self, data_path, name=' ',train=True,test=False, train_num=None, test_num=None, use_cache=True, normalize_y=False, y_normalizer=None, x_normalizer=None, up_normalizer=None,normalize_x = False, sort_data=False):
        self.data_path = data_path
        self.cached_path = self.data_path[:-4] + '_' + 'train' + '_cached' +self.data_path[-4:] if train else  self.data_path[:-4] + '_' + 'test' + '_cached' +self.data_path[-4:]
        self.use_cache = use_cache
        self.normalize_y = normalize_y
        self.normalize_x = normalize_x
        self.y_normalizer = y_normalizer
        self.x_normalizer = x_normalizer
        self.up_normalizer = up_normalizer
        self.sort_data = sort_data
        self.num_inputs = 0

        ####  debug timing
        time0 = time.time()
        if not os.path.exists(self.cached_path):
            data_all = pickle.load(open(self.data_path, "rb"))
            print('Load dataset finished {}'.format(time.time()-time0))
            #### initialize dataset
            # processing logic: if both train_num and test_num is None, then divide by 0.8:0.2 using a single file, else use two files
            self.train = train
            if ((train_num == 'none') and (test_num == 'none')):
                self.train_num = int(0.8 * len(data_all))
                self.test_num = len(data_all) - self.train_num
            if self.train:
                if train_num == 'all':   # use all to train
                    self.train_num = len(data_all)
                else:
                    self.train_num = min(train_num, len(data_all))
                    if train_num > len(data_all):
                        print('Warnings: there is no enough train data {} / {}'.format(train_num, len(data_all)))
                self.data_list = data_all[:self.train_num]
                print('Training with {} samples'.format(self.train_num))
            else:
                if test_num == "all":
                    self.test_num = len(data_all)
                else:
                    self.test_num = min(test_num, len(data_all))
                    if test_num > len(data_all):
                        print('Warnings: there is no enough test data {} / {}'.format(test_num, len(data_all)))

                self.data_list = data_all[-self.test_num:]
                print('Testing with {} samples'.format(self.test_num))

        super(MIODataset, self).__init__(name)   #### invoke super method after read data

        # self.__initialize_tensor_dataset()


    def process(self):


        self.data_len = len(self.data_list)
        self.graphs = []
        self.inputs_f = []
        self.u_p = []
        for i in range(len(self)):
            x, y, u_p, input_f = self.data_list[i]
            g = dgl.DGLGraph()
            g.add_nodes(x.shape[0])
            g.ndata['x'] = torch.from_numpy(x).float()
            g.ndata['y'] = torch.from_numpy(y).float()
            up = torch.from_numpy(u_p).float()
            self.graphs.append(g)
            self.u_p.append(up) # global input parameters
            if input_f is not None:
                input_f = MultipleTensors([torch.from_numpy(f).float() for f in input_f])
                self.inputs_f.append(input_f)
                self.num_inputs = len(input_f)

        if len(self.inputs_f) == 0:
            self.inputs_f = torch.zeros([len(self)])  # pad values, tensor of 0, not list

            # print('processing {}'.format(i))d

        #### sort data if necessary
        if self.sort_data:
            self.__sort_dataset()

        self.u_p = torch.stack(self.u_p)


        #### normalize_y
        if self.normalize_y != 'none':
            self.__normalize_y()
        if self.normalize_x != 'none':
            self.__normalize_x()

        self.__update_dataset_config()

        return

    def __sort_dataset(self):
        zipped_lists = list(zip(self.graphs, self.u_p, self.inputs_f))
        sorted_lists = sorted(zipped_lists, key=lambda x: x[0].number_of_nodes(),reverse=True)

        self.graphs, self.u_p, self.inputs_f = zip(*sorted_lists)
        print('Dataset sorted by number of nodes')
        return


    def __normalize_y(self):
        if self.y_normalizer is None:
            y_feats_all = torch.cat([g.ndata['y'] for g in self.graphs],dim=0)
            if self.normalize_y == 'unit':
                self.y_normalizer = UnitTransformer(y_feats_all)
                print('Target features are normalized using unit transformer')
                print(self.y_normalizer.mean, self.y_normalizer.std)


            elif self.normalize_y == 'minmax':
                self.y_normalizer = MinMaxTransformer(y_feats_all)
                print('Target features are normalized using unit transformer')
                print(self.y_normalizer.max, self.y_normalizer.min)

            elif self.normalize_y == 'quantile':
                self.y_normalizer = QuantileTransformer(output_distribution='normal')
                self.y_normalizer = self.y_normalizer.fit(y_feats_all)
                self.y_normalizer = TorchQuantileTransformer(self.y_normalizer.output_distribution, self.y_normalizer.references_,self.y_normalizer.quantiles_)
                print('Target features are normalized using quantile transformer')


        for g in self.graphs:
            g.ndata['y'] = self.y_normalizer.transform(g.ndata["y"], inverse=False)  # a torch quantile transformer



    def __normalize_x(self):
        if self.x_normalizer is None:
            x_feats_all = torch.cat([g.ndata["x"] for g in self.graphs],dim=0)
            if self.normalize_x == 'unit':
                self.x_normalizer = UnitTransformer(x_feats_all)
                self.up_normalizer = UnitTransformer(self.u_p)

            elif self.normalize_x == 'minmax':
                self.x_normalizer = MinMaxTransformer(x_feats_all)
                self.up_normalizer = MinMaxTransformer(self.u_p)

            else:
                raise NotImplementedError


        for g in self.graphs:
            g.ndata['x'] = self.x_normalizer.transform(g.ndata['x'], inverse=False)
        self.u_p = self.up_normalizer.transform(self.u_p, inverse=False)

        print('Input features are normalized using unit transformer')


    def __update_dataset_config(self):
        self.config = {
            'input_dim': self.graphs[0].ndata['x'].shape[1],
            'theta_dim': self.u_p.shape[1],
            'output_dim': self.graphs[0].ndata['y'].shape[1],
            'branch_sizes': [x.shape[1] for x in self.inputs_f[0]] if isinstance(self.inputs_f, list) else 0

        }
        return


    def __len__(self):
        return self.data_len


    def __getitem__(self, idx):
        return self.graphs[idx], self.u_p[idx], self.inputs_f[idx]






class MIOScatterDataset(DGLDataset):
    def __init__(self, data_path, name=' ',train=True,test=False, train_num=None, test_num=None, use_cache=True, normalize_y=False, y_normalizer=None, normalize_x = False, merge_all_inputs=True, x_normalizer=None, up_normalizer=None):
        self.data_path = data_path
        self.cached_path = self.data_path[:-4] + '_' + 'train' + '_cached' +self.data_path[-4:] if train else  self.data_path[:-4] + '_' + 'test' + '_cached' +self.data_path[-4:]
        self.use_cache = use_cache
        self.train = train
        self.normalize_y = normalize_y
        self.normalize_x = normalize_x
        self.y_normalizer = y_normalizer
        self.x_normalizer = x_normalizer
        self.up_normalizer = up_normalizer
        self.num_inputs = 0
        self.theta_dim = 0
        self.branch_sizes = []
        self.len_inputs = []
        self.data_indexes = [0,]
        self.merge_inputs = merge_all_inputs


        ####  debug timing
        time0 = time.time()
        if not os.path.exists(self.cached_path):
            data_all = pickle.load(open(self.data_path, "rb"))
            print('Load dataset finished {}'.format(time.time()-time0))
            #### initialize dataset
            if ((train_num == 'none') and (test_num == 'none')):
                self.train_num = int(0.8 * len(data_all))
                self.test_num = len(data_all) - self.train_num
            if self.train:
                if train_num == 'all':   # use all to train
                    self.train_num = len(data_all)
                else:
                    self.train_num = min(train_num, len(data_all))
                    if train_num > len(data_all):
                        print('Warnings: there is no enough train data {} / {}'.format(train_num, len(data_all)))
                self.data_list = data_all[:self.train_num]
                print('Training with {} samples'.format(self.train_num))
            else:
                if test_num == "all":
                    self.test_num = len(data_all)
                else:
                    self.test_num = min(test_num, len(data_all))
                    if test_num > len(data_all):
                        print('Warnings: there is no enough test data {} / {}'.format(test_num, len(data_all)))

                self.data_list = data_all[-self.test_num:]
                print('Testing with {} samples'.format(self.test_num))

        super(MIOScatterDataset, self).__init__(name)   #### invoke super method after read data

        # self.__initialize_tensor_dataset()


    def process(self):


        self.len_graphs = len(self.data_list)
        self.graphs = []
        self.inputs_f = []
        self.u_p = []
        for i in range(self.len_graphs):
            x, y, u_p, input_f = self.data_list[i]
            g = dgl.DGLGraph()
            g.add_nodes(x.shape[0])
            g.ndata['x'] = torch.from_numpy(x).float()
            g.ndata['y'] = torch.from_numpy(y).float()
            up = torch.from_numpy(u_p).float()
            self.graphs.append(g)
            self.u_p.append(up) # global input parameters
            self.len_inputs.append(x.shape[0])
            self.data_indexes.append(self.data_indexes[-1] + x.shape[0])
            if input_f is not None:
                input_f = MultipleTensors([torch.from_numpy(f).float() for f in input_f])
                self.inputs_f.append(input_f)
                self.num_inputs = len(input_f)

        if len(self.inputs_f) == 0:
            self.inputs_f = torch.zeros([len(self.graphs)])  # pad values, tensor of 0, not list
            # print('processing {}'.format(i))d



        self.data_indexes = torch.LongTensor(self.data_indexes)
        self.len_inputs = torch.LongTensor(self.len_inputs)
        self.u_p = torch.stack(self.u_p)


        #### normalize_y
        if self.normalize_y != 'none':
            self.__normalize_y()
        if self.normalize_x != 'none':
            self.__normalize_x()


        self.__scatter_dataset()

        self.__update_dataset_config()


        return

    def __scatter_dataset(self):
        #### check shape first
        print('preparing scatter dataset')
        f_sizes = [x.numel() for x in self.inputs_f]
        if self.merge_inputs and len(set(f_sizes)) > 1 :
            raise ValueError("Input functions have different lengths")


        self.X_data = []
        self.Y_data = []
        self.theta = []
        for i in range(self.len_graphs):
            X_i, Y_i = self.graphs[i].ndata['x'], self.graphs[i].ndata['y']
            theta_i = torch.tile(self.u_p[i].unsqueeze(0), [X_i.shape[0], 1])
            if self.merge_inputs:
                f_i = torch.cat([x_.view(-1) for x_ in self.inputs_f[i]], dim=0)
                theta_i = torch.cat([theta_i, f_i], dim=0)

            self.X_data.append(X_i)
            self.Y_data.append(Y_i)
            self.theta.append(theta_i)

        self.theta_dim = self.theta[0].shape[1]

        self.X_data, self.Y_data, self.theta = torch.cat(self.X_data, dim=0), torch.cat(self.Y_data, dim=0), torch.cat(self.theta, dim=0)
        self.X_data, self.Y_data, self.theta = self.X_data.contiguous(), self.Y_data.contiguous(), self.theta.contiguous()
        return



    def __normalize_y(self):
        if self.y_normalizer is None:
            y_feats_all = torch.cat([g.ndata['y'] for g in self.graphs],dim=0)
            if self.normalize_y == 'unit':
                self.y_normalizer = UnitTransformer(y_feats_all)
                print('Target features are normalized using unit transformer')
                print(self.y_normalizer.mean, self.y_normalizer.std)


            elif self.normalize_y == 'minmax':
                self.y_normalizer = MinMaxTransformer(y_feats_all)
                print('Target features are normalized using unit transformer')
                print(self.y_normalizer.max, self.y_normalizer.min)

            elif self.normalize_y == 'quantile':
                self.y_normalizer = QuantileTransformer(output_distribution='normal')
                self.y_normalizer = self.y_normalizer.fit(y_feats_all)
                self.y_normalizer = TorchQuantileTransformer(self.y_normalizer.output_distribution, self.y_normalizer.references_,self.y_normalizer.quantiles_)
                print('Target features are normalized using quantile transformer')

        for g in self.graphs:
            g.ndata['y'] = self.y_normalizer.transform(g.ndata["y"], inverse=False)  # a torch quantile transformer



    def __normalize_x(self):
        if self.x_normalizer is None:
            x_feats_all = torch.cat([g.ndata["x"] for g in self.graphs],dim=0)
            if self.normalize_x == 'unit':
                self.x_normalizer = UnitTransformer(x_feats_all)
                self.up_normalizer = UnitTransformer(self.u_p)

            elif self.normalize_x == 'minmax':
                self.x_normalizer = MinMaxTransformer(x_feats_all)
                self.up_normalizer = MinMaxTransformer(self.u_p)

            else:
                raise NotImplementedError


        for g in self.graphs:
            g.ndata['x'] = self.x_normalizer.transform(g.ndata['x'], inverse=False)
        self.u_p = self.up_normalizer.transform(self.u_p, inverse=False)

        print('Input features are normalized using unit transformer')

    #### summary at least these information:
    def __update_dataset_config(self):
        self.config = {'input_dim': self.X_data.shape[1],
                       'theta_dim': self.theta_dim,
                       'output_dim': self.Y_data.shape[1],
                       'branch_sizes': [x.shape[1] for x in self.inputs_f[0]] if isinstance(self.inputs_f, list) else 0
                       }
        print('dataset config: {}'.format(self.config))
        return


    def __len__(self):
        return self.X_data.shape[0]


    def __getitem__(self, idx):
        return self.X_data[idx], self.theta[idx], self.Y_data[idx]



class IdxDataset(torch.utils.data.Dataset):
    def __init__(self, data_len, batch_size, drop_last=False,shuffle=True):
        random_idxs = torch.randperm(data_len) if shuffle else torch.arange(data_len)
        data_idx = [random_idxs[torch.arange(i, min(i+batch_size, data_len))] for i in range(0, data_len, batch_size)]
        if drop_last:
            data_idx = data_idx[:-1]
        self.total_data_len = (data_len // batch_size)*batch_size if drop_last else data_len
        self.data_idx = data_idx
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def reset(self):

        random_idxs = torch.randperm(self.total_data_len) if self.shuffle else torch.arange(self.total_data_len)
        data_idx = [random_idxs[torch.arange(i, min(i + self.batch_size, self.total_data_len))] for i in range(0, self.total_data_len, self.batch_size)]

        self.data_idx = data_idx



    def __getitem__(self, item):
        return self.data_idx[item]


    def __len__(self):
        return len(self.data_idx)






class MIODataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1,sort_data=True, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):
        super(MIODataLoader, self).__init__(dataset=dataset, batch_size=batch_size,
                                           shuffle=shuffle, sampler=sampler,
                                           batch_sampler=batch_sampler,
                                           num_workers=num_workers,
                                           collate_fn=collate_fn,
                                           pin_memory=pin_memory,
                                           drop_last=drop_last, timeout=timeout,
                                           worker_init_fn=worker_init_fn)

        self.sort_data = sort_data
        if sort_data:
            self.batch_indices = [list(range(i, min(i+batch_size, len(dataset)))) for i in range(0, len(dataset), batch_size)]
            if drop_last:
                self.batch_indices = self.batch_indices[:-1]
        else:
            self.batch_indices = list(range(0, (len(dataset) // batch_size)*batch_size)) if drop_last else list(range(0, len(dataset)))
        if shuffle:
            np.random.shuffle(self.batch_indices)







    def __iter__(self):
        # 返回一个迭代器，用于遍历数据集中的每个批次
        for indices in self.batch_indices:
            transposed = zip(*[self.dataset[idx] for idx in indices])
            batched = []
            for sample in transposed:
                if isinstance(sample[0], dgl.DGLGraph):
                    batched.append(dgl.batch(list(sample)))
                elif isinstance(sample[0], torch.Tensor):
                    batched.append(torch.stack(sample))
                elif isinstance(sample[0], MultipleTensors):
                    sample_ = MultipleTensors(
                        [pad_sequence([sample[i][j] for i in range(len(sample))]).permute(1, 0, 2) for j in range(len(sample[0]))])
                    batched.append(sample_)
                else:
                    raise NotImplementedError
            yield batched

    def __len__(self):
        # 返回数据集的批次数
        return len(self.batch_indices)






'''
    A simple warped class for Lp loss in scatter training, supports L1, L2
    Note that L2 loss is MSE loss
'''
class ScatterLpLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, component=0, regularizer=False, normalizer=None):
        super(ScatterLpLoss, self).__init__()

        self.d = d
        self.p = p
        self.component = component if component in ['all' , 'all-reduce'] else int(component)

        self.regularizer = regularizer
        self.normalizer = normalizer
        self.avg_pool = AvgPooling()

    def _lp_losses(self, pred, target, offsets=None):
        if offsets is None:
            if self.component == 'all':
                losses =  ((pred - target).abs() ** self.p)
                loss = losses.mean() ** (1 / self.p)
                metrics = (losses.mean(dim=0) ** (1 /self.p)).clone().detach().cpu().numpy()

            else:
                assert self.component <= target.shape[1]
                losses = (pred - target[:, self.component: self.component + 1]).abs() ** self.p
                loss = losses.mean() ** (1 / self.p)
                metrics = (losses.mean() ** (1 / self.p)).clone().detach().cpu().numpy()
        else:
            if self.component == 'all':

                losses = segment_reduce(offsets, ((pred - target).abs() ** self.p),'mean') ** (1/ self.p)
                loss = losses.mean()
                metrics = losses.mean(dim=0).clone().detach().cpu().numpy()

            else:
                assert self.component <= target.shape[1]
                losses = segment_reduce(offsets,  (pred - target[:, self.component:self.component +1]).abs() ** self.p, 'mean') ** (1 / self.p)
                loss = losses.mean()
                metrics = losses.mean().clone().detach().cpu().numpy()


        return loss, metrics



    def forward(self, pred, target, offsets=None):

        #### only for computing metrics

        loss, metrics = self._lp_losses(pred, target, offsets=offsets)

        if self.normalizer is not None:
            ori_pred, ori_target = self.normalizer.transform(pred, component=self.component,inverse=True), self.normalizer.transform(target,inverse=True)
            _, metrics = self._lp_losses(ori_pred, ori_target, offsets=offsets)

        if self.regularizer:
            raise NotImplementedError
        else:
            reg = torch.zeros_like(loss)

        return loss, reg, metrics



class ScatterLpRelLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, component=0, regularizer=False, normalizer=None):
        super(ScatterLpRelLoss, self).__init__()

        self.d = d
        self.p = p
        self.component = component if component in ['all' , 'all-reduce'] else int(component)

        self.regularizer = regularizer
        self.normalizer = normalizer
        self.avg_pool = AvgPooling()

    def _lp_losses(self, pred, target, offsets=None):
        if offsets is None:     ## single batch case
            if self.component in ['all','all-reduce']:
                err =  ((pred - target).abs() ** self.p)
                losses = (err.sum(dim=0)/ (target.abs() ** self.p).sum(dim=0)) ** (1 / self.p)
                loss = losses.mean()
                if self.component == 'all':
                    metrics = losses.clone().detach().cpu().numpy()
                else:
                    metrics = loss.clone().detach().cpu().numpy()

            else:
                assert self.component <= target.shape[1]
                err = (pred - target[:, self.component: self.component + 1]).abs() ** self.p
                losses = (err.sum(dim=0) / (target[:, self.component : self.component + 1].abs() ** self.p).sum(dim=0)) ** (1 / self.p)
                loss = losses.mean()
                metrics = loss.clone().detach().cpu().numpy()
        else:
            if self.component in ['all', 'all-reduce']:
                err_pool = segment_reduce(offsets, ((pred - target).abs() ** self.p),'sum') ** (1/ self.p)
                target_pool = segment_reduce(offsets, (target.abs()**self.p),'sum')**(1/self.p)
                losses = (err_pool / target_pool)**(1/self.p)
                loss = losses.mean()
                if self.component == 'all':
                    metrics = losses.mean(dim=0).clone().detach().cpu().numpy()
                else:
                    metrics = losses.mean().clone().detach().cpu().numpy()


            else:
                assert self.component <= target.shape[1]
                err_pool = segment_reduce(offsets, ((pred - target).abs() ** self.p), 'sum') ** (1 / self.p)
                target_pool = segment_reduce(offsets, (target.abs() ** self.p), 'sum') ** (1 / self.p)
                losses = (err_pool / target_pool) ** (1 / self.p)
                loss = losses.mean()
                metrics = losses.mean().clone().detach().cpu().numpy()


        return loss, metrics



    def forward(self, pred, target, offsets=None):

        #### only for computing metrics

        loss, metrics = self._lp_losses(pred, target, offsets=offsets)

        if self.normalizer is not None:
            ori_pred = self.normalizer.transform(pred, component=self.component,inverse=True) #, self.normalizer.transform(target,inverse=True)
            ori_target = self.normalizer.transform(target, inverse=True)
            _, metrics = self._lp_losses(ori_pred, ori_target, offsets=offsets)
        else:
            loss, metrics = self._lp_losses(pred, target, offsets=offsets)

        if self.regularizer:
            raise NotImplementedError
        else:
            reg = torch.zeros_like(loss)

        return loss, reg, metrics




class WeightedLpRelLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, component=0,regularizer=False, normalizer=None):
        super(WeightedLpRelLoss, self).__init__()

        self.d = d
        self.p = p
        self.component = component if component in ['all' , 'all-reduce'] else int(component)
        self.regularizer = regularizer
        self.normalizer = normalizer
        self.sum_pool = SumPooling()

    ### all reduce is used in temporal cases, use only one metric for all components
    def _lp_losses(self, g, pred, target):
        if (self.component == 'all') or (self.component == 'all-reduce'):
            err_pool = (self.sum_pool(g, (pred - target).abs() ** self.p))
            target_pool = (self.sum_pool(g, target.abs() ** self.p))
            losses = (err_pool / target_pool)**(1/ self.p)
            if self.component == 'all':
                metrics = losses.mean(dim=0).clone().detach().cpu().numpy()
            else:
                metrics = losses.mean().clone().detach().cpu().numpy()

        else:
            assert self.component <= target.shape[1]
            err_pool = (self.sum_pool(g, (pred - target[:,self.component]).abs() ** self.p))
            target_pool = (self.sum_pool(g, target[:,self.component].abs() ** self.p))
            losses = (err_pool / target_pool)**(1/ self.p)
            metrics = losses.mean().clone().detach().cpu().numpy()

        loss = losses.mean()

        return loss, metrics

    def forward(self, g,  pred, target):

        #### only for computing metrics


        loss, metrics = self._lp_losses(g, pred, target)

        if self.normalizer is not None:
            ori_pred, ori_target = self.normalizer.transform(pred,component=self.component,inverse=True), self.normalizer.transform(target, inverse=True)
            _ , metrics = self._lp_losses(g, ori_pred, ori_target)

        if self.regularizer:
            raise NotImplementedError
        else:
            reg = torch.zeros_like(loss)


        return loss, reg, metrics


class WeightedLpLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, component=0, regularizer=False, normalizer=None):
        super(WeightedLpLoss, self).__init__()

        self.d = d
        self.p = p
        self.component = component if component in ['all' , 'all-reduce'] else int(component)

        self.regularizer = regularizer
        self.normalizer = normalizer
        self.avg_pool = AvgPooling()

    def _lp_losses(self, g, pred, target):
        if self.component == 'all':
            losses = self.avg_pool(g, ((pred - target).abs() ** self.p)) ** (1 / self.p)
            metrics = losses.mean(dim=0).clone().detach().cpu().numpy()

        else:
            assert self.component <= target.shape[1]
            losses = self.avg_pool(g, (pred - target[:, self.component]).abs() ** self.p) ** (1 / self.p)
            metrics = losses.mean().clone().detach().cpu().numpy()

        loss = losses.mean()

        return loss, metrics

    def forward(self, g, pred, target):

        #### only for computing metrics

        loss, metrics = self._lp_losses(g, pred, target)

        if self.normalizer is not None:
            ori_pred, ori_target = self.normalizer.transform(pred,component=self.component, inverse=True), self.normalizer.transform(target, inverse=True)
            _, metrics = self._lp_losses(g, ori_pred, ori_target)

        if self.regularizer:
            raise NotImplementedError
        else:
            reg = torch.zeros_like(loss)

        return loss, reg, metrics


#
#
# '''
#     Simple Mesh FEM dataset class, data should be a list of dict containing the following keys (could be None)
#         x       :   spatial location of points
#         y       :   target physical quantities
#         g       :   nx.Graph with edges
#         u_p     :   input parameter vector
#         u       :   input parameter function, if u_nodes is None, shape should be the same with x
#         u_flag  :   u function defined on a sub-mesh of g
#         u_nodes :   spatial location of parameter functions defined, length should be the same with u
#         edge    :   edges for building graphs, TBD
#
#     use_cache   :  use cached dgl dataset
#     normalize_y :  use quantile transformer for processing data
# '''
#
#
# class SimpleDataset(DGLDataset):
#     def __init__(self, data_path, name=' ', train=True, test=False, train_num=None, test_num=None, use_cache=True,
#                  normalize_y=False, y_normalizer=None, normalize_x=False):
#         self.data_path = data_path
#         self.cached_path = self.data_path[:-4] + '_' + 'train' + '_cached' + self.data_path[
#                                                                              -4:] if train else self.data_path[
#                                                                                                 :-4] + '_' + 'test' + '_cached' + self.data_path[
#                                                                                                                                   -4:]
#         self.use_cache = use_cache
#         self.normalize_y = normalize_y
#         self.normalize_x = normalize_x
#         self.y_normalizer = y_normalizer
#
#         ####  debug timing
#         time0 = time.time()
#         if not os.path.exists(self.cached_path):
#             data_all = pickle.load(open(self.data_path, "rb"))
#             print('Load dataset finished {}'.format(time.time() - time0))
#             #### initialize dataset
#             self.train = train
#             if (train_num is None) or (train_num >= len(data_all)):
#                 self.train_num = int(0.8 * len(data_all))
#                 self.test_num = len(data_all) - self.train_num
#             else:
#                 self.train_num = train_num
#                 self.test_num = test_num
#
#             if self.train:
#                 self.data_list = data_all[:self.train_num]
#             else:
#                 self.data_list = data_all[-self.test_num:] if (self.test_num is not None) else data_all[train_num:]
#
#         super(SimpleDataset, self).__init__(name)  #### invoke super method after read data
#
#         # self.__initialize_tensor_dataset()
#
#     def process(self):
#         if self.use_cache and (os.path.exists(self.cached_path)):
#             self.graphs, self.graphs_u, self.u_p = pickle.load(open(self.cached_path, 'rb'))
#             self.data_len = len(self.graphs)
#         else:
#             self.data_len = len(self.data_list)
#             self.graphs = []
#             self.graphs_u = []
#             self.u_p = []
#             for i in range(len(self)):
#                 x, y, nx_g, u_p, u, u_flag, u_nodes, u_edges = self.data_list[i]
#                 g = dgl.from_networkx(nx_g)
#                 g.ndata['x'] = torch.from_numpy(x).float()
#                 g.ndata['y'] = torch.from_numpy(y).float()
#                 up = torch.from_numpy(u_p).float()
#                 u = torch.zeros([1]) if u is None else torch.from_numpy(u).float()
#                 u_flag = torch.zeros(g.number_of_nodes(), 1) if u_flag is None else torch.from_numpy(u_flag).long()
#                 g.ndata['u_flag'] = u_flag
#                 self.graphs.append(g)
#                 self.u_p.append(up)  # global input parameters
#                 # print(u_nodes.shape)
#                 if u_nodes is not None:  #### build dgl graph for parameter function
#                     g_u = dgl.DGLGraph()
#                     g_u.add_nodes(u_nodes.shape[0])
#                     g_u.add_edges(u_edges) if u_edges is not None else g_u.add_edges([], [])
#                     g_u = dgl.to_bidirected(g_u)
#                     g_u = dgl.add_self_loop(g_u)
#                     g_u.ndata['x'] = torch.from_numpy(
#                         u_nodes).float()  ####TODO: check the order of this and to bidirectional graph
#                     g_u.ndata['u'] = u
#                 else:
#                     g_u = dgl.DGLGraph()
#                 # print(g_u.ndata['x'].shape)
#                 self.graphs_u.append(g_u)
#
#                 print('processing {}'.format(i))
#
#             self.u_p = torch.stack(self.u_p)
#
#             if self.use_cache:
#                 pickle.dump((self.graphs, self.graphs_u, self.u_p), open(self.cached_path, "wb"))
#                 print('cached dataset saved at {}'.format(self.cached_path))
#
#         #### normalize_y
#         if self.normalize_y:
#             self.__normalize_y()
#         if self.normalize_x:
#             self.__normalize_x()
#
#         return
#
#     def __normalize_y(self):
#         if self.y_normalizer is None:
#             y_feats_all = torch.cat([g.ndata['y'] for g in self.graphs], dim=0)
#             # self.y_normalizer = QuantileTransformer(output_distribution='normal')
#             # self.y_normalizer = self.y_normalizer.fit(y_feats_all)
#             # self.y_normalizer = TorchQuantileTransformer(self.y_normalizer.output_distribution, self.y_normalizer.references_,self.y_normalizer.quantiles_)
#             self.y_normalizer = UnitTransformer(y_feats_all)
#             print(self.y_normalizer.mean, self.y_normalizer.std)
#
#         for g in self.graphs:
#             g.ndata['y'] = self.y_normalizer.transform(g.ndata["y"], inverse=False)  # a torch quantile transformer
#
#         # print('Target features are normalized using quantile transformer')
#         print('Target features are normalized using unit transformer')
#
#     ### TODO: use train X normalizer since test data is not available
#     def __normalize_x(self):
#         x_feats_all = torch.cat([g.ndata["x"] for g in self.graphs], dim=0)
#
#         self.x_normalizer = UnitTransformer(x_feats_all)
#
#         # for g in self.graphs:
#         #     g.ndata['x'] = self.x_normalizer.transform(g.ndata['x'], inverse=False)
#
#         # if self.graphs_u[0].number_of_nodes() > 0:
#         #     for g in self.graphs_u:
#         #         g.ndata['x'] = self.x_normalizer.transform(g.ndata['x'], inverse=False)
#
#         self.up_normalizer = UnitTransformer(self.u_p)
#         self.u_p = self.up_normalizer.transform(self.u_p, inverse=False)
#
#         print('Input features are normalized using unit transformer')
#
#     def __len__(self):
#         return self.data_len
#
#     def __getitem__(self, idx):
#         return self.graphs[idx], self.u_p[idx], self.graphs_u[idx]

