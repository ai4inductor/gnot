#!/usr/bin/env python
#-*- coding:utf-8 _*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import dgl


'''
    A simple MLP class, includes at least 2 layers and n hidden layers
'''
class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu'):
        super(MLP, self).__init__()

        if act == 'gelu':
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.linear_pre = nn.Linear(n_input, n_hidden)
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(n_layers)])

        # self.bns = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_layers)])



    def forward(self, x):
        x = self.act(self.linear_pre(x))
        for i in range(self.n_layers):
            x = self.act(self.linears[i](x)) + x
            # x = self.act(self.bns[i](self.linears[i](x))) + x

        x = self.linear_post(x)
        return x





class MLPNOScatter(nn.Module):
    def __init__(self, input_size=2, output_size=3, n_layers=2, n_hidden=64):
        super(MLPNOScatter, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.mlp = MLP(input_size, n_hidden, output_size, n_layers=n_layers)
        self.__name__ = 'MLP_s'


    def forward(self, x, theta):

        feats = torch.cat([x, theta],dim=1)

        feats = self.mlp(feats)

        return feats



class FourierMLP(nn.Module):
    def __init__(self, space_dim=2, theta_dim=1, output_size=3, n_layers=2, n_hidden=64, act='gelu',fourier_dim=0, sigma=1):
        super(FourierMLP, self).__init__()
        self.space_dim = space_dim
        self.theta_dim = theta_dim
        self.output_size = output_size
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.act = act
        self.sigma = sigma
        self.fourier_dim = fourier_dim


        if fourier_dim > 0:
            self.B = nn.Parameter(sigma *torch.randn([space_dim, fourier_dim]),requires_grad=False)
            self.theta_mlp = MLP(theta_dim, fourier_dim, fourier_dim, n_layers=3, act=act)
            self.mlp = MLP(2*fourier_dim + fourier_dim, n_hidden, output_size, n_layers=n_layers,act=act)
        else:
            self.mlp = MLP(space_dim + theta_dim, n_hidden, output_size, n_layers=n_layers,act=act)

        self.__name__ = 'FourierMLP'


    def forward(self, *args):
        if len(args) == 1:
            x = args[0]
            theta = torch.zeros([x.shape[0],1]).to(x.device)   # an ineffective operation
        elif len(args) == 2:
            x, theta = args

        elif len(args) == 3:
            g, u_p, g_u = args
            x = g.ndata['x']
            theta = dgl.broadcast_nodes(g, u_p)

        else:
            raise ValueError
        if self.fourier_dim > 0:
            theta_feats = self.theta_mlp(theta)
            x = torch.cat([torch.sin(2*np.pi*x @ self.B), torch.cos(2*np.pi * x @ self.B), theta_feats],dim=1)
        else:
            x = torch.cat([x, theta],dim=1)

        x = self.mlp(x)

        return x





class MLPNO(nn.Module):
    def __init__(self, input_size=2, output_size=3, n_layers=2, n_hidden=64):
        super(MLPNO, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.mlp = MLP(input_size, n_hidden, output_size, n_layers=n_layers)
        self.__name__ = 'MLP'


    def forward(self, g, u_p, g_u):
        u_p_nodes = dgl.broadcast_nodes(g, u_p)
        feats = torch.cat([g.ndata['x'], u_p_nodes], dim=1)
        feats = self.mlp(feats)

        return feats










