#!/usr/bin/env python
#-*- coding:utf-8 _*-
import torch
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


    def forward(self, x):
        x = self.act(self.linear_pre(x))
        for i in range(self.n_layers):
            x = self.act(self.linears[i](x)) + x
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










