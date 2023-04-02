#!/usr/bin/env python  
#-*- coding:utf-8 _*-

import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

import time

from gnot.models.MLP import MLP

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        # return output, coords
        return output

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

if __name__ == "__main__":
    device = torch.device('cuda:0')

    # net = Siren(in_features=1, out_features=1, hidden_features=128,hidden_layers=3, outermost_linear=True).to(device)
    net = MLP(n_input=1,n_hidden=128,n_output=1,n_layers=10,act='relu').to(device)
    ## function fitting demo


    X = torch.linspace(-4,4,1000).unsqueeze(-1).to(device)

    Y = 0.2*torch.sin(6*X)*(X<=0)+(1+0.1*X*torch.cos(12*X))*(X>0)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3,betas=(0.9,0.999))

    num_epochs = 3000
    plot_interval = num_epochs //5
    losses = []

    for epoch in range(num_epochs):
        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = net(X)
        # 计算损失
        loss = criterion(outputs, Y)

        # 反向传播
        loss.backward()

        # 更新权重
        optimizer.step()

        losses.append(loss.item())

        if epoch % plot_interval == 0:
            print('epoch {} loss {}'.format(epoch, loss.item()))
            plt.figure()
            X_np = X.squeeze().detach().cpu().numpy()
            Y_np = Y.squeeze().detach().cpu().numpy()
            out_np =outputs.squeeze().detach().cpu().numpy()
            plt.plot(X_np,Y_np,label="true")
            plt.plot(X_np, out_np, label="pred")
            plt.legend()
            plt.show()

    plt.figure()
    plt.semilogy(np.arange(num_epochs),losses)
    plt.show()