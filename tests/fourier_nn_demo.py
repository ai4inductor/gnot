import urllib.request

import torch

import torch.nn as nn
import tqdm

import numpy as np
import cv2

import cv2
import imageio
import torch
import numpy as np
from tqdm.notebook import tqdm as tqdm
from torch import nn
import torch.nn.functional as F
from gnot.models.MLP import FourierMLP, MLP
import matplotlib.pyplot as plt

from gnot.tests.siren_demo import Siren

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor * 256
    tensor[tensor > 255] = 255
    tensor[tensor < 0] = 0
    tensor = tensor.type(torch.uint8).permute(1, 2, 0).cpu().numpy()

    return tensor

def get_image():
    image_url = 'https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg'
    img = imageio.imread(image_url)[..., :3] / 255.
    c = [img.shape[0] // 2, img.shape[1] // 2]
    r = 256
    img = img[c[0] - r:c[0] + r, c[1] - r:c[1] + r]

    return img


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

downsample = 4
# Get an image that will be the target for our model.
target = torch.tensor(get_image()).unsqueeze(0).permute(0, 3, 1, 2).to(device).float()
target = target[:,:,::downsample,::downsample]

plt.imshow(tensor_to_numpy(target[0]))
plt.show()

# Create input pixel coordinates in the unit square. This will be the input to the model.
coords = np.linspace(0, 1, target.shape[2], endpoint=False)
xy_grid = np.stack(np.meshgrid(coords, coords), -1)
xy_grid = torch.tensor(xy_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous().to(device)

train_no_fourier_feats = False
if train_no_fourier_feats:
    model = nn.Sequential(
            nn.Conv2d(
                2,
                256,
                kernel_size=1,
                padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(
                256,
                256,
                kernel_size=1,
                padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(
                256,
                256,
                kernel_size=1,
                padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(
                256,
                3,
                kernel_size=1,
                padding=0),
            nn.Sigmoid(),

        ).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-4)

    for epoch in range(400):
        optimizer.zero_grad()

        generated = model(xy_grid)

        loss = torch.nn.functional.l1_loss(target, generated)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
          print('Epoch %d, loss = %.03f' % (epoch, float(loss)))
          plt.imshow(tensor_to_numpy(generated[0]))
          plt.show()


train_conv_fourier_feats = False
if train_conv_fourier_feats:
    class GaussianFourierFeatureTransform(torch.nn.Module):
        """
        An implementation of Gaussian Fourier feature mapping.

        "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
           https://arxiv.org/abs/2006.10739
           https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        Given an input of size [batches, num_input_channels, width, height],
         returns a tensor of size [batches, mapping_size*2, width, height].
        """

        def __init__(self, num_input_channels, mapping_size=256, scale=10):
            super().__init__()

            self._num_input_channels = num_input_channels
            self._mapping_size = mapping_size
            self._B = torch.randn((num_input_channels, mapping_size)) * scale

        def forward(self, x):
            assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

            batches, channels, width, height = x.shape

            assert channels == self._num_input_channels, \
                "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

            # Make shape compatible for matmul with _B.
            # From [B, C, W, H] to [(B*W*H), C].
            x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

            x = x @ self._B.to(x.device)

            # From [(B*W*H), C] to [B, W, H, C]
            x = x.view(batches, width, height, self._mapping_size)
            # From [B, W, H, C] to [B, C, W, H]
            x = x.permute(0, 3, 1, 2)

            x = 2 * np.pi * x
            return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


    input_dim = 3
    model = nn.Sequential(
            nn.Conv2d(input_dim, 256,kernel_size=1,padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256,256,kernel_size=1,padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256,256,kernel_size=1,padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256,3,kernel_size=1,padding=0),
            nn.Sigmoid(),
        ).to(device)


    # Note: this can be done outside of the training loop, since the result at this stage is unchanged during the course of training.
    x = GaussianFourierFeatureTransform(2, 128, 10)(xy_grid)


    xy_grid_flat = xy_grid.view(-1,2)

    # model = FourierMLP(input_size=2,output_size=3,n_layers=5,n_hidden=128,act='relu',fourier_dim=128, sigma=10).to(device)
    # model = nn.Sequential(nn.Linear(2, 128),nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),nn.Linear(128, 128),nn.ReLU(),nn.Linear(128,3), nn.Sigmoid()).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-4)

    for epoch in range(400):
        optimizer.zero_grad()

        generated = model(x)
        # generated = F.sigmoid(model(xy_grid_flat).view_as(target))

        loss = torch.nn.functional.l1_loss(target, generated)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
          print('Epoch %d, loss = %.03f' % (epoch, float(loss)))
          plt.imshow(tensor_to_numpy(generated[0]))
          plt.show()

train_mlp = True
if train_mlp:
    input_dim = 3
    model = nn.Sequential(
        nn.Conv2d(input_dim, 256, kernel_size=1, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(256),

        nn.Conv2d(256, 256, kernel_size=1, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(256),

        nn.Conv2d(256, 256, kernel_size=1, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(256),

        nn.Conv2d(256, 3, kernel_size=1, padding=0),
        nn.Sigmoid(),
    ).to(device)

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.n_layers = 4
            self.n_hidden =256
            self.linear_in = nn.Linear(2, self.n_hidden)
            self.mlp = nn.ModuleList([nn.Linear(self.n_hidden, self.n_hidden) for _ in range(self.n_layers)])
            self.bns = nn.ModuleList([nn.BatchNorm1d(self.n_hidden) for _ in range(self.n_layers)])
            self.linear_out = nn.Linear(self.n_hidden,3)
            self.act = F.relu

        def forward(self, x):
            # orig_shape = x.shape
            # x = x.permute(0,2,3,1).view(-1,2)
            x = self.linear_in(x)
            for i in range(len(self.mlp)):
                x = self.bns[i](self.act(self.mlp[i](x))) +x
            x = self.linear_out(x)
            return x

    # Note: this can be done outside of the training loop, since the result at this stage is unchanged during the course of training.
    # x = GaussianFourierFeatureTransform(2, 128, 10)(xy_grid)
    x = xy_grid
    xy_grid_flat = xy_grid.permute(0,2,3,1).view(-1, 2)

    # model = MLP().to(device)
    # model =FourierMLP(input_size=2,output_size=3,n_hidden=128,n_layers=4,act='gelu',fourier_dim=128,sigma=32).to(device)
    model = Siren(in_features=2,out_features=3,hidden_layers=4,hidden_features=128,outermost_linear=True).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3)

    for epoch in range(400):
        optimizer.zero_grad()


        generated = model(xy_grid_flat).permute(1,0).view_as(target)
        # generated = F.sigmoid(model(xy_grid_flat).view_as(target))

        loss = torch.nn.functional.mse_loss(target, generated)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:

            err = (((target - generated)**2).sum()/(target**2).sum())**0.5

            print('Epoch {}, loss = {} err {}'.format(epoch, float(loss), err))
            plt.imshow(tensor_to_numpy(generated.view_as(target)[0]))
            plt.show()