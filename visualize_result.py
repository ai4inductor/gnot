#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import pickle
import torch
import numpy as np
import torch.nn as nn
import dgl
import matplotlib.pyplot as plt
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from gnot.utils import get_seed, get_num_params
from gnot.args import get_args
from gnot.data_utils import get_dataset, get_model, get_loss_func
from gnot.train import validate_epoch
from gnot.utils import plot_heatmap


if __name__ == "__main__":

    # model_path = '/home/zhongkai/files/ml4phys/tno/operator_transformer/models/checkpoints/ns2d_4ball_b_all_CGPTsearch_1111_22_33_18.pt'
    model_path = '/home/zhongkai/files/ml4phys/tno/operator_transformer/models/checkpoints/inductor2d_all_CGPT0_1213_10_27_10.pt'
    result = torch.load(model_path,map_location='cpu')


    args = result['args']
    config = result['config']
    model_dict = result['model']

    vis_component = 2 if args.component == 'all' else int(args.component)

    device = torch.device('cpu')
    # if not args.no_cuda and torch.cuda.is_available():
    #     device = torch.device('cpu')
        # device = torch.device('cuda:{}'.format(str(args.gpu)))

    kwargs = {'pin_memory': False} if args.gpu else {}
    get_seed(args.seed, printout=False)

    train_dataset, test_dataset = get_dataset(args)

    test_sampler = SubsetRandomSampler(torch.arange(len(test_dataset)))

    test_loader = GraphDataLoader(test_dataset, sampler=test_sampler, batch_size=1, drop_last=False)

    loss_func = get_loss_func(args.loss_name, args, regularizer=True, gamma=args.gamma, normalizer=config['normalizer'])
    metric_func = get_loss_func(args.loss_name, args , regularizer=False, normalizer=config['normalizer'])

    model = get_model(args, **config)


    model.load_state_dict(model_dict)

    model.eval()
    with torch.no_grad():
        # val_metric = validate_epoch(model, metric_func, test_loader, device)
        # print('validation metric {}'.format(val_metric))

        # errs = []
        #
        # for idx, data in enumerate(test_loader):
        #     with torch.no_grad():
        #         g, u_p, g_u = data
        #         g, g_u, u_p = g.to(device), g_u.to(device), u_p.to(device)
        #
        #         out = model(g, u_p, g_u)
        #
        #         y_pred, y = out.squeeze(), g.ndata['y'].squeeze()
        #         _, _, metric = metric_func(g, y_pred, y)
        #
        #         errs.append(metric)
        #         print('idx {} err {}'.format(idx,metric))
        #### view err distribution
        # plt.figure()
        # plt.plot(range(len(errs)),errs)
        # plt.show()

        #### test single case
        idx = 15
        g, u_p, g_u =  test_dataset[idx]
        u_p = u_p.unsqueeze(0)      ### test if necessary
        out = model(g, u_p, g_u)

        x, y = g.ndata['x'][:,0].cpu().numpy(), g.ndata['x'][:,1].cpu().numpy()
        pred = out[:,vis_component].squeeze().cpu().numpy()
        target =g.ndata['y'][:,vis_component].squeeze().cpu().numpy()
        err = pred - target
        print(pred)
        print(target)
        print(err)
        print(np.linalg.norm(err)/np.linalg.norm(target))







        #### choose one to visualize
        cm = plt.cm.get_cmap('rainbow')

        plot_heatmap(x, y, pred,cmap=cm,show=True)
        plot_heatmap(x, y, target,cmap=cm,show=True)


        plt.figure()
        plt.scatter(x, y, c=pred, cmap=cm,s=2)
        plt.colorbar()
        plt.show()
        plt.figure()
        plt.scatter(x, y, c=err, cmap=cm,s=2)
        plt.colorbar()
        plt.show()
        plt.scatter(x, y, c=target, s=2,cmap=cm)
        plt.colorbar()
        plt.show()





