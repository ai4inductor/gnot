#!/usr/bin/env python
#-*- coding:utf-8 _*-
import pickle
import torch
import numpy as np
import torch.nn as nn
import dgl
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from gnot.utils import get_seed, get_num_params
from gnot.args import get_args
from gnot.data_utils import get_dataset, get_model, get_loss_func, get_scatter_dataset, ScatterLpRelLoss
from gnot.train import validate_epoch
from gnot.utils import plot_heatmap


if __name__ == "__main__":

    # model_path = '/home/haozhongkai/files/ml4phys/tno/gnot/data/checkpoints/inductor2d_1_MLP_s0_0324_22_34_15.pt'
    model_path = '/home/haozhongkai/files/ml4phys/tno/gnot/data/checkpoints/inductor2d_b_all_MLP_srel2_0329_10_09_07.pt'
    result = torch.load(model_path,map_location='cpu')


    args = result['args']
    model_dict = result['model']



    device = torch.device('cpu')
    # if not args.no_cuda and torch.cuda.is_available():
    #     device = torch.device('cpu')
        # device = torch.device('cuda:{}'.format(str(args.gpu)))

    kwargs = {'pin_memory': False} if args.gpu else {}
    get_seed(args.seed, printout=False)

    train_dataset, test_dataset = get_scatter_dataset(args)

    # test_dataset = train_dataset

    metric_func = ScatterLpRelLoss(p=2, component=args.component, normalizer=args.normalizer)
    # metric_func = ScatterLpRelLoss(p=2, component=args.component, normalizer=None)

    model = get_model(args)


    model.load_state_dict(model_dict)

    if args.component == 'all':
        args.component = 0
        vis_component = args.component
    else:
        args.component = int(args.component)
        vis_component = 0

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
        idx = 0
        test_idx = torch.arange(test_dataset.data_indexes[idx], test_dataset.data_indexes[idx+1])
        x, theta, y =  test_dataset.X_data[test_idx], test_dataset.theta[test_idx], test_dataset.Y_data[test_idx]
        # u_p = u_p.unsqueeze(0)      ### test if necessary
        out = model(x, theta)

        _,_ ,metric = metric_func(out, y)

        ### in orig space
        ori_pred = args.normalizer.transform(out, component =vis_component,inverse=True)[:,vis_component].squeeze().cpu().numpy()
        ori_target = args.normalizer.transform(y, component = args.component, inverse=True)[:,args.component].squeeze().cpu().numpy()

        pred = out[:,vis_component].squeeze().cpu().numpy()
        target = y[:,args.component].squeeze().cpu().numpy()
        err = pred - target
        # print(pred)
        # print(target)
        # print(err)
        print(np.linalg.norm(err)/np.linalg.norm(target))
        print(metric)







        #### choose one to visualize
        cm = plt.cm.get_cmap('rainbow')
        x = x.numpy()
        plot_heatmap(x[:,0], x[:,1], pred,cmap=cm,show=True)
        plot_heatmap(x[:,0], x[:,1], target,cmap=cm,show=True)



        def plot_scatters(x, pred, target, cm='rainbow'):
            # 假设 x, pred, target, err 已经定义
            err = pred - target
            # fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
            # scatters = []
            #
            # sc1 = axes[0].scatter(x[:, 0], x[:, 1], c=pred, cmap=cm, s=2)
            # scatters.append(sc1)
            # axes[0].set_xlabel('Pred')
            #
            # sc2 = axes[1].scatter(x[:, 0], x[:, 1], c=target, s=2, cmap=cm)
            # scatters.append(sc2)
            # axes[1].set_xlabel('Target')
            #
            # sc3 = axes[2].scatter(x[:, 0], x[:, 1], c=err, cmap=cm, s=2)
            # scatters.append(sc3)
            # axes[2].set_xlabel('Error')
            #
            # # 找到所有子图中的最小值和最大值，以便 colorbar 可以覆盖整个范围
            # vmin = min([sc.get_array().min() for sc in scatters])
            # vmax = max([sc.get_array().max() for sc in scatters])
            #
            # # 对每个子图的颜色映射进行归一化，以便它们可以共享 colorbar
            # norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            # for sc in scatters[:2]:
            #     sc.set_norm(norm)
            #
            # # 在第二张图和第三张图之间添加一个 colorbar
            # cbar_ax = fig.add_axes([0.4, 0.15, 0.02, 0.7])  # 调整 colorbar 位置和尺寸
            # fig.colorbar(scatters[0], cax=cbar_ax)
            #
            # # 为第三张图创建单独的 colorbar
            # fig.colorbar(sc3, ax=axes[2])
            #
            # plt.show()

            fig = plt.figure(figsize=(15, 5))

            gs = GridSpec(1, 5, width_ratios=[1, 1, 0.03, 1, 0.03],wspace=0.4)

            ax1 = plt.subplot(gs[0])
            sc1 = ax1.scatter(x[:, 0], x[:, 1], c=pred, cmap=cm, s=2)
            ax1.set_xlabel('Pred')

            ax2 = plt.subplot(gs[1], sharex=ax1, sharey=ax1)
            sc2 = ax2.scatter(x[:, 0], x[:, 1], c=target, s=2, cmap=cm)
            ax2.set_xlabel('Target')

            vmin = min([sc1.get_array().min(), sc2.get_array().min()])
            vmax = max([sc1.get_array().max(), sc2.get_array().max()])
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

            sc1.set_norm(norm)
            sc2.set_norm(norm)

            cbar_ax1 = plt.subplot(gs[2])
            fig.colorbar(sc1, cax=cbar_ax1)

            ax3 = plt.subplot(gs[3], sharex=ax1, sharey=ax1)
            sc3 = ax3.scatter(x[:, 0], x[:, 1], c=err, cmap=cm, s=2)
            ax3.set_xlabel('Error')

            cbar_ax2 = plt.subplot(gs[4])
            fig.colorbar(sc3, cax=cbar_ax2)

            plt.show()


        plot_scatters(x, pred, target)
        plot_scatters(x, ori_pred, ori_target)








