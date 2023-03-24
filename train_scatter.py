#!/usr/bin/env python
# -*- coding:utf-8 _*-
import sys
import os

sys.path.append('../..')
sys.path.append('..')

'''
    A general code framework for training neural operator on irregular domains
    This file contains full sample training for neural operators, i.e. takes meshes as inputs and outputs values on mesh points

    Supported method:
    1. Transformer
    2. FNO
    3. DeepONet / MLP
'''

import yaml
import pickle
import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import dgl
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from collections import OrderedDict
from torch.optim.lr_scheduler import OneCycleLR, StepLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from gnot.args import get_args
from gnot.data_utils import get_scatter_dataset,get_model, ScatterLpLoss
from gnot.utils import get_seed, get_num_params, timing
from gnot.models.optimizer import Adam, AdamW

EPOCH_SCHEDULERS = ['ReduceLROnPlateau', 'StepLR', 'MultiplicativeLR',
                    'MultiStepLR', 'ExponentialLR', 'LambdaLR']


def train(args,
          model,
          loss_func,
          metric_func,
          train_dataset,
          test_dataset,
          train_loader,
          test_loader,
          optimizer,
          lr_scheduler,
          epochs=10,
          writer=None,
          device="cuda",
          patience=10,
          grad_clip=0.999,
          start_epoch: int = 0,
          print_freq: int = 20,
          model_save_path='./models',
          save_mode='state_dict',
          model_name='model.pt',
          result_name='result.pt'):
    loss_train = []
    loss_val = []
    loss_epoch = []
    lr_history = []
    it = 0

    if patience is None or patience == 0:
        patience = epochs
    result = None
    start_epoch = start_epoch
    end_epoch = start_epoch + epochs
    best_val_metric = np.inf
    best_val_epoch = None
    save_mode = 'state_dict' if save_mode is None else save_mode
    stop_counter = 0
    is_epoch_scheduler = any(s in str(lr_scheduler.__class__) for s in EPOCH_SCHEDULERS)

    for epoch in range(start_epoch, end_epoch):
        model.train()
        torch.cuda.empty_cache()

        train_idxs = torch.randperm(len(train_dataset)).to(device).split(args.scatter_batch_size)

        for idx in train_idxs:

            batch = (train_dataset.X_data[idx], train_dataset.theta[idx], train_dataset.Y_data[idx])
            loss = train_batch(model, loss_func, batch, optimizer, lr_scheduler, device, grad_clip=grad_clip)


            loss = np.array(loss)
            loss_epoch.append(loss)
            it += 1
            lr = optimizer.param_groups[0]['lr']
            lr_history.append(lr)
            log = f"epoch: [{epoch + 1}/{end_epoch}]"
            if loss.ndim == 0:  # 1 target loss
                _loss_mean = np.mean(loss_epoch)
                log += " loss: {:.6f}".format(_loss_mean)
            else:
                _loss_mean = np.mean(loss_epoch, axis=0)
                for j in range(len(_loss_mean)):
                    log += " | loss {}: {:.6f}".format(j, _loss_mean[j])
            log += " | current lr: {:.3e}".format(lr)

            if it % print_freq == 0:
                print(log)

            if writer is not None:
                for j in range(len(_loss_mean)):
                    writer.add_scalar("train_loss_{}".format(j), _loss_mean[j], it)  #### loss 0 seems to be the sum of all loss
        loss_train.append(_loss_mean)
        loss_epoch = []

        val_result = validate_epoch(model, metric_func, test_dataset, args.scatter_batch_size * 4, device)

        loss_val.append(val_result["metric"])
        val_metric = val_result["metric"].sum()

        if val_metric < best_val_metric:
            best_val_epoch = epoch
            best_val_metric = val_metric
        #     stop_counter = 0
        #     if save_mode == 'state_dict':
        #         torch.save(model.state_dict(), os.path.join(model_save_path, model_name))
        #     else:
        #         torch.save(model, os.path.join(model_save_path, model_name))
        #     best_model_state_dict = {k: v.to('cpu') for k, v in model.state_dict().items()}
        #     best_model_state_dict = OrderedDict(best_model_state_dict)
        #
        # else:
        #     stop_counter += 1

        if lr_scheduler and is_epoch_scheduler:
            if 'ReduceLROnPlateau' in str(lr_scheduler.__class__):
                lr_scheduler.step(val_metric)
            else:
                lr_scheduler.step()

        if val_result["metric"].size == 1:
            log = "| val metric 0: {:.6f} ".format(val_metric)

        else:
            log = ''
            for i, metric_i in enumerate(val_result['metric']):
                log += '| val metric {} : {:.6f} '.format(i, metric_i)
            # metric_0, metric_1 = val_result["metric"][0], val_result["metric"][1]
            # log = "| val metric 1: {:.6f} | val metric 2: {:.6f} ".format(metric_0, metric_1)

        if writer is not None:
            if val_result["metric"].size == 1:
                writer.add_scalar('val loss {}'.format(metric_func.component), val_metric, epoch)
            else:
                for i, metric_i in enumerate(val_result['metric']):
                    writer.add_scalar('val loss {}'.format(i), metric_i, epoch)

        log += "| best val: {:.6f} at epoch {} | current lr: {:.3e}".format(best_val_metric, best_val_epoch + 1, lr)

        desc_ep = ""
        if _loss_mean.ndim == 0:  # 1 target loss
            desc_ep += "| loss: {:.6f}".format(_loss_mean)
        else:
            for j in range(len(_loss_mean)):
                if _loss_mean[j] > 0:
                    desc_ep += "| loss {}: {:.3e}".format(j, _loss_mean[j])

        desc_ep += log
        print(desc_ep)

        result = dict(
            best_val_epoch=best_val_epoch,
            best_val_metric=best_val_metric,
            loss_train=np.asarray(loss_train),
            loss_val=np.asarray(loss_val),
            lr_history=np.asarray(lr_history),
            # best_model=best_model_state_dict,
            optimizer_state=optimizer.state_dict()
        )
        # pickle.dump(result, open(os.path.join(model_save_path, result_name),'wb'))
    return result

# @timing
def train_batch(model, loss_func, data, optimizer, lr_scheduler, device, grad_clip=0.999):
    optimizer.zero_grad()

    x, theta, y = data

    x, theta, y = x.to(device), theta.to(device), y.to(device)

    out = model(x, theta)
    # y, out = y.squeeze(), out.squeeze()
    loss, reg, _ = loss_func(out, y)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    if lr_scheduler:
        lr_scheduler.step()

    return loss.item(), reg.item()

@timing
def validate_epoch(model, metric_func, valid_dataset, batchsize, device):
    model.eval()
    y_data, y_pred = [],  []

    with torch.no_grad():
        test_idxs = torch.arange(len(valid_dataset)).split(batchsize)
        # for _, data in enumerate(valid_loader):
        for idx in test_idxs:

            x, theta, y = test_dataset.X_data[idx].to(device), test_dataset.theta[idx].to(device), test_dataset.Y_data[idx].to(device)

            out = model(x, theta)

            out, y = out.cpu(), y.cpu()  # 12 GB memory can store 5e8 points, so do not need to transfer it to cpu
            y_data.append(y)
            y_pred.append(out)
            # _, _, metric = metric_func(g, y_pred, y)

            # metric_val.append(metric)

        y_data, y_pred = torch.cat(y_data, dim=0), torch.cat(y_pred, dim=0)
        offsets =  valid_dataset.len_inputs.to(y_data.device)
        ### GPU
        _, _, metric_val = loss_func(y_pred, y_data, offsets)




    return dict(metric=metric_val)


if __name__ == "__main__":
    args = get_args()
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(str(args.gpu)))
    else:
        device = torch.device("cpu")

    kwargs = {'pin_memory': False} if args.gpu else {}
    get_seed(args.seed, printout=False)

    train_dataset, test_dataset = get_scatter_dataset(args)
    #### DO not use dataloader for long tensordataset
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.scatter_batch_size , shuffle=True, drop_last=False,num_workers=8,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.scatter_batch_size * 5, shuffle=False, drop_last=False,num_workers=8,pin_memory=True)




    args.normalizer = train_dataset.y_normalizer.to(device) if train_dataset.y_normalizer is not None else None

    #### set random seeds
    get_seed(args.seed)
    torch.cuda.empty_cache()

    if args.loss_name in ['rel1' , 'l1']:
        loss_func = ScatterLpLoss(p=1, component=args.component, normalizer=args.normalizer)
        metric_func = ScatterLpLoss(p=1, component=args.component, normalizer=args.normalizer)
    elif args.loss_name in ["rel2" , 'l2']:
        loss_func = ScatterLpLoss(p=2, component=args.component, normalizer=args.normalizer)
        metric_func = ScatterLpLoss(p=2, component=args.component, normalizer=args.normalizer)
    else:
        raise NotImplementedError

    model = get_model(args)
    model = model.to(device)
    print(f"\nModel: {model.__name__}\t Number of params: {get_num_params(model)}")

    path_prefix = args.dataset + '_{}_'.format(args.component) + model.__name__ + args.comment + time.strftime(
        '_%m%d_%H_%M_%S')
    model_path, result_path = path_prefix + '.pt', path_prefix + '.pkl'

    print(f"Saving model and result in ./../models/checkpoints/{model_path}\n")

    if args.use_tb:
        writer_path = './data/logs/' + path_prefix
        log_path = writer_path + '/params.txt'
        writer = SummaryWriter(log_dir=writer_path)
        fp = open(log_path, "w+")
        sys.stdout = fp

    else:
        writer = None
        log_path = None

    print(model)
    # print(config)

    epochs = args.epochs
    lr = args.lr

    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == "AdamW":
        # if hasattr(model, 'configure_optimizers'):
        # print('Using model specified configured optimizer')
        # optimizer = model.configure_optimizers(lr=lr, weight_decay=args.weight_decay,betas=(0.9,0.999))
        # else:
        # optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    else:
        raise NotImplementedError

    if args.lr_method == 'cycle':
        print('Using cycle learning rate schedule')
        scheduler = OneCycleLR(optimizer, max_lr=lr, div_factor=1e4, pct_start=0.2, final_div_factor=1e4,
                               steps_per_epoch=len(train_loader), epochs=epochs)
    elif args.lr_method == 'step':
        print('Using step learning rate schedule')
        scheduler = StepLR(optimizer, step_size=args.lr_step_size * len(train_loader), gamma=0.7)
    elif args.lr_method == 'warmup':
        print('Using warmup learning rate schedule')
        scheduler = LambdaLR(optimizer, lambda steps: min((steps + 1) / (args.warmup_epochs * len(train_loader)),
                                                          np.power(
                                                              args.warmup_epochs * len(train_loader) / float(steps + 1),
                                                              0.5)))

    time_start = time.time()

    result = train(args, model, loss_func, metric_func, train_dataset, test_dataset,
                   train_loader, test_loader,
                   optimizer, scheduler,
                   epochs=epochs,
                   grad_clip=args.grad_clip,
                   patience=None,
                   model_name=model_path,
                   model_save_path='./../data/checkpoints/',
                   result_name=result_path,
                   writer=writer,
                   device=device)

    print('Training takes {} seconds.'.format(time.time() - time_start))

    # result['args'], result['config'] = args, config
    checkpoint = {'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, os.path.join('./../data/checkpoints/{}'.format(model_path)))
    # pickle.dump(checkpoint, open(os.path.join('./../models/checkpoints/{}'.format(model_path), result_path),'wb'))
    # model.load_state_dict(torch.load(os.path.join('./../models/checkpoints/', model_path)))
    model.eval()
    val_metric = validate_epoch(model, metric_func, test_loader, device)
    print(f"\nBest model's validation metric in this run: {val_metric}")




