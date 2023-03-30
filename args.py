#!/usr/bin/env python
#-*- coding:utf-8 _*-

import argparse






def get_args():
    parser = argparse.ArgumentParser(description='GNOT for operator learning')
    parser.add_argument('--dataset',type=str,
                        default='inductor2d',
                        choices = ['ns2d_4ball','inductor2d','inductor2d_b'])

    parser.add_argument('--space-dim',type=int, default=0,
                        help='If set to 0, auto search the first number in the dataset name as space dim')

    parser.add_argument('--component',type=str,
                        default='all',)



    parser.add_argument('--seed', type=int, default=2022, metavar='Seed',
                        help='random seed (default: 1127802)')

    parser.add_argument('--gpu', type=int, default=1, help='gpu id')
    parser.add_argument('--use-tb', type=int, default=0, help='whether use tensorboard')
    parser.add_argument('--comment',type=str,default="",help="comment for the experiment")

    #### new add options
    # parser.add_argument('--train-portion', type=float, default=0.5)
    # parser.add_argument('--valid-portion', type=float, default=0.1)

    parser.add_argument('--sort-data',type=int, default=0)

    parser.add_argument('--normalize_x',type=str, default='minmax',
                        choices=['none','minmax','unit'])
    parser.add_argument('--use-normalizer',type=str, default='unit',
                        choices=['none','minmax','unit','quantile'],
                        help = "whether normalize y")


    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--optimizer', type=str, default='AdamW',choices=['Adam','AdamW'])

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='max learning rate (default: 0.001)')
    parser.add_argument('--weight-decay',type=float,default=5e-6
                        )
    parser.add_argument('--grad-clip', type=str, default=1000.0
                        )
    #### full batch training
    parser.add_argument('--batch-size', type=int, default=4, metavar='bsz',
                        help='input batch size for training (default: 8)')
    #### scatter training
    parser.add_argument('--scatter-batch-size',type=int,default=40000,
                        help='input batch size for scatter training')
    parser.add_argument('--merge-inputs',type=int, default=0,
                        help='whether merge input functions for scatter training')

    parser.add_argument('--val-batch-size', type=int, default=8, metavar='bsz',
                        help='input batch size for validation (default: 4)')


    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')


    parser.add_argument('--lr-method',type=str, default='step',
                        choices=['cycle','step','warmup'])
    parser.add_argument('--lr-step-size',type=int, default=50
                        )
    parser.add_argument('--warmup-epochs',type=int, default=50)

    parser.add_argument('--loss-name',type=str, default='rel2',
                        choices=['rel2','rel1', 'l2', 'l1'])
    #### public model architecture parameters

    parser.add_argument('--model-name', type=str, default='FourierMLP',
                        choices=['CGPT', 'GNOT', 'MLP','MLP_s','FourierMLP'])
    parser.add_argument('--n-hidden',type=int, default=128)
    parser.add_argument('--n-layers',type=int, default=5)

    #### MLP/DeepONet parameters
    #### GNN parameters
    # parser.add_argument('--gnn-layers',type=int, default=3)

    #### FNO parameters
    # parser.add_argument('--modes',type=int, default=16)

    #### FFNO parameters
    # parser.add_argument('--n-grid',type=int, default=2048)

    ### parameters for Galerkin Transformer, Performer, GPT
    # common
    parser.add_argument('--act', type=str, default='gelu',choices=['gelu','relu','tanh','sigmoid'])
    parser.add_argument('--n-head',type=int, default=1)
    parser.add_argument('--ffn-dropout', type=float, default=0.0, metavar='ffn_dropout',
                        help='dropout for the FFN in attention (default: 0.0)')
    parser.add_argument('--attn-dropout',type=float, default=0.0)
    parser.add_argument('--mlp-layers',type=int, default=3)
    # GPT
    # parser.add_argument('--subsampled-len',type=int, default=256)
    parser.add_argument('--attn-type',type=str, default='linear', choices=['random','linear','gated','hydra','kernel'])
    parser.add_argument('--hfourier-dim',type=int,default=128)
    parser.add_argument('--sigma',type=float, default=2**3)

    # GNOT
    parser.add_argument('--n-experts',type=int, default=1)
    parser.add_argument('--branch-sizes',nargs="*",type=int, default=[2])
    parser.add_argument('--n-inner',type=int, default=4)



    # performer
    # parser.add_argument('--dim-head',type=float, default=64)

    # oformer
    # parser.add_argument('--random-fourier-dim',type=int, default=48)
    # parser.add_argument('--random-fourier-sigma',type=float, default=1.0)
    # parser.add_argument('--biased-init',type=int, default=1)
    # GKTransformer
    # parser.add_argument('--attention-type', type=str, default='galerkin', metavar='attn_type',
    #                     help='input attention type for encoders (possile: fourier (alias integral, local), galerkin (alias global), softmax (official PyTorch implementation), linear (standard Q(K^TV) with softmax), default: fourier)')
    # parser.add_argument('--xavier-init', type=float, default=0.01, metavar='xavier_init',
    #                     help='input Xavier initialization strength for Q,K,V weights (default: 0.01)')
    # parser.add_argument('--diagonal-weight', type=float, default=0.01, metavar='diagonal weight',
    #                     help='input diagonal weight initialization strength for Q,K,V weights (default: 0.01)')
    #
    # parser.add_argument('--encoder-dropout', type=float, default=0.0, metavar='encoder_dropout',
    #                     help='dropout after the scaled dot-product in attention (default: 0.0)')
    # parser.add_argument('--decoder-dropout', type=float, default=0.0, metavar='decoder_dropout',
    #                     help='dropout for the decoder layers (default: 0.0)')
    # parser.add_argument('--atten-norm',type=int,default=1)
    # parser.add_argument('--batch-norm',type=int,default=0)
    # parser.add_argument('--layer-norm', type=int, default=0,
    #                     help='use the conventional layer normalization')
    # parser.add_argument('--show-batch', action='store_true', default=False,
    #                     help='show batch training result')

    # parser.add_argument('--pos-dim',type=int, default=2,
    #                     help = "dim of position, 1 for 1d...")
    # parser.add_argument('--num-feat-layers',type=int, default=0)
    # parser.add_argument('--num-encoder-layers',type=int, default=4)
    # parser.add_argument('--dim-feedforward',type=int, default=192)
    # parser.add_argument('--freq-dim',type=int, default=48)
    # parser.add_argument('--num-regressor-layers',type=int, default=2)






    return parser.parse_args()

