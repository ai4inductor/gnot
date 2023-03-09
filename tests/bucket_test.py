#!/usr/bin/env python  
#-*- coding:utf-8 _*-
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler

import math
import torch
import numpy as np

# 自定义 collate_fn 函数，用于实现 bucketing
def collate_fn(data, num_buckets=1):
    # 将数据按照长度排序
    data.sort(key=lambda x: len(x[0]), reverse=True)
    # 计算每个桶的长度范围
    batch_size = len(data)
    bucket_size = math.ceil(batch_size / num_buckets)
    buckets = [data[i:i+bucket_size] for i in range(0, batch_size, bucket_size)]

    # 填充每个桶的数据
    batches = []
    for bucket in buckets:
        inputs, targets = zip(*bucket)
        padded_inputs = pad_sequence(inputs, batch_first=False, padding_value=0)
        padded_targets = pad_sequence(targets, batch_first=False, padding_value=0)
        batches.append((padded_inputs, padded_targets))

    return batches


## generate random data
len_list = np.random.randint(2,100, [100])
x = [torch.randn([l,1]) for l in len_list]
y = torch.randn([100, 1])

class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return [self.x[index], self.y[index]]

    def __len__(self):
        return len(self.x)

# 加载数据集
dataset = MyDataset(x,y)

# dataset.sort(key=lambda x: len(x[0]), reverse=True)



class SubsetRandomSampler(torch.utils.data.sampler.Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    # indices: Sequence[int]

    def __init__(self, indices, generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self):
        for i in torch.randperm(len(self.indices), generator=self.generator):
            yield self.indices[i]

    def __len__(self):
        return len(self.indices)

sampler = SubsetRandomSampler(torch.arange(len(dataset)))
loader = DataLoader(dataset, sampler=sampler,  batch_size=32, collate_fn=collate_fn)

# print(list(loader))
# 遍历数据集
for data in loader:
    # 在每个桶内进行前向计算和反向传播等操作
    print(data)
