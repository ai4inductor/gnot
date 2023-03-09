#!/usr/bin/env python  
#-*- coding:utf-8 _*-

import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader


x = torch.arange(100).unsqueeze(-1)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MIODataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1,ordered_data=True, shuffle=False, sampler=None,
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

        if ordered_data:
            self.batch_indices = [list(range(i*batch_size, min(i*batch_size+batch_size, len(dataset)))) for i in range(len(dataset)// batch_size)]
            if not drop_last and (len(dataset)//batch_size!=0):
                self.batch_indices = self.batch_indices + [list(range((len(dataset) // batch_size)*batch_size, len(dataset)))]
        else:
            self.batch_indices = list(range(0, (len(dataset) // batch_size)*batch_size)) if drop_last else list(range(0, len(dataset)))
        if shuffle:
            self.batch_indices = np.random.shuffle(self.batch_indices)





    def __iter__(self):
        # 返回一个迭代器，用于遍历数据集中的每个批次
        for indices in self.batch_indices:
            yield [self.dataset[idx] for idx in indices]

    def __len__(self):
        # 返回数据集的批次数
        return len(self.batch_indices)


data = list(range(10))
dataset = MyDataset(data)
dataloader = MIODataLoader(data, batch_size=3,drop_last=True)
for i, batch in enumerate(dataloader):
    print(batch)


