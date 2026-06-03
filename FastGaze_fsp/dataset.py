from torch import nn, Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from os.path import join
import numpy as np

import random
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from os.path import join

class fixation_dataset(Dataset):
    def __init__(self, fixs, img_ftrs_dir):
        self.fixs = fixs
        self.img_ftrs_dir = img_ftrs_dir

        # 记录每类的数量
        self.class_counts = [len(self.fixs[0]), len(self.fixs[1]), len(self.fixs[2])]
        print('class_counts',self.class_counts)
        # 计算每类的权重，使得数量较少的类别具有更高的权重
        total_count = sum(self.class_counts)
        self.class_weights = [total_count / count for count in self.class_counts]
        print('class_weights',self.class_weights)
        
        # 为每个样本分配权重，基于所属类别的权重
        self.sample_weights = []
        for i, class_count in enumerate(self.class_counts):
            # 为该类别的每个样本分配相同的权重
            self.sample_weights.extend([self.class_weights[i]] * class_count)
    
    def __len__(self):
        # 数据集的总长度是三类数据的总和
        return sum(self.class_counts)

    def __getitem__(self, idx):
        # 确定样本属于哪个类别，并获取对应的 fixation
        if idx < len(self.fixs[0]):
            number = 0
            fixation = self.fixs[number][idx]
        elif idx < len(self.fixs[0]) + len(self.fixs[1]):
            number = 1
            fixation = self.fixs[number][idx - len(self.fixs[0])]
        else:
            number = 2
            fixation = self.fixs[number][idx - len(self.fixs[0]) - len(self.fixs[1])]

        # 根据索引加载图像特征
        image_ftrs = torch.load(join(self.img_ftrs_dir[number], fixation['task'].replace(' ', '_'), fixation['img_name'].replace('jpg', 'pth'))).unsqueeze(0)
        return {'task': fixation['task'], 'tgt_y': fixation['tgt_seq_y'].float(), 'tgt_x': fixation['tgt_seq_x'].float(), 'tgt_t': fixation['tgt_seq_t'].float(),'src_img': image_ftrs, 'tpye':number}
        

    def get_sampler(self):
        """返回 WeightedRandomSampler 实例，用于均衡采样"""
        return WeightedRandomSampler(weights=self.sample_weights, num_samples=len(self.sample_weights), replacement=True)

# class fixation_dataset(Dataset):
#     def __init__(self, fixs, img_ftrs_dir):
#         self.fixs = fixs
#         self.img_ftrs_dir = img_ftrs_dir

        
#     def __len__(self):
#         return len(self.fixs[0]) + len(self.fixs[1]) + len(self.fixs[2]) 
        
#     def __getitem__(self, idx):
#         # 0 TP, 1 TA, 2 FV, number: 21622 3258 43249 有点不均衡
#         if idx < len(self.fixs[0]):
#             number = 0
#             fixation = self.fixs[number][idx]
#         elif idx < len(self.fixs[0])+len(self.fixs[1]):
#             number = 1
#             fixation = self.fixs[number][idx-len(self.fixs[0])]
#         else:
#             number = 2
#             fixation = self.fixs[number][idx-len(self.fixs[0])-len(self.fixs[1])]
            
#         image_ftrs = torch.load(join(self.img_ftrs_dir[number], fixation['task'].replace(' ', '_'), fixation['img_name'].replace('jpg', 'pth'))).unsqueeze(0)

        
#         return {'task': fixation['task'], 'tgt_y': fixation['tgt_seq_y'].float(), 'tgt_x': fixation['tgt_seq_x'].float(), 'tgt_t': fixation['tgt_seq_t'].float(),'src_img': image_ftrs, 'tpye':number}
        



class COCOSearch18Collator(object):
    def __init__(self, embedding_dict, max_len, im_h, im_w, patch_size):
        self.embedding_dict = embedding_dict
        self.max_len = max_len
        self.im_h = im_h
        self.im_w = im_w
        self.patch_size = patch_size
        self.PAD = [-3, -3, -3]

    def __call__(self, batch):
        batch_tgt_y = []
        batch_tgt_x = []
        batch_tgt_t = []
        batch_imgs = []
        batch_tasks = []
        batch_types = []
        
        for t in batch:
            batch_tgt_y.append(t['tgt_y'])
            batch_tgt_x.append(t['tgt_x'])
            batch_tgt_t.append(t['tgt_t'])
            batch_imgs.append(t['src_img'])
            batch_tasks.append(self.embedding_dict[t['task']])
            batch_types.append(t['tpye'])
            
        batch_tgt_y.append(torch.zeros(self.max_len))
        batch_tgt_x.append(torch.zeros(self.max_len))
        batch_tgt_t.append(torch.zeros(self.max_len))
        batch_tgt_y = pad_sequence(batch_tgt_y, padding_value=self.PAD[0])[:, :-1].unsqueeze(-1)
        batch_tgt_x = pad_sequence(batch_tgt_x, padding_value=self.PAD[1])[:, :-1].unsqueeze(-1)
        batch_tgt_t = pad_sequence(batch_tgt_t, padding_value=self.PAD[2])[:, :-1].unsqueeze(-1)
        
        batch_imgs = torch.cat(batch_imgs, dim = 0)
        batch_tgt = torch.cat([batch_tgt_y, batch_tgt_x, batch_tgt_t], dim = -1).long().permute(1, 0, 2)
        batch_firstfix = torch.tensor([(self.im_h//2)*self.patch_size, (self.im_w//2)*self.patch_size]).unsqueeze(0).repeat(batch_imgs.size(0), 1)
        batch_tgt_padding_mask = batch_tgt[:, :, 0] == self.PAD[0]
        
        
        return batch_imgs, batch_tgt, batch_tgt_padding_mask, torch.tensor(batch_tasks), batch_firstfix, torch.tensor(batch_types)

        
        
