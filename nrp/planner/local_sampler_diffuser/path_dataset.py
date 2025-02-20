import os.path as osp
import sys
import os
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math

import pickle
from collections import namedtuple

import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))

Batch = namedtuple('Batch', 'trajectories conditions occ_grid goal_pos collision')  # collision


class MyDataset(Dataset):
    def __init__(self, data_dir, dataset_size, transform=None, target_transform=None, device="cpu", occ_grid_dim=40):
        print("Instantiating dataset...")
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self.dataset_size = dataset_size
        self.fk = utils.FkTorch(device)
        self._data_dir = data_dir
        self.occ_grid_dim = occ_grid_dim
        self.joint_bounds = torch.asarray(([2.0] * 2 + [math.radians(180)] * 6), dtype=torch.float32)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        file_path = osp.join(self._data_dir, "data_{}.pkl".format(idx))
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        occ_grid, start_pos, goal_pos, sample, _, _, collision = data

        start_pos_normed = torch.Tensor(start_pos) / self.joint_bounds
        cond = {0: start_pos_normed}
        goal_pos_normed = torch.Tensor(goal_pos) / self.joint_bounds
        goal_direction = torch.atan2(goal_pos_normed[1], goal_pos_normed[0]).view(1)
        goal_pos_normed = torch.concat([goal_pos_normed, goal_direction])
        sample_normed = torch.Tensor(sample) / self.joint_bounds
        occ_grid = torch.Tensor(occ_grid).view(1, self.occ_grid_dim, self.occ_grid_dim)
        collision = torch.Tensor(collision)  # collision vector
        # collision = torch.mean(torch.Tensor(collision)).unsqueeze(0).to(torch.float32)  # single label
        # free proportion (from start)
        # try:
        #     index = collision.index(1)
        #     free_prop = torch.Tensor([index / len(collision)])
        # except:
        #     free_prop = torch.Tensor([1])
        # occ_grid_t = torch.Tensor(occ_grid).flatten()
        # condition = torch.concat([occ_grid_t, goal_pos])

        # linkinfo = self.fk.get_link_positions(start_pos.view(1, -1)).view(-1)
        # start_t = torch.cat((start_pos, linkinfo))
        # linkinfo = self.fk.get_link_positions(goal_pos.view(1, -1)).view(-1)
        # goal_direction = torch.atan2(goal_pos[1], goal_pos[0]).view(1)
        # goal_t = torch.cat((goal_pos, linkinfo, goal_direction))
        # linkinfo = self.fk.get_link_positions(sample.view(1, -1)).view(-1)
        # sample_t = torch.cat((sample, linkinfo), dim=-1)

        # return occ_grid_t, start_t, goal_t, sample_t
        # return occ_grid, start_pos, goal_pos, sample
        # return sample, condition
        return Batch(sample_normed, cond, occ_grid, goal_pos_normed, collision)  # collision
    
if __name__ == '__main__':
    data_dir = osp.join(CUR_DIR, "dataset/k_shortest_train")
    bs = 10
    data_cnt = 186388

    dataset = MyDataset(data_dir, data_cnt)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=10, pin_memory=True)

    for data in dataloader:
        # occ_grid, start, goal, samples = data
        # print(occ_grid.shape)
        # print(start.shape)
        # print(goal.shape)
        # print(samples.shape)
        x, cond, occ_grid, goal_pos = data
        print(x)
        print(cond)
        print(goal_pos)
        break
