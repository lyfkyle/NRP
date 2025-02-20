import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
import pickle
from pathlib import Path
from dataclasses import dataclass

from torch.utils.tensorboard import SummaryWriter

from model import FireModel

CUR_DIR = osp.dirname(osp.abspath(__file__))
COL_PRED_THRESHOLD = 0.5
USE_WEIGHTED_SAMPLER = False
DATA_DIR = osp.join(CUR_DIR, "dataset/fire_database")

@dataclass
class FireEntry:
    occ_grid: int
    center_pos: int
    q_target: int
    q_proj: int
    q: int
    prev_q: int
    next_q: int
    proj_num: int
    env_name: str = ""


def add_pos_channels(occ_grid):
    occ_grid_tmp = torch.zeros((4, occ_grid.shape[0], occ_grid.shape[1], occ_grid.shape[2]), device=occ_grid.device)

    voxel_coords = torch.from_numpy(np.indices((occ_grid.shape[2], occ_grid.shape[1], occ_grid.shape[0])).T)
    # print(voxel_coords)

    occ_grid_tmp[0, :,:,:] = occ_grid
    occ_grid_tmp[1, :,:,:] = voxel_coords[:, :, :, 2] / 10.0 - 1.95
    occ_grid_tmp[2, :,:,:] = voxel_coords[:, :, :, 1] / 10.0 - 1.95
    occ_grid_tmp[3, :,:,:] = voxel_coords[:, :, :, 0] / 10.0 + 0.05
    # print(occ_grid_tmp)

    return occ_grid_tmp

class MyDataset(Dataset):
    def __init__(self, data_dir, dataset_size, transform=None, target_transform=None, device="cpu"):
        print("Instantiating dataset...")
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self.dataset_size = dataset_size
        self._data_dir = data_dir
        self._env_entries = {}
        # print("dataset size = {}".format(len(self.dataset)))

        print("start loading")
        data_cnt = 86932
        self._entries = []
        for entry_idx in range(data_cnt):
            with open(osp.join(DATA_DIR, f"entry_{entry_idx}.pkl"), "rb") as f:
                entry = pickle.load(f)
            # assert entry.occ_grid.shape[0] == 4
            # assert entry.occ_grid.shape[1] == 4
            # assert entry.occ_grid.shape[2] == 4
            self._entries.append(entry)

        train_env_dir = osp.join(CUR_DIR, "../../dataset/fetch_11d/gibson/train")
        train_env_dirs = []
        for p in Path(train_env_dir).rglob('env_final.obj'):
            train_env_dirs.append(p.parent)

        for train_env_dir in train_env_dirs:
            env_name = str(train_env_dir).split("/")[-1]
            self._env_entries[env_name] = [e for e in self._entries if e.env_name == env_name]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        file_path = osp.join(self._data_dir, "data_{}.pkl".format(idx))
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        env_name, idx1, idx2, label = data
        e1 = self._env_entries[env_name][idx1]
        e2 = self._env_entries[env_name][idx2]

        occ_grid_1, center_pos_1, q_target_1, q_proj_1 = e1.occ_grid, e1.center_pos, e1.q_target, e1.q_proj
        occ_grid_2, center_pos_2, q_target_2, q_proj_2 = e2.occ_grid, e2.center_pos, e2.q_target, e2.q_proj
        # print(occ_grid_2.shape)

        q_target_1_t = torch.Tensor(q_target_1)
        center_pos_1_t = torch.Tensor(center_pos_1)
        q_proj_1_t = torch.Tensor(q_proj_1).view(-1)  # flatten
        occ_grid_1_t = torch.Tensor(occ_grid_1).view(occ_grid_dim, occ_grid_dim, occ_grid_dim)
        occ_grid_1_t = add_pos_channels(occ_grid_1_t)
        q_target_2_t = torch.Tensor(q_target_2)
        center_pos_2_t = torch.Tensor(center_pos_2)
        q_proj_2_t = torch.Tensor(q_proj_2).view(-1)  # flatten
        occ_grid_2_t = torch.Tensor(occ_grid_2).view(occ_grid_dim, occ_grid_dim, occ_grid_dim)
        occ_grid_2_t = add_pos_channels(occ_grid_2_t)
        label_t = torch.Tensor([label])

        # print(center_pos_1_t, center_pos_2_t)

        return q_target_1_t, q_proj_1_t, occ_grid_1_t, center_pos_1_t, q_target_2_t, q_proj_2_t, occ_grid_2_t, center_pos_2_t, label_t


def contrastive_loss(z1, z2, label):
    dist = torch.norm(z2 - z1, dim=-1)
    y = torch.maximum(torch.zeros_like(dist), 0.5 - dist)
    label = label.view(-1)
    loss = torch.where(label > 0, dist, y)
    # print(dist, y, label, loss)
    return torch.sum(loss)


if __name__ == '__main__':
    bce_loss = torch.nn.BCELoss()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--name', default='1')
    parser.add_argument('--checkpoint', default='')
    args = parser.parse_args()

    # constants
    robot_dim = 11
    linkpos_dim = 24
    occ_grid_dim = 20
    model_name = "fire"
    model_path = osp.join(CUR_DIR, "models/{}.pt".format(model_name))
    data_dir = osp.join(CUR_DIR, "./dataset/similarity_dataset2")
    eval_data_dir = osp.join(CUR_DIR, "./dataset/model_eval")

    # hyperparameters
    bs = 32
    lr = 0.001
    num_epochs = 100
    data_cnt = 98964

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    writer = SummaryWriter(comment = model_name)

    # define networks
    print("robot_dim = ", robot_dim, "linkpos_dim = ", linkpos_dim)
    model = FireModel()
    if args.checkpoint != '':
        print("Loading checkpoint {}.pt".format(args.checkpoint))
        model.load_state_dict(torch.load(osp.join(CUR_DIR, 'models/{}.pt'.format(args.checkpoint))))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_parallel = torch.nn.DataParallel(model)
    else:
        model_parallel = model

    model.to(device)
    model_parallel.to(device)
    # print(model)
    # torch.save(model.state_dict(), model_path)

    # Define the loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5000, verbose=True, factor=0.5)

    train_num = 0
    iter_num = 0
    best_loss = float('inf')

    dataset = MyDataset(data_dir, data_cnt)
    train_size = int(0.9 * data_cnt)
    test_size = data_cnt - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    dataloader = DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=True, num_workers=10, pin_memory=True)
    eval_dataloader = DataLoader(val_set, batch_size=bs, shuffle=True, drop_last=True, num_workers=10, pin_memory=True)
    # eval_dataset = MyDataset(eval_data_dir, eval_data_cnt, None, None)
    # eval_dataloader = DataLoader(eval_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=10, pin_memory=True)
    for iter_num in range(num_epochs):
        print("train")
        model.train()
        model_parallel.train()

        all_labels = []
        all_preds = []
        total_loss = 0
        iter_num_tmp = 0
        for data in dataloader:
            q_target_1_t, q_proj_1_t, occ_grid_1_t, center_pos_1_t, q_target_2_t, q_proj_2_t, occ_grid_2_t, center_pos_2_t, label_t = data
            q_target_1_t = q_target_1_t.to(device)
            q_proj_1_t = q_proj_1_t.to(device)
            occ_grid_1_t = occ_grid_1_t.to(device)
            center_pos_1_t = center_pos_1_t.to(device)
            q_target_2_t = q_target_2_t.to(device)
            q_proj_2_t = q_proj_2_t.to(device)
            occ_grid_2_t = occ_grid_2_t.to(device)
            center_pos_2_t = center_pos_2_t.to(device)
            label_t = label_t.to(device)

            # Perform forward pass
            z1 = model_parallel(occ_grid_1_t, center_pos_1_t, q_target_1_t, q_proj_1_t)
            z2 = model_parallel(occ_grid_2_t, center_pos_2_t, q_target_2_t, q_proj_2_t)

            # calculate loss
            loss = contrastive_loss(z1, z2, label_t)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()
            scheduler.step(loss)

            # Print statistics
            if train_num % 100 == 0:
                print("Learning rate: {}".format(optimizer.param_groups[0]['lr']))
                print('Loss after epoch %d, mini-batch %5d, dataset_sizes %d: , loss: %.3f' % (iter_num, train_num, data_cnt, loss.item()))
                writer.add_scalar('loss/train', loss.item(), train_num)

            train_num += 1
            iter_num_tmp += 1

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), osp.join(CUR_DIR, "models/{}_best.pt".format(model_name)))

            total_loss += loss.item()

        total_loss /= iter_num_tmp
        print('total_loss: %.3f' % (total_loss))

        torch.save(model.state_dict(), model_path)
        print("saved session to ", model_path)

        print("eval")
        model.eval()
        model_parallel.eval()
        for data in eval_dataloader:
            q_target_1_t, q_proj_1_t, occ_grid_1_t, center_pos_1_t, q_target_2_t, q_proj_2_t, occ_grid_2_t, center_pos_2_t, label_t = data
            q_target_1_t = q_target_1_t.to(device)
            q_proj_1_t = q_proj_1_t.to(device)
            occ_grid_1_t = occ_grid_1_t.to(device)
            center_pos_1_t = center_pos_1_t.to(device)
            q_target_2_t = q_target_2_t.to(device)
            q_proj_2_t = q_proj_2_t.to(device)
            occ_grid_2_t = occ_grid_2_t.to(device)
            center_pos_2_t = center_pos_2_t.to(device)
            label_t = label_t.to(device)

            # Perform forward pass
            z1 = model_parallel(occ_grid_1_t, center_pos_1_t, q_target_1_t, q_proj_1_t)
            z2 = model_parallel(occ_grid_2_t, center_pos_2_t, q_target_2_t, q_proj_2_t)

            # calculate loss
            loss = contrastive_loss(z1, z2, label_t)

            # Print statistics
            print('Loss after epoch %d, mini-batch %5d, dataset_sizes %d: , loss: %.3f' % (iter_num, train_num, data_cnt, loss.item()))
            writer.add_scalar('loss/eval', loss.item(), train_num)

    writer.close()