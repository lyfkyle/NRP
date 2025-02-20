import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import torch
import networkx as nx
import random
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import argparse
import pickle
from torch.utils.tensorboard import SummaryWriter

from model import VAE
import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name', default='3')
parser.add_argument('--checkpoint', default='')
args = parser.parse_args()

class MyDataset(Dataset):
    def __init__(self, data_dir, dataset_size, transform=None, target_transform=None, device="cpu"):
        print("Instantiating dataset...")
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self.dataset_size = dataset_size
        self.fk = utils.FkTorch(device)
        self._data_dir = data_dir

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        file_path = osp.join(self._data_dir, "data_{}.pkl".format(idx))
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        occ_grid, start_pos, goal_pos, sample = data

        start_pos = utils.normalize_state(start_pos)
        goal_pos = utils.normalize_state(goal_pos)
        sample = utils.normalize_state(sample)

        start_pos = torch.Tensor(start_pos)
        goal_pos = torch.Tensor(goal_pos)
        sample = torch.Tensor(sample)
        occ_grid_t = torch.Tensor(occ_grid).view(1, occ_grid_dim, occ_grid_dim)

        linkinfo = self.fk.get_link_positions(start_pos.view(1, -1)).view(-1)
        start_t = torch.cat((start_pos, linkinfo))
        linkinfo = self.fk.get_link_positions(goal_pos.view(1, -1)).view(-1)
        goal_direction = torch.atan2(goal_pos[1], goal_pos[0]).view(1)
        goal_t = torch.cat((goal_pos, linkinfo, goal_direction))
        linkinfo = self.fk.get_link_positions(sample.view(1, -1)).view(-1)
        sample_t = torch.cat((sample, linkinfo), dim=-1)

        return occ_grid_t, start_t, goal_t, sample_t

def vae_loss(recon_x, x, mean, log_var):
    bs = recon_x.shape[0]
    BCE = torch.nn.functional.mse_loss(recon_x.view(bs, -1), x.view(bs, -1))
    KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1), dim=0)

    return BCE + alpha * KLD, BCE, KLD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# constants
z_dim = 5
robot_dim = 8
linkpos_dim = 12
goal_dim = robot_dim + linkpos_dim + 1
occ_grid_dim = 100
model_name = "cvae_global"
path = osp.join(CUR_DIR, "models/{}.pt".format(model_name))
data_dir = osp.join(CUR_DIR, "dataset/cvae_train")
eval_data_dir = osp.join(CUR_DIR, "dataset/cvae_eval")
writer = SummaryWriter(comment = "_{}".format(model_name))

# hyperparameters
bs = 512
lr = 1e-3
num_steps = 10000
num_epochs = 50
alpha = 0.01
data_cnt = 1000000
eval_data_cnt = 150000

# define networks
print("dim = ", robot_dim)
print("z_dim = ", z_dim)
model = VAE(z_dim, robot_dim + linkpos_dim + goal_dim, robot_dim + linkpos_dim)
if args.checkpoint != '':
    print("Loading checkpoint {}.pt".format(args.checkpoint))
    model.load_state_dict(torch.load(osp.join(CUR_DIR, 'models/{}.pt'.format(args.checkpoint))))
model.to(device)

# Define the loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10000, verbose=True, factor=0.5)

# dataset and dataloader
dataset = MyDataset(data_dir, data_cnt)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=10, pin_memory=True)
eval_dataset = MyDataset(eval_data_dir, eval_data_cnt, None, None)
eval_dataloader = DataLoader(eval_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=10, pin_memory=True)

# Run the training loop
i = 0
for epoch in range(num_epochs):
    model.train()
    for data in dataloader:
        occ_grid_t, start_t, goal_t, samples_t = data
        occ_grid_t = occ_grid_t.to(device)
        start_t = start_t.to(device)
        goal_t = goal_t.to(device)
        samples_t = samples_t.to(device)
        context_t = torch.cat((start_t, goal_t), dim=-1)

        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        recon_samples, means, log_var, z = model(samples_t, occ_grid_t, context_t)

        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(recon_samples, samples_t, means, log_var)

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        scheduler.step(loss)

        # Print statistics
        if i % 100 == 0:
            print('Loss after mini-batch %5d, epoch %d : %.3f' % (i, epoch, loss.item()))
            print('recon_loss after mini-batch %5d, epoch %d: %.3f' % (i, epoch, recon_loss.item()))
            print('kl_loss after mini-batch %5d, epoch %d: %.3f' % (i, epoch, kl_loss.item()))
            writer.add_scalar('Loss/train', loss.item(), i)
            writer.add_scalar('KL_loss/train', kl_loss.item(), i)
            writer.add_scalar('Recon_loss/train', recon_loss.item(), i)

        if i % 500 == 0:
            torch.save(model.state_dict(), path)
            print("saved session to ", path)

        i+=1

    # eval
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    for data in eval_dataloader:
        occ_grid_t, start_t, goal_t, samples_t = data
        occ_grid_t = occ_grid_t.to(device)
        start_t = start_t.to(device)
        goal_t = goal_t.to(device)
        samples_t = samples_t.to(device)
        context_t = torch.cat((start_t, goal_t), dim=-1)

        # Perform forward pass
        recon_samples, means, log_var, z = model(samples_t, occ_grid_t, context_t)

        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(recon_samples, samples_t, means, log_var)

        total_loss += loss.item()
        total_kl_loss += kl_loss.item()
        total_recon_loss += recon_loss.item()

    print("Evaluation----")
    print('Loss after epoch %5d: %.3f' % (epoch, total_loss / len(eval_dataloader)))
    print('recon after epoch %5d: %.3f' % (epoch, total_recon_loss / len(eval_dataloader)))
    print('kl_loss after epoch %5d: %.3f' % (epoch, total_kl_loss / len(eval_dataloader)))
    writer.add_scalar('Loss/test', total_loss / len(eval_dataloader), epoch)
    writer.add_scalar('KL_loss/test', total_kl_loss / len(eval_dataloader), epoch)
    writer.add_scalar('Recon_loss/test', total_recon_loss / len(eval_dataloader), epoch)