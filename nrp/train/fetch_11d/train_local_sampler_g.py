import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import pickle
from torch.utils.tensorboard import SummaryWriter
import time

from nrp.env.fetch_11d.fk.model import ProxyFkTorch
from nrp.planner.local_sampler_g.model_11d import VAE
from nrp.env.fetch_11d import utils
from nrp import ROOT_DIR

CUR_DIR = osp.dirname(osp.abspath(__file__))


class MyDataset(Dataset):
    def __init__(self, data_dir, dataset_size, transform=None, target_transform=None, device="cpu"):
        print("Instantiating dataset...")
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self.dataset_size = dataset_size

        fkmodel_path = osp.join(ROOT_DIR, "env/fetch_11d/fk/models/model_fk_v2.pt")
        self.fk = ProxyFkTorch(robot_dim, linkpos_dim, fkmodel_path, device)
        self._data_dir = data_dir

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        file_path = osp.join(self._data_dir, "data_{}.pkl".format(idx))
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        occ_grid, start_pos, goal_pos, expert_waypoint_pos, _, _ = data
        # expert_waypoint_pos = expert_path[1]

        start_pos = torch.Tensor(start_pos)
        goal_pos = torch.Tensor(goal_pos)
        expert_waypoint_pos = torch.Tensor(expert_waypoint_pos)
        occ_grid = torch.Tensor(occ_grid).view(occ_grid_dim, occ_grid_dim, occ_grid_dim_z)
        occ_grid_t = utils.add_pos_channels(occ_grid)

        link_pos = self.fk.get_link_positions(start_pos.view(1, -1)).view(-1)
        start_t = torch.cat((start_pos, link_pos))
        link_pos = self.fk.get_link_positions(goal_pos.view(1, -1)).view(-1)
        goal_direction = torch.atan2(goal_pos[1], goal_pos[0]).view(1)
        goal_t = torch.cat((goal_pos, link_pos, goal_direction))
        link_pos = self.fk.get_link_positions(expert_waypoint_pos.view(1, -1)).view(-1)
        expert_wp_pos_t = torch.cat((expert_waypoint_pos, link_pos), dim=-1)

        return occ_grid_t, start_t, goal_t, expert_wp_pos_t

def vae_loss(recon_x, x, mean, log_var, alpha=1.0):
    bs = recon_x.shape[0]
    BCE = torch.nn.functional.mse_loss(recon_x.view(bs, -1), x.view(bs, -1), reduction="mean")
    KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1), dim=0)

    return BCE + alpha * KLD, BCE, KLD


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name', default='sampler_g_01_v4')
    parser.add_argument('--checkpoint', default='')
    args = parser.parse_args()

    model_name = 'train_01_critical_v2'
    data_dir = osp.join(CUR_DIR, "dataset/train_01_critical_v2_out_g")

    # constants
    z_dim = 8
    robot_dim = 11
    linkpos_dim = 24
    state_dim = robot_dim + linkpos_dim
    goal_dim = state_dim + 1
    occ_grid_dim = 40
    occ_grid_dim_z = 20
    model_path = osp.join(CUR_DIR, "models/{}.pt".format(model_name))
    # data_dir = osp.join(CUR_DIR, "dataset/fetch_11d_waypoint")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(comment = "_{}".format(model_name))

    # hyperparameters
    bs = 128
    lr = 1e-3
    num_steps = 10000
    num_epochs = 100
    alpha = 0.01
    # data_cnt = 19281
    # data_cnt = 35103
    data_cnt = 250000
    eval_data_cnt = 8560

    # define networks
    print("dim = ", robot_dim)
    print("z_dim = ", z_dim)
    model = VAE(z_dim, state_dim + goal_dim, state_dim)
    if args.checkpoint != '':
        print("Loading checkpoint {}.pt".format(args.checkpoint))
        model.load_state_dict(torch.load(osp.join(CUR_DIR, 'models/{}.pt'.format(args.checkpoint))))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_parallel = torch.nn.DataParallel(model)
    else:
        model_parallel = model
    model_parallel.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    # Define the loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10000, verbose=True, factor=0.5)

    # dataset and dataloader
    dataset = MyDataset(data_dir, data_cnt)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=10, pin_memory=True)
    # eval_dataset = MyDataset(eval_data_dir, eval_data_cnt, None, None)
    # eval_dataloader = DataLoader(eval_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=10, pin_memory=True)

    # Run the training loop
    i = 0
    for epoch in range(num_epochs):
        model.train()
        for data in dataloader:
            occ_grid_t, start_t, goal_t, expert_wp_t = data
            occ_grid_t = occ_grid_t.to(device)
            start_t = start_t.to(device)
            goal_t = goal_t.to(device)
            expert_wp_t = expert_wp_t.to(device)
            context_t = torch.cat((start_t, goal_t), dim=-1)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            # start_time = time.time()
            recon_samples, means, log_var, z = model_parallel(expert_wp_t, occ_grid_t, context_t)
            # end_time = time.time()
            # print(end_time - start_time)

            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(recon_samples, expert_wp_t, means, log_var, alpha)

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
                torch.save(model.state_dict(), model_path)
                print("saved session to ", model_path)

            i+=1

        # eval
        # model.eval()
        # total_loss = 0
        # total_recon_loss = 0
        # total_kl_loss = 0
        # for data in eval_dataloader:
        #     occ_grid_t, start_t, goal_t, samples_t = data
        #     occ_grid_t = occ_grid_t.to(device)
        #     start_t = start_t.to(device)
        #     goal_t = goal_t.to(device)
        #     samples_t = samples_t.to(device)
        #     context_t = torch.cat((start_t, goal_t), dim=-1)

        #     # Perform forward pass
        #     recon_samples, means, log_var, z = model(samples_t, occ_grid_t, context_t)

        #     # Compute loss
        #     loss, recon_loss, kl_loss = vae_loss(recon_samples, samples_t, means, log_var)

        #     total_loss += loss.item()
        #     total_kl_loss += kl_loss.item()
        #     total_recon_loss += recon_loss.item()

        # print("Evaluation----")
        # print('Loss after epoch %5d: %.3f' % (epoch, total_loss / len(eval_dataloader)))
        # print('recon after epoch %5d: %.3f' % (epoch, total_recon_loss / len(eval_dataloader)))
        # print('kl_loss after epoch %5d: %.3f' % (epoch, total_kl_loss / len(eval_dataloader)))
        # writer.add_scalar('Loss/test', total_loss / len(eval_dataloader), epoch)
        # writer.add_scalar('KL_loss/test', total_kl_loss / len(eval_dataloader), epoch)
        # writer.add_scalar('Recon_loss/test', total_recon_loss / len(eval_dataloader), epoch)