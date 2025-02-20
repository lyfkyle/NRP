import os.path as osp

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import pickle
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter

from nrp.env.fetch_11d.fk.model import ProxyFkTorch
from nrp.planner.local_sampler_d.model_11d import Selector
from nrp.env.fetch_11d import utils
from nrp import ROOT_DIR

CUR_DIR = osp.dirname(osp.abspath(__file__))

SEL_PRED_THRESHOLD = 0.5
USE_WEIGHTED_SAMPLER = False

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

        occ_grid, start_pos, goal_pos, expert_waypoint_pos, label, _, _ = data
        if label <= 0:
            label = 0

        start_pos = torch.Tensor(start_pos)
        goal_pos = torch.Tensor(goal_pos)
        expert_waypoint_pos = torch.Tensor(expert_waypoint_pos)
        occ_grid = torch.Tensor(occ_grid).view(occ_grid_dim, occ_grid_dim, occ_grid_dim_z)
        occ_grid_t = utils.add_pos_channels(occ_grid)
        label_t = torch.Tensor([label])

        link_pos = self.fk.get_link_positions(start_pos.view(1, -1)).view(-1)
        start_t = torch.cat((start_pos, link_pos))
        link_pos = self.fk.get_link_positions(goal_pos.view(1, -1)).view(-1)
        goal_direction = torch.atan2(goal_pos[1], goal_pos[0]).view(1)
        goal_t = torch.cat((goal_pos, link_pos, goal_direction))
        link_pos = self.fk.get_link_positions(expert_waypoint_pos.view(1, -1)).view(-1)
        expert_wp_pos_t = torch.cat((expert_waypoint_pos, link_pos), dim=-1)

        return occ_grid_t, start_t, goal_t, expert_wp_pos_t, label_t

if __name__ == '__main__':
    bce_loss = torch.nn.BCELoss()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name', default='sampler_g_01_v4')
    parser.add_argument('--checkpoint', default='')
    args = parser.parse_args()

    model_name = 'train_01_critical_v2'
    data_dir = osp.join(CUR_DIR, "dataset/train_01_critical_v2_out_d")

    # constants
    robot_dim = 11
    linkpos_dim = 24
    state_dim = robot_dim + linkpos_dim
    goal_dim = state_dim + 1
    occ_grid_dim = 40
    occ_grid_dim_z = 20
    model_path = osp.join(CUR_DIR, "models/{}.pt".format(model_name))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(comment = "_{}".format(model_name))

    # hyperparameters
    bs = 128
    lr = 1e-3
    num_steps = 10000
    num_epochs = 100
    data_cnt = 625000 # 500000 samples, half pos, half neg

    # define networks
    model = Selector(state_dim, goal_dim)
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
    train_num = 0
    best_loss = float('inf')
    dataset = MyDataset(data_dir, data_cnt)
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=20, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=20, pin_memory=True)

    # check dataset balance
    # pos_cnt = 0
    # neg_cnt = 0
    # i = 0
    # for data in dataloader:
    #     occ_grid, start, goal, sample, label = data
    #     if label[0][0] == 0:
    #         neg_cnt += 1
    #     else:
    #         pos_cnt += 1
    #     i += 1
    #     if i > 5000:
    #         break

    # pos_cnt = 0
    # neg_cnt = 0
    # for data in eval_dataloader:
    #     occ_grid, start, goal, sample, label = data
    #     if label[0] == 0:
    #         neg_cnt += 1
    #     else:
    #         pos_cnt += 1
    # print(pos_cnt, neg_cnt)

    for epoch in range(num_epochs):
        print("train")
        model.train()
        model_parallel.train()

        all_labels = []
        all_preds = []
        total_loss = 0
        iter_num_tmp = 0
        for data in dataloader:
            occ_grid, start, goal, sample, label = data

            start = start.to(device)
            goal = goal.to(device)
            occ_grid = occ_grid.to(device)
            sample = sample.to(device)
            label = label.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            sel_scores = model_parallel(occ_grid, start, goal, sample, fixed_env=False)

            # calculate loss
            loss = bce_loss(sel_scores, label)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()
            scheduler.step(loss)

            # Print statistics
            if train_num % 100 == 0:
                print("Learning rate: {}".format(optimizer.param_groups[0]['lr']))
                print('Loss after epoch %d, mini-batch %5d, dataset_sizes %d: , sel_pred: %.3f' % (epoch, train_num, data_cnt, loss.item()))
                writer.add_scalar('sel_loss/train', loss.item(), train_num)

            train_num += 1
            iter_num_tmp += 1

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), osp.join(CUR_DIR, "models/{}_best.pt".format(model_name)))

            label_list = label.detach().cpu().numpy().reshape(-1).tolist()
            all_labels += label_list
            pred_list = sel_scores.detach().cpu().numpy().reshape(-1).tolist()
            all_preds += pred_list

            total_loss += loss.item()

        total_loss /= iter_num_tmp
        print('total_loss: %.3f' % (total_loss))

        preds_binary = (np.array(all_preds) > SEL_PRED_THRESHOLD).astype(int)
        accuracy = accuracy_score(np.array(all_labels), preds_binary)
        precision = precision_score(np.array(all_labels), preds_binary)
        recall = recall_score(np.array(all_labels), preds_binary)
        print("iter_num: {}, dataset_accuracy : {}, dataset_precision : {}, dataset_recall: {} ".format(epoch, accuracy, precision, recall))
        writer.add_scalar('dataset_accuracy/iter', accuracy, epoch)
        writer.add_scalar('dataset_precision/iter', precision, epoch)
        writer.add_scalar('dataset_recall/iter', recall, epoch)
        writer.add_scalar('dataset_loss/iter', total_loss, epoch)

        torch.save(model.state_dict(), model_path)
        print("saved session to ", model_path)

        print("eval")
        model.eval()
        model_parallel.eval()

        all_labels = []
        all_preds = []
        total_loss = 0
        iter_num_tmp = 0
        for data in eval_dataloader:
            occ_grid, start, goal, sample, label = data

            start = start.to(device)
            goal = goal.to(device)
            occ_grid = occ_grid.to(device)
            sample = sample.to(device)
            label = label.to(device)

            # Perform forward pass
            with torch.no_grad():
                sel_scores = model_parallel(occ_grid, start, goal, sample, fixed_env=False)

            # calculate loss
            loss = bce_loss(sel_scores, label)

            label_list = label.detach().cpu().numpy().reshape(-1).tolist()
            all_labels += label_list
            pred_list = sel_scores.detach().cpu().numpy().reshape(-1).tolist()
            all_preds += pred_list
            total_loss += loss.item()
            iter_num_tmp += 1

        total_loss /= iter_num_tmp
        print('total_loss: %.3f' % (total_loss))

        preds_binary = (np.array(all_preds) > SEL_PRED_THRESHOLD).astype(int)
        accuracy = accuracy_score(np.array(all_labels), preds_binary)
        precision = precision_score(np.array(all_labels), preds_binary)
        recall = recall_score(np.array(all_labels), preds_binary)
        print("iter_num: {}, eval_accuracy : {}, eval_precision : {}, eval_recall: {} ".format(epoch, accuracy, precision, recall))
        writer.add_scalar('eval_accuracy/iter', accuracy, epoch)
        writer.add_scalar('eval_precision/iter', precision, epoch)
        writer.add_scalar('eval_recall/iter', recall, epoch)
        writer.add_scalar('eval_loss/iter', total_loss, epoch)

    writer.close()