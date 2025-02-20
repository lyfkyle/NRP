import os.path as osp
import sys
import os
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
import pickle

from model import SelectModel
import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))

SEL_PRED_THRESHOLD = 0.5
COL_PRED_THRESHOLD = 0.5

class MyDataset(Dataset):
    def __init__(self, data_dir, dataset_size, transform=None, target_transform=None, device="cpu"):
        print("Instantiating dataset...")
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self.dataset_size = dataset_size
        self.data_dir = data_dir

        self.fk = utils.FkTorch(device)
        # self.dataset = self.load_dataset_from_file()

        # print("dataset size = {}".format(len(self.dataset)))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        file_path = osp.join(self.data_dir, "data_{}.pkl".format(idx))
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        occ_grid, start_pos, goal_pos, sample, col_label, sel_label = data

        start_pos = torch.Tensor(start_pos)
        goal_pos = torch.Tensor(goal_pos)
        sample = torch.Tensor(sample)
        occ_grid_t = torch.Tensor(occ_grid).view(1, occ_grid_dim, occ_grid_dim)
        col_label_t = torch.Tensor([col_label])
        sel_label_t = torch.Tensor([sel_label])
        # self_col_label = torch.Tensor([self_col_label])

        linkinfo = self.fk.get_link_positions(start_pos.view(1, -1)).view(-1)
        start_t = torch.cat((start_pos, linkinfo))
        linkinfo = self.fk.get_link_positions(goal_pos.view(1, -1)).view(-1)
        goal_direction = torch.atan2(goal_pos[1], goal_pos[0]).view(1)
        goal_t = torch.cat((goal_pos, linkinfo, goal_direction))
        linkinfo = self.fk.get_link_positions(sample.view(1, -1)).view(-1)
        sample_t = torch.cat((sample, linkinfo), dim=-1)

        return occ_grid_t, start_t, goal_t, sample_t, col_label_t, sel_label_t

if __name__ == '__main__':
    # mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCELoss()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--name', default='model_integrated')
    parser.add_argument('--datadir', default='model')
    parser.add_argument('--checkpoint', default='')
    args = parser.parse_args()

    # constants
    robot_dim = 8
    linkpos_dim = 12
    occ_grid_dim = 40
    goal_dim = robot_dim + linkpos_dim + 1
    model_name = args.name
    model_path = osp.join(CUR_DIR, "models/{}.pt".format(model_name))
    data_dir = osp.join(CUR_DIR, "./dataset/model_integrated")
    eval_data_dir = osp.join(CUR_DIR, "./dataset/{}_eval".format(args.datadir))

    # hyperparameters
    bs = 128
    lr = 0.001
    # num_steps = 10000s
    num_epochs = 50
    max_iter = 100
    data_cnt = 1079820
    eval_data_cnt = 66351

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    writer = SummaryWriter(comment = '_{}'.format(model_name))

    # define networks
    print("robot_dim = ", robot_dim, "linkinfo_dim = ", linkpos_dim)
    model = SelectModel(robot_dim, occ_grid_dim)
    if args.checkpoint != '':
        print("Loading checkpoint {}.pt".format(args.checkpoint))
        model.load_state_dict(torch.load(osp.join(CUR_DIR, 'models/{}.pt'.format(args.checkpoint))))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_parallel = torch.nn.DataParallel(model)
    model.to(device)
    model_parallel.to(device)
    torch.save(model.state_dict(), model_path)

    # Define the loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15000, verbose=True, factor=0.5)

    train_num = 0
    iter_num = 0
    best_loss = float('inf')

    # Collect enough amount of data, just train
    dataset = MyDataset(data_dir, data_cnt, None, None)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=10, pin_memory=True)
    eval_dataset = MyDataset(eval_data_dir, eval_data_cnt, None, None)
    eval_dataloader = DataLoader(eval_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=10, pin_memory=True)
    for iter_num in range(num_epochs):
        print("train")
        model.train()
        model_parallel.train()

        col_all_labels = []
        col_all_preds = []
        sel_all_labels = []
        sel_all_preds = []
        total_loss = 0
        iter_num_tmp = 0
        for data in dataloader:
            occ_grid, start, goal, sample, col_label, sel_label = data

            # print(samples)
            start = start.to(device)
            goal = goal.to(device)
            occ_grid = occ_grid.to(device)
            sample = sample.to(device)
            col_label = col_label.to(device)
            sel_label = sel_label.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            # sel_scores = model_parallel(occ_grid_batch, start_batch, goal_batch, samples_batch)
            col_scores, sel_scores = model_parallel(occ_grid, start, goal, sample)
            # sel_scores = sel_scores.view(occ_grid.shape[0], -1)

            # calculate loss
            col_loss = bce_loss(col_scores, col_label)
            sel_loss = bce_loss(sel_scores, sel_label)
            loss = col_loss + sel_loss

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()
            scheduler.step(loss)

            # Print statistics
            if train_num % 100 == 0:
                print("Learning rate: {}".format(optimizer.param_groups[0]['lr']))
                print('Loss after epoch %d, mini-batch %5d, dataset_sizes %d: , col_loss: %.3f, sel_loss: %.3f' % (iter_num, train_num, data_cnt, col_loss.item(), sel_loss.item()))
                writer.add_scalar('col_loss/train', col_loss.item(), train_num)
                writer.add_scalar('sel_loss/train', sel_loss.item(), train_num)

            train_num += 1

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), osp.join(CUR_DIR, "models/{}_best.pt".format(model_name)))

            col_label_list = col_label.detach().cpu().numpy().reshape(-1).tolist()
            col_all_labels += col_label_list
            pred_list = col_scores.detach().cpu().numpy().reshape(-1).tolist()
            col_all_preds += pred_list
            sel_label_list = sel_label.detach().cpu().numpy().reshape(-1).tolist()
            sel_all_labels += sel_label_list
            pred_list = sel_scores.detach().cpu().numpy().reshape(-1).tolist()
            sel_all_preds += pred_list

        preds_binary = (np.array(col_all_preds) > COL_PRED_THRESHOLD).astype(int)
        accuracy = accuracy_score(np.array(col_all_labels), preds_binary)
        precision = precision_score(np.array(col_all_labels), preds_binary)
        recall = recall_score(np.array(col_all_labels), preds_binary)
        print("iter_num: {}, dataset_col_accuracy : {}, dataset_col_precision : {}, dataset_col_recall: {} ".format(iter_num, accuracy, precision, recall))
        writer.add_scalar('dataset_col_accuracy/iter', accuracy, train_num)
        writer.add_scalar('dataset_col_precision/iter', precision, train_num)
        writer.add_scalar('dataset_col_recall/iter', recall, train_num)

        preds_binary = (np.array(sel_all_preds) > SEL_PRED_THRESHOLD).astype(int)
        accuracy = accuracy_score(np.array(sel_all_labels), preds_binary)
        precision = precision_score(np.array(sel_all_labels), preds_binary)
        recall = recall_score(np.array(sel_all_labels), preds_binary)
        print("iter_num: {}, dataset_sel_accuracy : {}, dataset_sel_precision : {}, dataset_sel_recall: {} ".format(iter_num, accuracy, precision, recall))
        writer.add_scalar('dataset_sel_accuracy/iter', accuracy, train_num)
        writer.add_scalar('dataset_sel_precision/iter', precision, train_num)
        writer.add_scalar('dataset_sel_recall/iter', recall, train_num)

        torch.save(model.state_dict(), model_path)
        print("saved session to ", model_path)

        print("eval")
        model.eval()
        model_parallel.eval()

        col_all_labels = []
        col_all_preds = []
        sel_all_labels = []
        sel_all_preds = []
        total_loss = 0
        iter_num_tmp = 0
        for data in eval_dataloader:
            occ_grid, start, goal, sample, col_label, sel_label = data

            # print(samples)
            start = start.to(device)
            goal = goal.to(device)
            occ_grid = occ_grid.to(device)
            sample = sample.to(device)
            col_label = col_label.to(device)
            sel_label = sel_label.to(device)

            # Perform forward pass
            # sel_scores = model_parallel(occ_grid_batch, start_batch, goal_batch, samples_batch)
            with torch.no_grad():
                col_scores, sel_scores = model_parallel(occ_grid, start, goal, sample)
            # sel_scores = sel_scores.view(occ_grid.shape[0], -1)

            # calculate loss
            col_loss = bce_loss(col_scores, col_label)
            sel_loss = bce_loss(sel_scores, sel_label)
            loss = col_loss + sel_loss

            # Print statistics
            total_loss += loss.item()
            iter_num_tmp += 1

            col_label_list = col_label.detach().cpu().numpy().reshape(-1).tolist()
            col_all_labels += col_label_list
            pred_list = col_scores.detach().cpu().numpy().reshape(-1).tolist()
            col_all_preds += pred_list
            sel_label_list = sel_label.detach().cpu().numpy().reshape(-1).tolist()
            sel_all_labels += sel_label_list
            pred_list = sel_scores.detach().cpu().numpy().reshape(-1).tolist()
            sel_all_preds += pred_list

        total_loss /= iter_num_tmp
        print('total_loss: %.3f' % (total_loss))
        writer.add_scalar('eval_loss/iter', recall, iter_num)

        preds_binary = (np.array(col_all_preds) > COL_PRED_THRESHOLD).astype(int)
        accuracy = accuracy_score(np.array(col_all_labels), preds_binary)
        precision = precision_score(np.array(col_all_labels), preds_binary)
        recall = recall_score(np.array(col_all_labels), preds_binary)
        print("iter_num: {}, eval_col_accuracy : {}, eval_col_precision : {}, eval_col_recall: {} ".format(iter_num, accuracy, precision, recall))
        writer.add_scalar('eval_col_accuracy/iter', accuracy, iter_num)
        writer.add_scalar('eval_col_precision/iter', precision, iter_num)
        writer.add_scalar('eval_col_recall/iter', recall, iter_num)

        preds_binary = (np.array(sel_all_preds) > SEL_PRED_THRESHOLD).astype(int)
        accuracy = accuracy_score(np.array(sel_all_labels), preds_binary)
        precision = precision_score(np.array(sel_all_labels), preds_binary)
        recall = recall_score(np.array(sel_all_labels), preds_binary)
        print("iter_num: {}, eval_sel_accuracy : {}, eval_sel_precision : {}, eval_sel_recall: {} ".format(iter_num, accuracy, precision, recall))
        writer.add_scalar('eval_sel_accuracy/iter', accuracy, iter_num)
        writer.add_scalar('eval_sel_precision/iter', precision, iter_num)
        writer.add_scalar('eval_sel_recall/iter', recall, iter_num)

    writer.close()