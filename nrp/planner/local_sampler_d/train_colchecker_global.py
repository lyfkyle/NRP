import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
import pickle

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score

from model import ColChecker, ColCheckerSmall
import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))
COL_PRED_THRESHOLD = 0.5
USE_WEIGHTED_SAMPLER = False

class MyDataset(Dataset):
    def __init__(self, data_dir, dataset_size, transform=None, target_transform=None, device="cpu"):
        print("Instantiating dataset...")
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self.dataset_size = dataset_size
        self.fk = utils.FkTorch(device)
        self._data_dir = data_dir
        # print("dataset size = {}".format(len(self.dataset)))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        file_path = osp.join(self._data_dir, "data_{}.pkl".format(idx))
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        occ_grid, start_pos, _, sample, label, _ = data

        start_pos = utils.normalize_state(start_pos)
        sample = utils.normalize_state(sample)

        start_pos = torch.Tensor(start_pos)
        sample = torch.Tensor(sample)
        occ_grid_t = torch.Tensor(occ_grid).view(1, occ_grid_dim, occ_grid_dim)
        label_t = torch.Tensor([label])

        linkinfo = self.fk.get_link_positions(start_pos.view(1, -1)).view(-1)
        start_t = torch.cat((start_pos, linkinfo))
        linkinfo = self.fk.get_link_positions(sample.view(1, -1)).view(-1)
        sample_t = torch.cat((sample, linkinfo), dim=-1)

        return occ_grid_t, start_t, sample_t, label_t

if __name__ == '__main__':
    bce_loss = torch.nn.BCELoss()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--name', default='1')
    parser.add_argument('--checkpoint', default='')
    args = parser.parse_args()

    # constants
    robot_dim = 8
    linkpos_dim = 12
    occ_grid_dim = 100
    model_name = "model_col_global"
    model_path = osp.join(CUR_DIR, "models/{}.pt".format(model_name))
    data_dir = osp.join(CUR_DIR, "./dataset/model_col_global")
    eval_data_dir = osp.join(CUR_DIR, "./dataset/model_col_eval_global")

    # hyperparameters
    bs = 128
    lr = 0.001
    num_epochs = 20
    if USE_WEIGHTED_SAMPLER:
        data_cnt = 4478674
        pos_data_cnt = 1115753
        neg_data_cnt = 3362921
    else:
        data_cnt = 100000
    eval_data_cnt = 10000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    writer = SummaryWriter(comment = model_name)

    # define networks
    print("robot_dim = ", robot_dim, "linkpos_dim = ", linkpos_dim)
    model = ColCheckerSmall(robot_dim + linkpos_dim, occ_grid_dim)
    if args.checkpoint != '':
        print("Loading checkpoint {}.pt".format(args.checkpoint))
        model.load_state_dict(torch.load(osp.join(CUR_DIR, 'models/{}.pt'.format(args.checkpoint))))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_parallel = torch.nn.DataParallel(model)
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

    if USE_WEIGHTED_SAMPLER:
        class_sample_count = np.array([pos_data_cnt, neg_data_cnt])
        weight = 1. / class_sample_count
        samples_weight = [weight[0]] * pos_data_cnt + [weight[1]] * neg_data_cnt
        print(weight, len(samples_weight))
        samples_weight = np.array(samples_weight)
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        dataset = MyDataset(data_dir, data_cnt)
        dataloader = DataLoader(dataset, batch_size=bs, sampler=sampler, drop_last=True, num_workers=10, pin_memory=True)
    else:
        dataset = MyDataset(data_dir, data_cnt)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=10, pin_memory=True)
    eval_dataset = MyDataset(eval_data_dir, eval_data_cnt, None, None)
    eval_dataloader = DataLoader(eval_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=10, pin_memory=True)
    for iter_num in range(num_epochs):
        print("train")
        model.train()
        model_parallel.train()

        all_labels = []
        all_preds = []
        total_loss = 0
        iter_num_tmp = 0
        for data in dataloader:
            occ_grid_t, v1_t, v2_t, label_t = data
            v1_t = v1_t.to(device)
            v2_t = v2_t.to(device)
            occ_grid_t = occ_grid_t.to(device)
            label_t = label_t.to(device)

            # Perform forward pass
            col_scores = model_parallel(occ_grid_t, v1_t, v2_t, fixed_env=False)

            # calculate loss
            loss = bce_loss(col_scores, label_t)

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
                print('Loss after epoch %d, mini-batch %5d, dataset_sizes %d: , col_loss: %.3f' % (iter_num, train_num, data_cnt, loss.item()))
                writer.add_scalar('col_loss/train', loss.item(), train_num)

            train_num += 1
            iter_num_tmp += 1

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), osp.join(CUR_DIR, "models/{}_best.pt".format(model_name)))

            label_list = label_t.detach().cpu().numpy().reshape(-1).tolist()
            all_labels += label_list
            pred_list = col_scores.detach().cpu().numpy().reshape(-1).tolist()
            all_preds += pred_list

            total_loss += loss.item()

        total_loss /= iter_num_tmp
        print('total_loss: %.3f' % (total_loss))

        preds_binary = (np.array(all_preds) > COL_PRED_THRESHOLD).astype(int)
        accuracy = accuracy_score(np.array(all_labels), preds_binary)
        precision = precision_score(np.array(all_labels), preds_binary)
        recall = recall_score(np.array(all_labels), preds_binary)
        print("iter_num: {}, dataset_col_accuracy : {}, dataset_col_precision : {}, dataset_col_recall: {} ".format(iter_num, accuracy, precision, recall))
        writer.add_scalar('dataset_col_accuracy/iter', accuracy, iter_num)
        writer.add_scalar('dataset_col_precision/iter', precision, iter_num)
        writer.add_scalar('dataset_col_recall/iter', recall, iter_num)
        writer.add_scalar('dataset_loss/iter', total_loss, iter_num)

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
            occ_grid_t, v1_t, v2_t, label_t = data
            v1_t = v1_t.to(device)
            v2_t = v2_t.to(device)
            occ_grid_t = occ_grid_t.to(device)
            label_t = label_t.to(device)

            # Perform forward pass
            with torch.no_grad():
                col_scores = model_parallel(occ_grid_t, v1_t, v2_t, fixed_env=False)

            # calculate loss
            loss = bce_loss(col_scores, label_t)

            label_list = label_t.detach().cpu().numpy().reshape(-1).tolist()
            all_labels += label_list
            pred_list = col_scores.detach().cpu().numpy().reshape(-1).tolist()
            all_preds += pred_list
            total_loss += loss.item()
            iter_num_tmp += 1

        total_loss /= iter_num_tmp
        print('total_loss: %.3f' % (total_loss))

        preds_binary = (np.array(all_preds) > COL_PRED_THRESHOLD).astype(int)
        accuracy = accuracy_score(np.array(all_labels), preds_binary)
        precision = precision_score(np.array(all_labels), preds_binary)
        recall = recall_score(np.array(all_labels), preds_binary)
        print("iter_num: {}, eval_col_accuracy : {}, eval_col_precision : {}, eval_col_recall: {} ".format(iter_num, accuracy, precision, recall))
        writer.add_scalar('eval_col_accuracy/iter', accuracy, iter_num)
        writer.add_scalar('eval_col_precision/iter', precision, iter_num)
        writer.add_scalar('eval_col_recall/iter', recall, iter_num)
        writer.add_scalar('eval_loss/iter', total_loss, iter_num)

    writer.close()