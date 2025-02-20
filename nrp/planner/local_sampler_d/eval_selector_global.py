import os.path as osp
import sys
import os
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import pickle
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from model import Selector, DiscriminativeSampler
import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))

SEL_PRED_THRESHOLD = 0.5

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

        occ_grid, start_pos, goal_pos, sample, _, label = data

        start_pos = utils.normalize_state(start_pos)
        goal_pos = utils.normalize_state(goal_pos)
        sample = utils.normalize_state(sample)

        start_pos = torch.Tensor(start_pos)
        goal_pos = torch.Tensor(goal_pos)
        sample = torch.Tensor(sample)
        occ_grid_t = torch.Tensor(occ_grid).view(1, occ_grid_dim, occ_grid_dim)
        label_t = torch.Tensor([label])

        linkinfo = self.fk.get_link_positions(start_pos.view(1, -1)).view(-1)
        start_t = torch.cat((start_pos, linkinfo))
        linkinfo = self.fk.get_link_positions(goal_pos.view(1, -1)).view(-1)
        goal_direction = torch.atan2(goal_pos[1], goal_pos[0]).view(1)
        goal_t = torch.cat((goal_pos, linkinfo, goal_direction))
        linkinfo = self.fk.get_link_positions(sample.view(1, -1)).view(-1)
        sample_t = torch.cat((sample, linkinfo), dim=-1)

        return occ_grid_t, start_t, goal_t, sample_t, label_t

if __name__ == '__main__':
    # mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCELoss()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--name', default='1')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--data', default='test')
    args = parser.parse_args()

    # constants
    robot_dim = 8
    linkpos_dim = 12
    occ_grid_dim = 100
    goal_dim = robot_dim + linkpos_dim + 1

    if args.data == "train":
        model_name = "model_sel_global"
        data_cnt = 500000
    elif args.data == "eval":
        model_name = "model_eval_global"
        data_cnt = 226662
    elif args.data == "test":
        model_name = "model_test_global"
        data_cnt = 68685

    data_dir = osp.join(CUR_DIR, "./dataset/{}".format(model_name))

    # hyperparameters
    bs = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # define networks
    print("robot_dim = ", robot_dim, "linkinfo_dim = ", linkpos_dim)
    model = DiscriminativeSampler(robot_dim, occ_grid_dim, os.path.join(CUR_DIR, "models/model_col_global.pt"), osp.join(CUR_DIR, 'models/{}.pt'.format(args.checkpoint)))
    print("Loading checkpoint {}.pt".format(args.checkpoint))

    model.to(device)
    model.eval()

    dataset = MyDataset(data_dir, data_cnt, None, None)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=40, pin_memory=True)

    all_labels = []
    all_preds = []
    eval_total_loss = 0
    iter_num = 0
    for data in dataloader:
        occ_grid, start, goal, sample, label = data

        start = start.to(device)
        goal = goal.to(device)
        occ_grid = occ_grid.to(device)
        sample = sample.to(device)
        label = label.to(device)

        # Perform forward pass
        with torch.no_grad():
            # sel_scores = model.sel_model(occ_grid, start, goal, sample, fixed_env=False)
            sel_scores = model.get_final_sel_scores(occ_grid, start, goal, sample)

        # calculate loss
        loss = bce_loss(sel_scores, label)

        eval_total_loss += loss.item()

        label_list = label.detach().cpu().numpy().reshape(-1).tolist()
        all_labels += label_list
        pred_list = sel_scores.detach().cpu().numpy().reshape(-1).tolist()
        all_preds += pred_list

        iter_num += 1

    # Print statistics
    eval_total_loss /= iter_num
    print('sel_pred: %.3f' % (eval_total_loss))

    preds_binary = (np.array(all_preds) > SEL_PRED_THRESHOLD).astype(int)
    labels_binary = (np.array(all_labels) > 0).astype(int)
    pos_lables = np.flatnonzero(labels_binary)
    accuracy = accuracy_score(np.array(all_labels), preds_binary)
    precision = precision_score(np.array(all_labels), preds_binary)
    recall = recall_score(np.array(all_labels), preds_binary)
    print("dataset_eval_accuracy : {}, dataset_eval_precision : {}, dataset_eval_recall: {} ".format(accuracy, precision, recall))
    print("dataset contains {} positive and {} negative samples".format(len(pos_lables), data_cnt - len(pos_lables)))

    cm = confusion_matrix(labels_binary, preds_binary)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("sel_confusion_matrix_global.png")