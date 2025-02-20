import os.path as osp
import sys
import os
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import pickle
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score

from model import ColChecker, ColCheckerSmall
import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))

LOCAL_ENV_SIZE = 2.0
COL_PRED_THRESHOLD = 0.5
best_eval_loss = float('inf')

def get_datas(train_env_dirs, env_idx, dataset_idx):
    file_path = osp.join(train_env_dirs, "data_{}_{}.pkl".format(env_idx, dataset_idx))
    if not os.path.exists(file_path):
        return []

    with open(file_path, 'rb') as f:
        datas = pickle.load(f)

    free_samples = []
    col_samples = []
    sel_lables = []
    col_labels = []
    for i, data in enumerate(datas):
        occ_grid, start_pos, goal_pos, sample_pos, expert_pos, selected_path_len, expert_path_len = data

        if selected_path_len == -1:
            col_samples.append(sample_pos)
        else:
            free_samples.append((sample_pos, selected_path_len))

    free_samples.sort(key=lambda a: a[1])

    num_pos = max(1, int(len(free_samples) * 0.25))
    neg_idx = max(1, int(len(free_samples) * 0.5))
    pos_s = free_samples[:num_pos]
    if neg_idx < len(free_samples):
        neg_s = free_samples[neg_idx:]
        neutral_s = free_samples[num_pos:neg_idx]
    else:
        neg_s = []
        neutral_s = []

    all_samples = []
    for sample_pos, _ in pos_s:
        all_samples.append(sample_pos)
        sel_lables.append(1)
        col_labels.append(1)
    for sample_pos, _ in neg_s:
        all_samples.append(sample_pos)
        sel_lables.append(0)
        col_labels.append(1)
    # for sample_pos, _ in neural_s:
    #     neutral_samples.append([occ_grid, start_pos, goal_pos, sample_pos])

    for sample_pos in col_samples:
        all_samples.append(sample_pos)
        sel_lables.append(0)
        col_labels.append(0)

    data_list = [occ_grid, start_pos, goal_pos, all_samples, col_labels, sel_lables]

    return data_list

def collect_gt(train_env_dirs, env_idx, output_data_list, num_samples_per_env=100):
    cnt = 0
    while cnt < num_samples_per_env:
        dataset_idx = random.randint(0, 2800)
        data_list = get_datas(train_env_dirs, env_idx, dataset_idx)
        if len(data_list) == 0:
            continue

        print("collected data from env {} and local_env {}".format(env_idx, dataset_idx))

        output_data_list.append(data_list)
        cnt += 1

def save_dataset(dataset, data_cnt, start_idx, end_idx):
    print("Saving dataset {} to {}".format(start_idx, end_idx))
    for idx in range(start_idx, end_idx):
        file_path = osp.join(data_dir, "data_{}.pkl".format(data_cnt + idx))
        with open(file_path, 'wb') as f:
            # print("Dumping to {}".format(file_path))
            pickle.dump(dataset[idx], f)

def collect_mistakes(dataset_list, colchecker, iter_num, balance=False):
    fk = utils.FkTorch(device)

    col_total_true_pos = 0
    col_total_true_neg = 0
    col_total_false_pos = 0
    col_total_false_neg = 0
    mistake_dataset = []
    for i, data in enumerate(dataset_list):
        local_occ_grid, local_start_pos, local_goal_pos, local_sample_pos, col_label, sel_label = data

        with torch.no_grad():
            occ_grid_t = torch.tensor(local_occ_grid, device=device, dtype=torch.float).view(1, occ_grid_dim, occ_grid_dim)
            start_t = torch.tensor(local_start_pos, device=device, dtype=torch.float)
            goal_t = torch.tensor(local_goal_pos, device=device, dtype=torch.float)
            samples_t = torch.tensor(local_sample_pos, device=device, dtype=torch.float)

            linkpos = fk.get_link_positions(start_t.view(1, -1)).view(-1)
            start_t = torch.cat((start_t, linkpos))
            linkpos = fk.get_link_positions(goal_t.view(1, -1)).view(-1)
            goal_direction = torch.atan2(goal_t[1], goal_t[0]).view(1)
            goal_t = torch.cat((goal_t, linkpos, goal_direction))
            linkpos = fk.get_link_positions(samples_t)
            samples_t = torch.cat((samples_t, linkpos), dim=-1)

            occ_grid_batch = occ_grid_t.unsqueeze(0) # 1 x 4 x occ_grid_dim x occ_grid_dim x occ_grid_dim_z
            start_batch = start_t.unsqueeze(0) # 1 x dim
            goal_batch = goal_t.unsqueeze(0) # 1 x dim
            samples_batch = samples_t.unsqueeze(0) # 1 x N x dim
            col_scores = colchecker(occ_grid_batch, start_batch, samples_batch, fixed_env=True)
            col_scores = col_scores.view(-1)

        # collecting data from mis-classifications
        col_false_neg_dataset = []
        col_true_neg_dataset = []
        col_false_pos_dataset = []
        col_true_pos_dataset = []
        print(len(col_scores), len(col_label), len(local_sample_pos))
        for i, score in enumerate(col_scores):
            if col_label[i] == 1:
                if score < COL_PRED_THRESHOLD:
                    col_false_neg_dataset.append([local_occ_grid, local_start_pos, local_goal_pos, local_sample_pos[i], col_label[i], sel_label[i]])
                else:
                    col_true_pos_dataset.append([local_occ_grid, local_start_pos, local_goal_pos, local_sample_pos[i], col_label[i], sel_label[i]])

            elif col_label[i] == 0:
                if score > COL_PRED_THRESHOLD:
                    col_false_pos_dataset.append([local_occ_grid, local_start_pos, local_goal_pos, local_sample_pos[i], col_label[i], sel_label[i]])
                else:
                    col_true_neg_dataset.append([local_occ_grid, local_start_pos, local_goal_pos, local_sample_pos[i], col_label[i], sel_label[i]])

        random.shuffle(col_false_neg_dataset)
        random.shuffle(col_true_neg_dataset)
        random.shuffle(col_false_pos_dataset)
        random.shuffle(col_true_pos_dataset)

        neg_dataset = col_false_pos_dataset
        pos_dataset = col_false_neg_dataset
        true_pos_dataset = col_true_pos_dataset
        true_neg_dataset = col_true_neg_dataset

        # balance positive to negative
        if balance:
            # maximum 50 negative data
            if len(neg_dataset) >= 50:
                neg_dataset = neg_dataset[:50]

            # maximum 50 positive data
            if len(pos_dataset) >= 50:
                pos_dataset = pos_dataset[:50]

            if len(pos_dataset) >= len(neg_dataset):
                neg_dataset = neg_dataset + true_neg_dataset[:min(len(true_neg_dataset), len(pos_dataset) - len(neg_dataset))]
            else:
                pos_dataset = pos_dataset + true_pos_dataset[:min(len(true_pos_dataset), len(neg_dataset) - len(pos_dataset))]

        print(len(pos_dataset), len(neg_dataset), len(true_pos_dataset), len(true_neg_dataset))
        mistake_dataset += (pos_dataset + neg_dataset)

        col_total_true_pos += len(col_true_pos_dataset)
        col_total_true_neg += len(col_true_neg_dataset)
        col_total_false_pos += len(col_false_pos_dataset)
        col_total_false_neg += len(col_false_neg_dataset)

    col_accuracy, col_precison, col_recall = utils.calculate_stats(col_total_true_pos, col_total_true_neg, col_total_false_pos, col_total_false_neg)
    print("Iteration: {}, col_accuracy : {}, col_precision : {}, col_recall: {} ".format(iter_num, col_accuracy, col_precison, col_recall))
    writer.add_scalar('col_accuracy/iter', col_accuracy, iter_num)
    writer.add_scalar('col_precision/iter', col_precison, iter_num)
    writer.add_scalar('col_recall/iter', col_recall, iter_num)

    return mistake_dataset

def start_collect_gt_process(train_env_dir, dataset_list):
    print("Collecting gt")
    processes = []
    for i in range(25):
        p = mp.Process(target=collect_gt, args=(train_env_dir, i, dataset_list, 10), daemon=True)
        p.start()
        processes.append(p)

    return processes

def collect_data_from_train_env(train_env_dirs, data_cnt, iter_num, dataset_list):
    # Collect data from train env
    print("Collecting gt")

    # Wait for previous process to finish
    processes = start_collect_gt_process(train_env_dirs, dataset_list)
    for p in processes:
        p.join()

    print("Collecting mistakes")
    colchecker = ColCheckerSmall(robot_dim + linkpos_dim)
    colchecker.load_state_dict(torch.load(model_path))
    colchecker.to(device)
    colchecker.eval()
    dataset = collect_mistakes(dataset_list, colchecker, iter_num, balance=first_time)

    # Save Dataset
    print("Iteration: {}, saving dataset of size {}".format(iter_num, len(dataset)))
    j = 0
    process_num = min(40, len(dataset))
    step_size = len(dataset) // process_num
    processes = []
    while j < len(dataset):
        p = mp.Process(target=save_dataset, args=(dataset, data_cnt, j, min(len(dataset), j + step_size)), daemon=True)
        p.start()
        processes.append(p)
        j += step_size

    for p in processes:
        p.join()

    data_cnt += len(dataset)
    writer.add_scalar('dataset_size', data_cnt, iter_num)
    print(data_cnt)

    return data_cnt

def train(data_cnt, epoch_num, iter_num, train_num, best_loss):
    dataset = MyDataset(data_dir, data_cnt, None, None)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=10, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15000, verbose=True, factor=0.5)
    for _ in range(epoch_num):
        model_parallel.train()
        col_all_labels = []
        col_all_preds = []
        for data in dataloader:
            occ_grid, start, _, sample, col_label, _ = data

            # print(samples)
            start = start.to(device)
            occ_grid = occ_grid.to(device)
            sample = sample.to(device)
            col_label = col_label.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            # sel_scores = model_parallel(occ_grid_batch, start_batch, goal_batch, samples_batch)
            col_scores = model_parallel(occ_grid, start, sample, fixed_env=False)
            # sel_scores = sel_scores.view(occ_grid.shape[0], -1)

            # calculate loss
            col_loss = bce_loss(col_scores, col_label)
            loss = col_loss

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()
            scheduler.step(loss)

            # Print statistics
            if train_num % 100 == 0:
                print("Learning rate: {}".format(optimizer.param_groups[0]['lr']))
                print('Loss after epoch %d, mini-batch %5d, dataset_sizes %d: , col_loss: %.3f' % (iter_num, train_num, data_cnt, col_loss.item()))
                writer.add_scalar('col_loss/train', col_loss.item(), train_num)

            train_num += 1

            # if loss.item() < best_loss:
            #     best_loss = loss.item()
            #     torch.save(model.state_dict(), osp.join(CUR_DIR, "models/{}_best.pt".format(model_name)))

            col_label_list = col_label.detach().cpu().numpy().reshape(-1).tolist()
            col_all_labels += col_label_list
            pred_list = col_scores.detach().cpu().numpy().reshape(-1).tolist()
            col_all_preds += pred_list

        preds_binary = (np.array(col_all_preds) > COL_PRED_THRESHOLD).astype(int)
        accuracy = accuracy_score(np.array(col_all_labels), preds_binary)
        precision = precision_score(np.array(col_all_labels), preds_binary)
        recall = recall_score(np.array(col_all_labels), preds_binary)
        print("train_num: {}, dataset_col_accuracy : {}, dataset_col_precision : {}, dataset_col_recall: {} ".format(train_num, accuracy, precision, recall))
        writer.add_scalar('dataset_col_accuracy/iter', accuracy, train_num)
        writer.add_scalar('dataset_col_precision/iter', precision, train_num)
        writer.add_scalar('dataset_col_recall/iter', recall, train_num)

        num_pos = 0
        num_neg = 0
        for label in col_all_labels:
            if label == 1:
                num_pos += 1
            else:
                num_neg += 1
        writer.add_scalar('dataset_num_pos/iter', num_pos, train_num)
        writer.add_scalar('dataset_num_neg/iter', num_neg, train_num)

        torch.save(model.state_dict(), model_path)
        print("saved session to ", model_path)

        eval()

    return train_num, best_loss

def eval():
    global best_eval_loss
    print("eval")
    model_parallel.eval()

    all_labels = []
    all_preds = []
    total_loss = 0
    iter_num_tmp = 0
    for data in eval_dataloader:
        occ_grid, start, _, sample, col_label, _ = data
        v1_t = start.to(device)
        v2_t = sample.to(device)
        occ_grid_t = occ_grid.to(device)
        label_t = col_label.to(device)

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
    if total_loss < best_eval_loss:
        best_eval_loss = total_loss
        torch.save(model.state_dict(), osp.join(CUR_DIR, "models/{}_best.pt".format(model_name)))

    preds_binary = (np.array(all_preds) > COL_PRED_THRESHOLD).astype(int)
    accuracy = accuracy_score(np.array(all_labels), preds_binary)
    precision = precision_score(np.array(all_labels), preds_binary)
    recall = recall_score(np.array(all_labels), preds_binary)
    print("iter_num: {}, eval_col_accuracy : {}, eval_col_precision : {}, eval_col_recall: {} ".format(iter_num, accuracy, precision, recall))
    writer.add_scalar('eval_col_accuracy/iter', accuracy, iter_num)
    writer.add_scalar('eval_col_precision/iter', precision, iter_num)
    writer.add_scalar('eval_col_recall/iter', recall, iter_num)
    writer.add_scalar('eval_loss/iter', total_loss, iter_num)


class MyDataset(Dataset):
    def __init__(self, data_dir, dataset_size, transform=None, target_transform=None, device="cpu"):
        print("Instantiating dataset...")
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self.dataset_size = dataset_size
        self._data_dir = data_dir

        self.fk = utils.FkTorch(device)
        # self.dataset = self.load_dataset_from_file()

        # print("dataset size = {}".format(len(self.dataset)))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        file_path = osp.join(self._data_dir, "data_{}.pkl".format(idx))
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        occ_grid, start_pos, goal_pos, sample, col_label, sel_label = data

        start_pos = torch.Tensor(start_pos)
        goal_pos = torch.Tensor(goal_pos)
        sample = torch.Tensor(sample)
        occ_grid_t = torch.Tensor(occ_grid).view(1, occ_grid_dim, occ_grid_dim)
        col_label_t = torch.Tensor([col_label])
        sel_label_t = torch.Tensor([sel_label])

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
    parser.add_argument('--name', default='1')
    parser.add_argument('--checkpoint', default='')
    args = parser.parse_args()

    # constants
    robot_dim = 8
    linkpos_dim = 12
    occ_grid_dim = 40
    model_name = "model_col_dagger_small"
    model_path = osp.join(CUR_DIR, "models/{}.pt".format(model_name))
    data_dir = osp.join(CUR_DIR, "./dataset/{}".format(model_name))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # env dirs
    train_env_dirs = osp.join(CUR_DIR, "dataset/model")
    eval_data_dir = osp.join(CUR_DIR, "./dataset/model_eval")

    # hyperparameters
    bs = 128
    lr = 0.001
    num_epochs = 40
    max_iter = 100
    data_cnt = 0
    target_data_cnt = 100000
    max_data_cnt = 500000
    epoch_num_1 = 20
    epoch_num_2 = 20
    epoch_num_3 = 50
    first_time = True
    train_data_step = 50000
    eval_data_cnt = 66351

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    writer = SummaryWriter(comment = model_name)

    # define networks
    print("robot_dim = ", robot_dim, "linkpos_dim = ", linkpos_dim)
    model = ColCheckerSmall(robot_dim + linkpos_dim)
    if args.checkpoint != '':
        print("Loading checkpoint {}.pt".format(args.checkpoint))
        model.load_state_dict(torch.load(osp.join(CUR_DIR, 'models/{}.pt'.format(args.checkpoint))))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model_parallel = torch.nn.DataParallel(model)
    model.to(device)
    model_parallel.to(device)
    torch.save(model.state_dict(), model_path)

    eval_dataset = MyDataset(eval_data_dir, eval_data_cnt, None, None)
    eval_dataloader = DataLoader(eval_dataset, batch_size=bs, shuffle=True, drop_last=False, num_workers=10, pin_memory=True)

    train_num = 0
    iter_num = 0
    best_loss = float('inf')
    manager = mp.Manager()
    dataset_list = manager.list()
    while True:
        print("Executing {} iterations out of {}".format(iter_num, max_iter))
        if data_cnt >= max_data_cnt:
            break

        data_cnt = collect_data_from_train_env(train_env_dirs, data_cnt, iter_num, dataset_list)

        if data_cnt < target_data_cnt:
            iter_num += 1
            continue

        if first_time:
            print("Training for the first time")
            train_num, best_loss = train(data_cnt, epoch_num_1, iter_num, train_num, best_loss)
            first_time = False
        else:
            print("Training for the iteration")
            train_num, best_loss = train(data_cnt, epoch_num_2, iter_num, train_num, best_loss)

        target_data_cnt += train_data_step
        iter_num += 1

    # Collect enough amount of data, just train
    print("Collected enough amount of data. Just train")
    col_all_labels = []
    col_all_preds = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10000, verbose=True, factor=0.5)
    dataset = MyDataset(data_dir, data_cnt, None, None)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=10, pin_memory=True)
    for _ in range(epoch_num_3):
        model_parallel.train()
        for data in dataloader:
            occ_grid, start, goal, sample, col_label, sel_label = data

            # print(samples)
            start = start.to(device)
            occ_grid = occ_grid.to(device)
            sample = sample.to(device)
            col_label = col_label.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            # sel_scores = model_parallel(occ_grid_batch, start_batch, goal_batch, samples_batch)
            col_scores = model_parallel(occ_grid, start, sample, fixed_env=False)
            # sel_scores = sel_scores.view(occ_grid.shape[0], -1)

            # calculate loss
            col_loss = bce_loss(col_scores, col_label)
            loss = col_loss

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()
            scheduler.step(loss)

            # Print statistics
            if train_num % 100 == 0:
                print("Learning rate: {}".format(optimizer.param_groups[0]['lr']))
                print('Loss after epoch %d, mini-batch %5d, dataset_sizes %d: , col_loss: %.3f' % (iter_num, train_num, data_cnt, col_loss.item()))
                writer.add_scalar('col_loss/train', col_loss.item(), train_num)

            train_num += 1

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), osp.join(CUR_DIR, "models/{}_best.pt".format(model_name)))

            col_label_list = col_label.detach().cpu().numpy().reshape(-1).tolist()
            col_all_labels += col_label_list
            pred_list = col_scores.detach().cpu().numpy().reshape(-1).tolist()
            col_all_preds += pred_list

        preds_binary = (np.array(col_all_preds) > COL_PRED_THRESHOLD).astype(int)
        accuracy = accuracy_score(np.array(col_all_labels), preds_binary)
        precision = precision_score(np.array(col_all_labels), preds_binary)
        recall = recall_score(np.array(col_all_labels), preds_binary)
        print("train_num: {}, dataset_col_accuracy : {}, dataset_col_precision : {}, dataset_col_recall: {} ".format(train_num, accuracy, precision, recall))
        writer.add_scalar('dataset_col_accuracy/iter', accuracy, train_num)
        writer.add_scalar('dataset_col_precision/iter', precision, train_num)
        writer.add_scalar('dataset_col_recall/iter', recall, train_num)

        torch.save(model.state_dict(), model_path)
        print("saved session to ", model_path)

        eval()

        iter_num += 1

    writer.close()