import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "../"))

import torch
import math
import os.path as osp
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import torch.multiprocessing as mp

# from env.maze_2d import Maze2D
import utils
from NEXT.model import Model
from NEXT.algorithm import NEXT_plan, RRTS_plan
from NEXT.environment.maze_env import MyMazeEnv

CUR_DIR = osp.dirname(osp.abspath(__file__))
ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)

# def gaussian_probability(sigma, mu, target):
#     """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
#     Arguments:
#         sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
#             size, G is the number of Gaussians, and O is the number of
#             dimensions per Gaussian.
#         mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
#             number of Gaussians, and O is the number of dimensions per Gaussian.
#         target (BxI): A batch of target. B is the batch size and I is the number of
#             input dimensions.
#     Returns:
#         probabilities (BxG): The probability of each point in the probability
#             of the distribution in the corresponding sigma/mu index.
#     """
#     target = target.unsqueeze(1).expand_as(sigma)
#     ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
#     return torch.prod(ret, 2)


def policy_loss(sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """

    # m = MultivariateNormal(mu, torch.diag(sigma))
    # prob = m.log_prob(target)

    # # prob = gaussian_probability(sigma, mu, target)
    # # nll = -torch.log(torch.sum(prob, dim=1))
    # return -torch.mean(prob)
    loss = mse_loss(mu, target)
    return loss


def value_loss(pred, target):
    loss = mse_loss(pred, target)
    return loss


def extract_path(search_tree):
    leaf_id = search_tree.states.shape[0] - 1

    path = [search_tree.states[leaf_id]]
    id = leaf_id
    while id:
        parent_id = search_tree.rewired_parents[id]
        if parent_id:
            path.append(search_tree.states[parent_id])

        id = parent_id

    path.append(search_tree.non_terminal_states[0])  # append the init state
    path.reverse()

    return path


class MyDataset(Dataset):
    def __init__(
        self, env, dataset_size, transform=None, target_transform=None, device="cpu"
    ):
        print("Instantiating dataset...")
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self.dataset_size = dataset_size
        # self.dataset = self.load_dataset_from_file()

        # print("dataset size = {}".format(len(self.dataset)))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        file_path = osp.join(data_dir, "data_{}.pkl".format(idx))
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # low = torch.Tensor([-2, -2, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi]).view(1, -1)
        # high = torch.Tensor([2, 2, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi]).view(1, -1)

        occ_grid, start, goal, pos, next_pos, dist_to_g = data

        # pos[0] -= 5
        # pos[1] -= 5
        # next_pos[0] -= 5
        # next_pos[1] -= 5
        # start[0] -= 5
        # start[1] -= 5
        # goal[0] -= 5
        # goal[1] -= 5

        start_t = torch.Tensor(start)
        goal_t = torch.Tensor(goal)
        occ_grid_t = torch.Tensor(occ_grid).view(1, occ_grid_dim, occ_grid_dim)
        pos_t = torch.Tensor(pos)
        next_pos_t = torch.Tensor(next_pos)
        dist_to_g_t = torch.Tensor([-dist_to_g]) # make it negative as the algorithm takes argmax

        action_t = next_pos_t - pos_t
        edge_cost = torch.linalg.norm(action_t[:2]).item()
        if edge_cost > env.LOCAL_ENV_SIZE:
            action = env.interpolate(
                np.array(pos), np.array(next_pos), env.LOCAL_ENV_SIZE / edge_cost
            ) - np.array(pos)
            action_t = torch.Tensor(action)

        return occ_grid_t, start_t, goal_t, pos_t, action_t, dist_to_g_t


def train_init(env, epoch=50):
    print("Training init on offline collected path dataset")
    global batch_num
    model.net.train()
    # Define the loss function and optimizer
    optimizer = torch.optim.Adam(model.net.parameters(), lr=lr, weight_decay=0.00005)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10000, verbose=True, factor=0.5)

    best_loss = float("inf")
    dataset = MyDataset(env, data_cnt, None, None)
    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        num_workers=10,
        pin_memory=True,
    )
    for epoch_n in range(epoch):
        for data in dataloader:
            occ_grid, start, goal, pos, action, dist_to_g = data

            start = start.to(device)
            goal = goal.to(device)
            occ_grid = occ_grid.to(device)
            pos = pos.to(device)
            action = action.to(device)
            dist_to_g = dist_to_g.to(device)

            # problem = {
            #     "map": occ_grid,
            #     "init_state": start,
            #     "goal_state": goal
            # }

            model.pb_forward(goal, occ_grid)
            # y = model.net.state_forward(pos, model.pb_rep)
            # action_pred = y[:, :robot_dim]
            # value_pred = y[:, -1].view(bs, 1)
            action_pred, value_pred = model.net_forward(pos, use_np=False)
            # print(action_pred.shape, value_pred.shape)

            p_loss = policy_loss(sigma, action_pred, action)
            v_loss = value_loss(value_pred, dist_to_g)

            loss = alpha_p * p_loss + alpha_v * v_loss

            # Zero the gradients
            optimizer.zero_grad()

            loss.backward()

            # Perform optimization
            optimizer.step()

            # scheduler.step(loss)

            if batch_num % 100 == 0:
                print(
                    "Loss after epoch %d, train_iter %d, dataset_sizes %d: , p_loss: %.3f, v_loss: %.3f"
                    % (
                        epoch_n,
                        batch_num,
                        data_cnt,
                        alpha_p * p_loss.item(),
                        alpha_v * v_loss.item(),
                    )
                )
                writer.add_scalar("p_loss/train", alpha_p * p_loss.item(), batch_num)
                writer.add_scalar("v_loss/train", alpha_v * v_loss.item(), batch_num)

            if loss.item() < best_loss:
                torch.save(model.net.state_dict(), best_model_path)
                best_loss = loss.item()

            batch_num += 1

        torch.save(model.net.state_dict(), model_path)
        print("saved session to ", model_path)


parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--name", default="1")
parser.add_argument("--checkpoint", default="")
args = parser.parse_args()

writer = SummaryWriter(comment="_next")

# Constatns
data_dir = osp.join(CUR_DIR, "dataset/train")
maze_dir = osp.join(CUR_DIR, "../dataset/gibson/train")

model_path = osp.join(CUR_DIR, "models/next_v3.pt")
best_model_path = osp.join(CUR_DIR, "models/next_v3_best.pt")

# Hyperparameters:
visualize = False
cuda = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch_num = 9000
train_num = 10
UCB_type = "kde"
robot_dim = 8
bs = 256
occ_grid_dim = 100
train_step_cnt = 2000
lr = 0.001
alpha_p = 1
alpha_v = 1 / 10
sigma = torch.tensor([0.25, 0.25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).to(device)

env = MyMazeEnv(robot_dim, maze_dir)
model = Model(env, cuda=cuda, dim=robot_dim, env_width=occ_grid_dim)
mse_loss = torch.nn.MSELoss()

if args.checkpoint != "":
    print("Loading checkpoint {}.pt".format(args.checkpoint))
    model.net.load_state_dict(
        torch.load(osp.join(CUR_DIR, "models/{}.pt".format(args.checkpoint)))
    )

start_epoch = 0
data_cnt = 57010
train_data_cnt = data_cnt + train_step_cnt
batch_num = 0
best_loss = float("inf")
success_rate = []

train_init(env)

for epoch in range(start_epoch, epoch_num):
    model.net.eval()
    problem = env.init_new_problem()
    model.set_problem(problem)

    g_explore_eps = 1 - ((epoch // 1000) + 1) / 10.0

    # if epoch < 2000:
    #     g_explore_eps = 1.0
    # elif epoch < 4000:
    #     g_explore_eps = 0.5 - 0.4 * (epoch - 2000) / 2000
    #     # g_explore_eps *= 0.7
    # else:
    #     g_explore_eps = 0.1

    # Get path
    print("Planning... with explore_eps: {}".format(g_explore_eps))
    path = None

    search_tree, done = NEXT_plan(
        env = env,
        model = model,
        T = 500,
        g_explore_eps = g_explore_eps,
        stop_when_success = True,
        UCB_type = UCB_type
    )
    if done:
        success_rate.append(1)
        path = extract_path(search_tree)
    else:
        success_rate.append(0)
        path = env.expert_path

    if path is not None:
        print("Get path, saving to data")

        if visualize:
            print(path[0], env.init_state, path[-1], env.goal_state)
            assert np.allclose(np.array(path[0]), np.array(env.init_state))
            assert np.allclose(np.array(path[-1]), np.array(env.goal_state))
            path_tmp = utils.interpolate(path)
            utils.visualize_nodes_global(env.map_orig, path_tmp, env.init_state, env.goal_state, show=False, save=True, file_name=osp.join(CUR_DIR, "tmp.png"))

        tmp_dataset = []
        for idx in range(1, len(path)):
            pos = path[idx - 1]
            next_pos = path[idx]
            dist_to_g = utils.cal_path_len(path[idx-1:])
            tmp_dataset.append([env.map, env.init_state, env.goal_state, pos, next_pos, dist_to_g])

        for idx, data in enumerate(tmp_dataset):
            file_path = osp.join(data_dir, "data_{}.pkl".format(data_cnt + idx))
            with open(file_path, 'wb') as f:
                # print("Dumping to {}".format(file_path))
                pickle.dump(tmp_dataset[idx], f)
        data_cnt += len(tmp_dataset)

    print("data_cnt: {}".format(data_cnt))
    writer.add_scalar('dataset_size', data_cnt, epoch)

    if len(success_rate) > 100:
        success_rate.pop(0)
        avg_success_rate = np.sum(np.array(success_rate)) / 100
        print("Average success rate: ", avg_success_rate)
        writer.add_scalar('avg_success_rate', avg_success_rate, epoch)

    if data_cnt > train_data_cnt:
        model.net.train()
        # Define the loss function and optimizer
        optimizer = torch.optim.Adam(model.net.parameters(), lr=lr, weight_decay=0.00005)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10000, verbose=True, factor=0.5)

        dataset = MyDataset(env, data_cnt, None, None)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=10, pin_memory=True)
        for j in range(train_num):
            for data in dataloader:
                occ_grid, start, goal, pos, action, dist_to_g = data

                start = start.to(device)
                goal = goal.to(device)
                occ_grid = occ_grid.to(device)
                pos = pos.to(device)
                action = action.to(device)
                dist_to_g = dist_to_g.to(device)

                # problem = {
                #     "map": occ_grid,
                #     "init_state": start,
                #     "goal_state": goal
                # }

                model.pb_forward(goal, occ_grid)
                # y = model.net.state_forward(pos, model.pb_rep)
                # action_pred = y[:, :robot_dim]
                # value_pred = y[:, -1].view(bs, 1)
                action_pred, value_pred = model.net_forward(pos, use_np=False)

                p_loss = policy_loss(sigma, action_pred, action)
                v_loss = value_loss(value_pred, dist_to_g)

                loss = alpha_p * p_loss + alpha_v * v_loss

                # Zero the gradients
                optimizer.zero_grad()

                # Backward
                loss.backward()

                # Perform optimization
                optimizer.step()

                # scheduler.step(loss)

                print('Loss after epoch %d, batch_num %d, dataset_sizes %d:, p_loss: %.3f, v_loss: %.3f' % (epoch, batch_num, data_cnt, alpha_p * p_loss.item(), alpha_v * v_loss.item()))
                writer.add_scalar('p_loss/train', alpha_p * p_loss.item(), batch_num)
                writer.add_scalar('v_loss/train', alpha_v * v_loss.item(), batch_num)

                batch_num += 1

                if loss.item() < best_loss:
                    torch.save(model.net.state_dict(), best_model_path)
                    print("saved session to ", best_model_path)
                    best_loss = loss.item()

            torch.save(model.net.state_dict(), model_path)
            print("saved session to ", model_path)

        train_data_cnt += train_step_cnt
