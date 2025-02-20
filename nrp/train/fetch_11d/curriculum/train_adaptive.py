import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "../../"))

import os
import torch
import os.path as osp
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import pickle
import random
import json
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import shutil

from env.fetch_11d.fk.model import ProxyFkTorch
from planner.local_sampler_g.model_11d import VAE
from env.fetch_11d import utils
from env.fetch_11d.maze import Fetch11DEnv
from planner.neural_planner_g import NRP_g
from generate_path_dataset import collect_gt, convert_data
from train_local_sampler_g import vae_loss

CUR_DIR = osp.dirname(osp.abspath(__file__))

MAX_TIME = 5
LOCAL_ENV_SIZE = 2


class MyDataset(Dataset):
    def __init__(self, data_dir, dataset_size, transform=None, target_transform=None, device="cpu"):
        print("Instantiating dataset...")
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self.dataset_size = dataset_size

        fkmodel_path = osp.join(CUR_DIR, "../../env/fetch_11d/fk/models/model_fk.pt")
        self.fk = ProxyFkTorch(robot_dim, linkpos_dim, fkmodel_path, device)
        self._data_dir = data_dir

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        file_path = osp.join(self._data_dir, "data_{}.pkl".format(idx))
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        occ_grid, start_pos, goal_pos, expert_path = data
        expert_waypoint_pos = expert_path[1]

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


def collect_data(local_env_size, cur_local_dataset_size, num_local_data_to_collect, local_data_parent_dir):
    local_data_dir = osp.join(local_data_parent_dir, "{}".format(local_env_size))
    # if not os.path.exists(local_data_dir):
    #     os.makedirs(local_data_dir)

    lst = os.listdir(local_data_dir)  # your directory path
    existing_local_dataset_size = len(lst)

    if cur_local_dataset_size + num_local_data_to_collect < existing_local_dataset_size:
        copy_local_data_to_global_data(local_data_dir, global_data_dir, num_local_data_to_collect, cur_local_dataset_size)
        return num_local_data_to_collect

    # Actually collect
    train_env_dir = osp.join(CUR_DIR, f"../../env/fetch_11d/dataset/gibson/train")
    train_env_dirs = []
    for p in Path(train_env_dir).rglob("env.obj"):
        train_env_dirs.append(p.parent)
    collect_data_parent_dir = osp.join(CUR_DIR, f"dataset/train_tmp")
    collect_data_local_env_dir = osp.join(collect_data_parent_dir, f"{local_env_size}")
    if not os.path.exists(collect_data_local_env_dir):
        os.makedirs(collect_data_local_env_dir)

    print("Collecting gt")
    while existing_local_dataset_size < cur_local_dataset_size + num_local_data_to_collect:
        process_num = 10
        manager = mp.Manager()
        env_obj_dict = manager.dict()
        for env_idx in range(len(train_env_dirs)):
            env_obj_dict[env_idx] = 0

        j = 0
        while j < len(train_env_dirs):
            processes = []
            print("Running on env {} to {}".format(j, min(len(train_env_dirs), j + process_num)))
            for i in range(j, min(len(train_env_dirs), j + process_num)):
                p = mp.Process(
                    target=collect_gt,
                    args=(collect_data_local_env_dir, train_env_dirs, i, env_obj_dict, local_env_size, 10),
                    daemon=True,
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            j += process_num

        convert_data(train_env_dirs, collect_data_parent_dir, local_data_parent_dir, local_env_size)
        lst = os.listdir(local_data_dir)
        existing_local_dataset_size = len(lst)

    # Convert local data into global
    copy_local_data_to_global_data(local_data_dir, global_data_dir, num_local_data_to_collect, cur_local_dataset_size)

    return num_local_data_to_collect


def copy_local_data_to_global_data(local_data_dir, global_data_dir, num_data_to_copy, start_local_data_cnt=0):
    data_cnt = 0
    # dir_list = os.listdir(local_data_dir)
    # num_local_data = len(dir_list)
    num_global_data = len(os.listdir(global_data_dir))

    data_cnt = 0
    for data_idx in range(start_local_data_cnt, start_local_data_cnt + num_data_to_copy):
        file = os.path.join(local_data_dir, f"data_{data_idx}.pkl")
        shutil.copy(
            os.path.join(local_data_dir, file), os.path.join(global_data_dir, f"data_{num_global_data + data_cnt}.pkl")
        )
        data_cnt += 1

    return data_cnt


def train(dataset_size):
    # dataset and dataloader
    dataset = MyDataset(global_data_dir, dataset_size)
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
            recon_samples, means, log_var, z = model_parallel(expert_wp_t, occ_grid_t, context_t)

            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(recon_samples, expert_wp_t, means, log_var)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            scheduler.step(loss)

            # Print statistics
            if i % 100 == 0:
                print("Loss after mini-batch %5d, epoch %d : %.3f" % (i, epoch, loss.item()))
                print("recon_loss after mini-batch %5d, epoch %d: %.3f" % (i, epoch, recon_loss.item()))
                print("kl_loss after mini-batch %5d, epoch %d: %.3f" % (i, epoch, kl_loss.item()))
                writer.add_scalar("Loss/train", loss.item(), i)
                writer.add_scalar("KL_loss/train", kl_loss.item(), i)
                writer.add_scalar("Recon_loss/train", recon_loss.item(), i)

            if i % 500 == 0:
                torch.save(model.state_dict(), model_path)
                print("saved session to ", model_path)

            i += 1

        torch.save(model.state_dict(), model_path)
        print("saved session to ", model_path)

def eval():
    # construct env
    env = Fetch11DEnv(gui=False)
    dim = 11
    occ_grid_dim = [40, 40, 20]

    # construct planner
    neural_goal_bias = 0.2
    sl_bias = 0.05
    # model_path = osp.join(CUR_DIR, "../planner/local_sampler_g/models/{}/cvae_sel.pt".format(args.env))
    # model_path = osp.join(CUR_DIR, "/models/sampler_g.pt")
    neural_planner_g = NRP_g(
        env,
        model_path,
        optimal=True,
        dim=dim,
        occ_grid_dim=occ_grid_dim,
    )

    neural_planner_g.algo.goal_bias = neural_goal_bias
    neural_planner_g.algo.add_intermediate_state = False
    neural_planner_g.sl_bias = sl_bias
    # Get robot bounds
    low_bounds = env.robot.get_joint_lower_bounds()
    high_bounds = env.robot.get_joint_higher_bounds()
    low_bounds[0] = -2
    low_bounds[1] = -2
    high_bounds[0] = 2
    high_bounds[1] = 2
    # print(low_bounds, high_bounds)

    neural_planner_g.sampler.set_robot_bounds(low_bounds, high_bounds)

    env_num = 100
    testset = "test_env_hard"
    success = 0
    for i in range(env_num):
        env.clear_obstacles()
        maze_dir = osp.join(CUR_DIR, "../../env/fetch_11d/dataset/{}/{}".format(testset, i))
        occ_grid = env.utils.get_occ_grid(maze_dir)
        mesh_path = env.utils.get_mesh_path(maze_dir)

        env.load_mesh(mesh_path)
        env.load_occupancy_grid(occ_grid, add_enclosing=True)

        print("Loading env {} from {}".format(i, testset))
        env_dir = osp.join(CUR_DIR, "../../env/fetch_11d/dataset/{}/{}".format(testset, i))
        # env_dir = osp.join(CUR_DIR, "../eval_env/{}".format(7)) # `18`
        # with open(osp.join(env_dir, "obstacle_dict.json")) as f:
        #     obstacle_dict = json.load(f)

        occ_grid = env.get_occupancy_grid()

        with open(osp.join(env_dir, "start_goal.json")) as f:
            start_goal = json.load(f)
        # start_goal = json.load(osp.join(env_dir, "start_goal.json"))
        env.start = start_goal[0]
        env.goal = start_goal[1]
        env.robot.set_state(env.start)

        neural_planner_g.algo.return_on_path_find = False
        res = neural_planner_g.solve_step_time(env, env.start, env.goal, 10, 10)
        success_res = [tmp[0] for tmp in res]
        # path_list = [tmp[1] for tmp in res]
        # for idx, p in enumerate(path_list):
        #     # path = utils.interpolate(p)
        #     # utils.visualize_nodes_global(mesh_path, occ_grid, path, maze.start, maze.goal, show=False, save=True, file_name=osp.join(base_log_dir, "planned_path.png"))
        #     with open(osp.join(log_dir, "planned_path_{}_{}.json".format(repeat, idx)), "w") as f:
        #         json.dump(p, f)

        # for idx, res in enumerate(success_res):
        #     if res:
        #         success_list[idx] += 1
        success += success_res[-1]

        print(f"success {success}/{i+1}")

    return success


class SuccessPlataeuChecker:
    def __init__(self, patience=5, cooldown=2):
        self.num_bad_epochs = 0
        self.patience = patience
        self.cooldown = cooldown
        self.best = 0
        self.cooldown_counter = 0

    def plateaud(self, current):
        if current > self.best:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            return True

        return False

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0


def adaptive_train():
    local_env_size = 2
    local_dataset_size = 0
    total_dataset_size = 0
    checker = SuccessPlataeuChecker()
    for epoch_idx in range(1000):
        if local_env_size > 5:
            break

        # collect datas
        # the larger the local_env_size, the more optimal the collected data.
        num_new_data = collect_data(local_env_size, local_dataset_size, 10000, local_data_parent_dir)
        # dataset.dataset_size += num_new_data
        total_dataset_size += num_new_data
        local_dataset_size += num_new_data

        # train
        train(total_dataset_size)

        # success rate
        success_rate = eval()
        writer.add_scalar("success_rate", success_rate, epoch_idx)
        writer.add_scalar("local_env_size", local_env_size, epoch_idx)
        writer.add_scalar("local_dataset_size", local_dataset_size, epoch_idx)
        writer.add_scalar("total_dataset_size", total_dataset_size, epoch_idx)

        # collect more optimal samples if success rate plateaud.
        if checker.plateaud(success_rate):
            local_env_size += 1
            local_dataset_size = 0
            checker.best = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--name", default="3")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    # constants
    z_dim = 8
    robot_dim = 11
    linkpos_dim = 24
    state_dim = robot_dim + linkpos_dim
    goal_dim = state_dim + 1
    occ_grid_dim = 40
    occ_grid_dim_z = 20
    model_name = "sampler_g"
    model_path = osp.join(CUR_DIR, "models/{}.pt".format(model_name))
    global_data_dir = osp.join(CUR_DIR, "dataset/fetch_11d_adaptive")
    if not os.path.exists(global_data_dir):
        os.makedirs(global_data_dir)

    local_data_parent_dir = osp.join(CUR_DIR, "dataset/train2")
    # eval_data_dir = osp.join(CUR_DIR, "dataset/model_eval")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(comment="_{}".format(model_name))

    # hyperparameters
    bs = 128
    lr = 1e-3
    num_steps = 10000
    num_epochs = 10
    alpha = 0.01

    # define networks
    print("dim = ", robot_dim)
    print("z_dim = ", z_dim)
    model = VAE(z_dim, state_dim + goal_dim, state_dim)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    exit()

    if args.checkpoint != "":
        print("Loading checkpoint {}.pt".format(args.checkpoint))
        model.load_state_dict(torch.load(osp.join(CUR_DIR, "models/{}.pt".format(args.checkpoint))))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_parallel = torch.nn.DataParallel(model)
    else:
        model_parallel = model
    model_parallel.to(device)

    # Define the loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10000, verbose=True, factor=0.5)

    adaptive_train()
