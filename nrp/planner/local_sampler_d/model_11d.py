import torch
import torch.nn as nn
import os.path as osp

from nrp.planner.voxnet.vox_net import VoxNetEncoderMagic, VoxNetEncoderGlobal

CUR_DIR = osp.dirname(osp.abspath(__file__))


def init_weight(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.0)

class Selector(nn.Module):
    def __init__(self, state_dim, goal_dim, h_dim=1024, linear_depth=2):
        super().__init__()
        self.state_dim = state_dim
        self.h_dim = h_dim
        self.goal_dim = goal_dim

        self.feature_extractor = VoxNetEncoderMagic(4)
        occ_feat_size = self.feature_extractor.output_size
        print("occ feat size = {}".format(occ_feat_size))

        # self.feature_extractor = UNet(1, state_dim)
        # feat_size = 100 * state_dim

        self.layers = nn.Sequential(
            nn.Linear(state_dim * 2 + self.goal_dim + occ_feat_size, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.LeakyReLU(),
        )

        for i in range(linear_depth):
            self.layers.add_module("linear_{}".format(i), nn.Linear(self.h_dim, self.h_dim))
            self.layers.add_module("batch_norm_{}".format(i), nn.BatchNorm1d(self.h_dim))
            self.layers.add_module("activation_{}".format(i), nn.LeakyReLU())

        self.head = nn.Sequential(nn.Linear(self.h_dim, 256), nn.LeakyReLU(), nn.Linear(256, 1), nn.Sigmoid())

        # self.apply(init_weight)

    def forward(self, occ_grid, start, goal, samples, fixed_env: bool = False):
        """
        occ_grid: bs x 4 x occ_dims
        start : bs x dim
        goal: bs x goal_dim
        samples: bs x dim
        """
        # print(occ_grid.shape, start.shape, goal.shape, samples.shape)

        bs = occ_grid.shape[0]
        f = self.feature_extractor(occ_grid)

        num_of_sample = 1
        if fixed_env:
            num_of_sample = samples.shape[1]
            f = f.unsqueeze(1)
            f = f.repeat(1, num_of_sample, 1).view(bs * num_of_sample, -1)  # (bs * N) * f_dim
            start = start.unsqueeze(1).repeat(1, num_of_sample, 1).view(bs * num_of_sample, -1)  # (bs * N) * dim
            goal = goal.unsqueeze(1).repeat(1, num_of_sample, 1).view(bs * num_of_sample, -1)  # (bs * N) * dim
            samples = samples.view(bs * num_of_sample, -1)  # (bs * N) * dim

        x = torch.cat((f, start, goal, samples), dim=-1)  # (bs * N) * (|f| + dim * 2 + goal_dim)
        x = self.layers(x)
        y = self.head(x)

        # forward pass
        y = y.view(bs, num_of_sample)

        return y


class DiscriminativeSampler(nn.Module):
    def __init__(self, state_dim, occ_grid_dim, model_path, linkpos_dim=24, global_mode=False):
        super().__init__()
        # self.occ_grid_size = occ_grid_dim

        if not global_mode:
            self.sel_model = Selector(state_dim + linkpos_dim, state_dim + linkpos_dim + 1)  # 11 + 24 + 1
        else:
            self.sel_model = SelectorGlobal(state_dim + linkpos_dim, state_dim + linkpos_dim + 1)  # 11 + 24 + 1

        if model_path is not None:
            self.sel_model.load_state_dict(torch.load(model_path))

        self.sel_model.eval()
        self.sel_model = torch.jit.script(self.sel_model)
        # self.strategy = "multiplication"

    def forward(self, occ_grid, start, goal, sample):
        sel_score = self.sel_model(occ_grid, start, goal, sample)
        return sel_score

    def get_sel_score(self, occ_grid, start, goal, samples, fixed_env=True):
        sel_score = self.sel_model(occ_grid, start, goal, samples, fixed_env=fixed_env)
        return sel_score

    # def select_from_samples(self, occ_grid, start, goal, samples, fixed_env=False):
    #     # select
    #     sel_score = self.sel_model(occ_grid, start, goal, samples, fixed_env=fixed_env)
    #     best_indice = torch.argmax(sel_score)

    #     # good_indices = torch.nonzero(sel_score.view(-1) >= 0.5).view(-1)
    #     # if len(good_indices) <= 0:
    #     #     good_indices = torch.topk(sel_score, min(10, samples.shape[1]))[1].view(-1)
    #     # best_indice = good_indices[random.randint(0, len(good_indices) - 1)]

    #     # good_indices = torch.topk(sel_score, min(10, samples.shape[1]))[1].view(-1)
    #     # best_indice = good_indices[random.randint(0, len(good_indices) - 1)]

    #     return best_indice

    # def select_from_samples_heuristic(self, occ_grid, start, goal, samples):
    #     sel_scores = self.sel_model(occ_grid, start, goal, samples, fixed_env=True)
    #     sel_good_indices = torch.nonzero(sel_scores >= self.sel_pred_threshold).view(-1)

    #     sel_filtered_samples_t = samples[sel_good_indices]
    #     # print(filtered_samples_t.shape, sel_filtered_samples_t.shape)
    #     # sel_filtered_samples_list = sel_filtered_samples_t.cpu().numpy()[:, :self.dim].tolist()

    #     # if self.visualize:
    #     #     utils.visualize_nodes_local(occ_grid_np, sel_filtered_samples_list, v, g,
    #     #         show=False, save=True, file_name = osp.join(self.log_dir, "sel_valid_samples_viz_{}.png".format(self.i)))

    #     sel_num_good_samples = sel_filtered_samples_t.shape[0]

    #     if sel_num_good_samples > 0:
    #         goal_t = goal.unsqueeze(0).repeat(sel_num_good_samples, 1)
    #         # print(sel_filtered_samples_t.shape, goal_t.shape)

    #         # We should use rs path length. But it might take too much time
    #         diff = sel_filtered_samples_t[:, : self.dim] - goal_t
    #         diff[:, 3:] *= 0.125
    #         dist = torch.linalg.norm(diff, dim=-1).view(-1)

    #         sel_best_filtered_indice = torch.argmin(dist)
    #         sel_best_indice = sel_good_indices[sel_best_filtered_indice]
    #     else:
    #         sel_best_indice = torch.argmax(sel_scores)

    #     return sel_best_indice

    # def get_final_sel_scores(self, occ_grid, start, goal, samples, fixed_env=False):
    #     if self.strategy == "filter":
    #         bad_indices = torch.nonzero(col_scores < 0.5)
    #         sel_scores = self.sel_model(occ_grid, start, goal, samples, fixed_env)
    #         sel_scores[bad_indices] = 0.0
    #         # filtered_samples = samples[good_indices]
    #         # if not fixed_env:
    #         #     occ_grid = occ_grid[good_indices]
    #         #     start = start[good_indices]
    #         #     goal = goal[good_indices]

    #     elif self.strategy == "multiplication":
    #         sel_scores = self.sel_model(occ_grid, start, goal, samples, fixed_env)

    #     return sel_scores

class SelectorGlobal(nn.Module):
    def __init__(self, state_dim, goal_dim, h_dim=1024, linear_depth=2):
        super().__init__()
        self.state_dim = state_dim
        self.h_dim = h_dim
        self.goal_dim = goal_dim

        self.feature_extractor = VoxNetEncoderGlobal(4)
        occ_feat_size = self.feature_extractor.output_size
        print("occ feat size = {}".format(occ_feat_size))

        # self.feature_extractor = UNet(1, state_dim)
        # feat_size = 100 * state_dim

        self.layers = nn.Sequential(
            nn.Linear(state_dim * 2 + self.goal_dim + occ_feat_size, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.LeakyReLU(),
        )

        for i in range(linear_depth):
            self.layers.add_module("linear_{}".format(i), nn.Linear(self.h_dim, self.h_dim))
            self.layers.add_module("batch_norm_{}".format(i), nn.BatchNorm1d(self.h_dim))
            self.layers.add_module("activation_{}".format(i), nn.LeakyReLU())

        self.head = nn.Sequential(nn.Linear(self.h_dim, 256), nn.LeakyReLU(), nn.Linear(256, 1), nn.Sigmoid())

        # self.apply(init_weight)

    def forward(self, occ_grid, start, goal, samples, fixed_env: bool = False):
        """
        occ_grid: bs x 4 x occ_dims
        start : bs x dim
        goal: bs x goal_dim
        samples: bs x dim
        """
        # print(occ_grid.shape, start.shape, goal.shape, samples.shape)

        bs = occ_grid.shape[0]
        f = self.feature_extractor(occ_grid)

        num_of_sample = 1
        if fixed_env:
            num_of_sample = samples.shape[1]
            f = f.unsqueeze(1)
            f = f.repeat(1, num_of_sample, 1).view(bs * num_of_sample, -1)  # (bs * N) * f_dim
            start = start.unsqueeze(1).repeat(1, num_of_sample, 1).view(bs * num_of_sample, -1)  # (bs * N) * dim
            goal = goal.unsqueeze(1).repeat(1, num_of_sample, 1).view(bs * num_of_sample, -1)  # (bs * N) * dim
            samples = samples.view(bs * num_of_sample, -1)  # (bs * N) * dim

        x = torch.cat((f, start, goal, samples), dim=-1)  # (bs * N) * (|f| + dim * 2 + goal_dim)
        x = self.layers(x)
        y = self.head(x)

        # forward pass
        y = y.view(bs, num_of_sample)

        return y
