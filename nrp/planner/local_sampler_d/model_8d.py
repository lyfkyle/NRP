import os.path as osp
import torch
import torch.nn as nn
import time


CUR_DIR = osp.dirname(osp.abspath(__file__))


def init_weight(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.0)


class CNNEncoder(nn.Module):
    def __init__(self, output_size, input_size=(1, 10, 10)):
        super(CNNEncoder, self).__init__()

        self.input_size = input_size
        self.channel_mult = 16

        # convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.channel_mult*1, kernel_size=4, stride=2, padding=1),  # out = (16, 5, 5)
            nn.LeakyReLU(),
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 3, 2, 1),  # out = (32, 3, 3)
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 3, 2, 1),  # out = (64, 2, 2)
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(),
            nn.Conv2d(self.channel_mult*4, self.channel_mult*8, 4, 2, 1),  # out = (128, 1, 1)
            nn.BatchNorm2d(self.channel_mult*8),
            nn.LeakyReLU(),
            nn.Flatten()
            # nn.Conv2d(self.channel_mult*8, self.channel_mult*16, 3, 2, 1),
            # nn.BatchNorm2d(self.channel_mult*16),
            # nn.LeakyReLU(0.2, inplace=True)
        )

        # print("here")
        self.flat_fts = self.get_flat_fts(self.conv)
        print(self.flat_fts)
        # self.flat_fts = 512

        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(),
        )

    def get_flat_fts(self, fts):
        tmp = torch.ones(1, *self.input_size)
        f = fts(tmp).view(1, -1)
        return f.shape[1]

    def extract_feat(self, x):
        x = self.conv(x)
        x = x.view(-1, self.flat_fts)
        return x

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.flat_fts)
        return x


class ColChecker(nn.Module):
    def __init__(self, state_dim, occ_grid_size=40, h_dim=1024, linear_depth=2):
        super().__init__()
        self.state_dim = state_dim
        self.h_dim = h_dim

        self.feature_extractor = CNNEncoder(1, (1, occ_grid_size, occ_grid_size))
        occ_feat_size = self.feature_extractor.flat_fts
        print("occ feat size = {}".format(occ_feat_size))

        # self.feature_extractor = UNet(1, state_dim)
        # feat_size = 100 * state_dim

        self.layers = nn.Sequential(
            nn.Linear(state_dim * 2 + occ_feat_size, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.LeakyReLU(),
        )

        for i in range(linear_depth):
            self.layers.add_module("linear_{}".format(i), nn.Linear(self.h_dim, self.h_dim))
            self.layers.add_module("batch_norm_{}".format(i), nn.BatchNorm1d(self.h_dim))
            self.layers.add_module("activation_{}".format(i), nn.LeakyReLU())

        self.head = nn.Sequential(
            nn.Linear(self.h_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # self.apply(init_weight)

    def forward(self, occ_grid, start, samples, fixed_env=True):
        """
        occ_grid: bs x 1 x occ_dims x occ_dims
        start : bs x dim
        samples: bs x dim
        """
        # print(occ_grid.shape, start.shape, samples.shape)

        bs = occ_grid.shape[0]
        f = self.feature_extractor(occ_grid)

        num_of_sample = 1
        if fixed_env:
            num_of_sample = samples.shape[1]
            f = f.unsqueeze(1)
            f = f.repeat(1, num_of_sample, 1).view(bs * num_of_sample, -1)  # (bs * N) * f_dim
            start = start.unsqueeze(1).repeat(1, num_of_sample, 1).view(bs * num_of_sample, -1)  # (bs * N) * dim
            samples = samples.view(bs * num_of_sample, -1)  # (bs * N) * dim

        x = torch.cat((f, start, samples), dim=-1)  # (bs * N) * (|f| + dim * 2 + goal_dim)
        x = self.layers(x)
        y = self.head(x)

        # forward pass
        y = y.view(bs, num_of_sample)

        return y


class Selector(nn.Module):
    def __init__(self, state_dim, goal_dim, occ_grid_size=40, h_dim=1024, linear_depth=2):
        super().__init__()
        self.state_dim = state_dim
        self.h_dim = h_dim
        self.goal_dim = goal_dim

        self.feature_extractor = CNNEncoder(1, (1, occ_grid_size, occ_grid_size))
        occ_feat_size = self.feature_extractor.flat_fts
        print("occ feat size = {}".format(occ_feat_size))

        # self.feature_extractor = UNet(1, state_dim)
        # feat_size = 100 * state_dim

        self.layers = nn.Sequential(
            nn.Linear(state_dim*2 + self.goal_dim + occ_feat_size, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.LeakyReLU(),
        )

        for i in range(linear_depth):
            self.layers.add_module("linear_{}".format(i), nn.Linear(self.h_dim, self.h_dim))
            self.layers.add_module("batch_norm_{}".format(i), nn.BatchNorm1d(self.h_dim))
            self.layers.add_module("activation_{}".format(i), nn.LeakyReLU())

        self.head = nn.Sequential(
            nn.Linear(self.h_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # self.apply(init_weight)

    def forward(self, occ_grid, start, goal, samples, fixed_env=True):
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


class SelectModel(nn.Module):
    def __init__(self, state_dim, occ_grid_size, col_checker_path=None, selector_path=None, linkpos_dim=12, device="cuda"):
        super().__init__()
        self.occ_grid_size = occ_grid_size

        self.col_model = ColChecker(state_dim + linkpos_dim, occ_grid_size)
        self.sel_model = Selector(state_dim + linkpos_dim, state_dim + linkpos_dim + 1, occ_grid_size)

        if col_checker_path is not None:
            self.col_model.load_state_dict(torch.load(col_checker_path))

        if selector_path is not None:
            self.sel_model.load_state_dict(torch.load(selector_path))

        self.col_model.to(device)
        self.sel_model.to(device)
        self.col_model.eval()
        self.sel_model.eval()
        # self.col_model = torch.jit.trace(self.col_model)
        # self.sel_model = torch.jit.trace(self.sel_model)

        self.col_model = torch.jit.optimize_for_inference(torch.jit.script(self.col_model))
        self.sel_model = torch.jit.optimize_for_inference(torch.jit.script(self.sel_model))

        self._s1 = torch.cuda.Stream()
        self._s2 = torch.cuda.Stream()

    def forward(self, occ_grid, start, goal, sample, fixed_env=True):
        col_score = self.col_model(occ_grid, start, sample, fixed_env)
        sel_score = self.sel_model(occ_grid, start, goal, sample, fixed_env)
        sel_score = col_score * sel_score
        return col_score, sel_score

    def get_col_score(self, occ_grid, start, sample):
        col_score = self.col_model(occ_grid, start, sample, fixed_env=True)
        return col_score

    def get_sel_score(self, occ_grid, start, goal, samples):
        sel_score = self.sel_model(occ_grid, start, goal, samples, fixed_env=True)
        return sel_score

    def get_final_sel_scores(self, occ_grid, start, goal, samples, fixed_env=False):
        col_scores = self.col_model(occ_grid, start, samples, fixed_env)
        sel_scores = self.sel_model(occ_grid, start, goal, samples, fixed_env)
        sel_scores = col_scores * sel_scores

        return sel_scores

    def select_from_samples(self, occ_grid, start, goal, samples, fixed_env=False):
        final_score = self.get_final_sel_scores(occ_grid, start, goal, samples, fixed_env)
        best_indice = torch.argmax(final_score)
        return best_indice

    def select_from_samples_heuristic(self, occ_grid, start, goal, samples, fixed_env=False):
        final_score = self.get_final_sel_scores(occ_grid, start, goal, samples, fixed_env)
        sel_good_indices = torch.nonzero(final_score >= 0.5).view(-1)

        sel_filtered_samples_t = samples[sel_good_indices]
        # print(filtered_samples_t.shape, sel_filtered_samples_t.shape)
        # sel_filtered_samples_list = sel_filtered_samples_t.cpu().numpy()[:, :self.dim].tolist()

        # if self.visualize:
        #     utils.visualize_nodes_local(occ_grid_np, sel_filtered_samples_list, v, g,
        #         show=False, save=True, file_name = osp.join(self.log_dir, "sel_valid_samples_viz_{}.png".format(self.i)))

        sel_num_good_samples = sel_filtered_samples_t.shape[0]

        if sel_num_good_samples > 0:
            goal_t = goal.unsqueeze(0).repeat(sel_num_good_samples, 1)
            # print(sel_filtered_samples_t.shape, goal_t.shape)

            # We should use rs path length. But it might take too much time
            diff = sel_filtered_samples_t[:, :self.dim] - goal_t
            diff[:, 3:] *= 0.125
            dist = torch.linalg.norm(diff, dim=-1).view(-1)

            sel_best_filtered_indice = torch.argmin(dist)
            sel_best_indice = sel_good_indices[sel_best_filtered_indice]
        else:
            sel_best_indice = torch.argmax(final_score)

        return sel_best_indice


class CNNEncoderSmall(nn.Module):
    def __init__(self, output_size, input_size=(1, 10, 10)):
        super(CNNEncoderSmall, self).__init__()

        self.input_size = input_size
        self.channel_mult = 16

        # convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.channel_mult*1, kernel_size=4, stride=2, padding=1),  # out = (16, 5, 5)
            nn.LeakyReLU(),
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 3, 2, 1),  # out = (32, 3, 3)
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 3, 2, 1),  # out = (64, 2, 2)
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        # print("here")
        self.flat_fts = self.get_flat_fts(self.conv)
        print(self.flat_fts)
        # self.flat_fts = 512

        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(),
        )

    def get_flat_fts(self, fts):
        tmp = torch.ones(1, *self.input_size)
        f = fts(tmp).view(1, -1)
        return f.shape[1]

    def extract_feat(self, x):
        x = self.conv(x)
        x = x.view(-1, self.flat_fts)
        return x

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.flat_fts)
        return x


class ColCheckerSmall(nn.Module):
    def __init__(self, state_dim, occ_grid_dim=[1, 40, 40], h_dim=512):
        super().__init__()
        self.state_dim = state_dim
        self.h_dim = h_dim

        self.feature_extractor = CNNEncoderSmall(1, (occ_grid_dim[0], occ_grid_dim[1], occ_grid_dim[2]))
        occ_feat_size = self.feature_extractor.flat_fts
        print("occ feat size = {}".format(occ_feat_size))

        self.layers = nn.Sequential(
            nn.Linear(state_dim * 2 + occ_feat_size, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.LeakyReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(self.h_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # self.apply(init_weight)

    def forward(self, occ_grid, start, samples, fixed_env: bool = True):
        """
        occ_grid: bs x 1 x occ_dims x occ_dims
        start : bs x dim
        samples: bs x dim
        """
        # print(occ_grid.shape, start.shape, samples.shape)

        bs = occ_grid.shape[0]
        f = self.feature_extractor(occ_grid)

        num_of_sample = 1
        if fixed_env:
            num_of_sample = samples.shape[1]
            f = f.unsqueeze(1)
            f = f.repeat(1, num_of_sample, 1).view(bs * num_of_sample, -1)  # (bs * N) * f_dim
            start = start.unsqueeze(1).repeat(1, num_of_sample, 1).view(bs * num_of_sample, -1)  # (bs * N) * dim
            samples = samples.view(bs * num_of_sample, -1)  # (bs * N) * dim

        x = torch.cat((f, start, samples), dim=-1)  # (bs * N) * (|f| + dim * 2 + goal_dim)
        x = self.layers(x)
        y = self.head(x)

        # forward pass
        y = y.view(bs, num_of_sample)

        return y


class SelectorSmall(nn.Module):
    def __init__(self, state_dim, goal_dim, occ_grid_dim=[1, 40, 40], h_dim=512):
        super().__init__()
        self.state_dim = state_dim
        self.h_dim = h_dim
        self.goal_dim = goal_dim

        self.feature_extractor = CNNEncoderSmall(1, (occ_grid_dim[0], occ_grid_dim[1], occ_grid_dim[2]))
        occ_feat_size = self.feature_extractor.flat_fts
        print("occ feat size = {}".format(occ_feat_size))

        # self.feature_extractor = UNet(1, state_dim)
        # feat_size = 100 * state_dim

        self.layers = nn.Sequential(
            nn.Linear(state_dim*2 + self.goal_dim + occ_feat_size, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.LeakyReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(self.h_dim, 1),
            # nn.LeakyReLU(),
            # nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # self.apply(init_weight)

    def forward(self, occ_grid, start, goal, samples, fixed_env: bool = True):
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


class SelectModelSmall(nn.Module):
    def __init__(self, state_dim, occ_grid_size, col_checker_path=None, selector_path=None, linkpos_dim=12, device="cuda"):
        super().__init__()
        self.occ_grid_size = occ_grid_size

        self.col_model = ColChecker(state_dim + linkpos_dim, occ_grid_size)
        self.sel_model = SelectorSmall(state_dim + linkpos_dim, state_dim + linkpos_dim + 1, occ_grid_size)

        if col_checker_path is not None:
            self.col_model.load_state_dict(torch.load(col_checker_path))

        if selector_path is not None:
            self.sel_model.load_state_dict(torch.load(selector_path))

        self.col_model.to(device)
        self.sel_model.to(device)
        self.col_model.eval()
        self.sel_model.eval()
        # self.col_model = torch.jit.trace(self.col_model)
        # self.sel_model = torch.jit.trace(self.sel_model)

        self.col_model = torch.jit.optimize_for_inference(torch.jit.script(self.col_model))
        self.sel_model = torch.jit.optimize_for_inference(torch.jit.script(self.sel_model))

    def forward(self, occ_grid, start, goal, sample, fixed_env=True):
        col_score = self.col_model(occ_grid, start, sample, fixed_env)
        sel_score = self.sel_model(occ_grid, start, goal, sample, fixed_env)
        sel_score = col_score * sel_score
        return col_score, sel_score

    def get_col_score(self, occ_grid, start, sample):
        col_score = self.col_model(occ_grid, start, sample, fixed_env=True)
        return col_score

    def get_sel_score(self, occ_grid, start, goal, samples):
        sel_score = self.sel_model(occ_grid, start, goal, samples, fixed_env=True)
        return sel_score

    def get_final_sel_scores(self, occ_grid, start, goal, samples, fixed_env=False):
        col_scores = self.col_model(occ_grid, start, samples, fixed_env)
        sel_scores = self.sel_model(occ_grid, start, goal, samples, fixed_env)
        sel_scores = col_scores * sel_scores

        return sel_scores

    def select_from_samples(self, occ_grid, start, goal, samples, fixed_env=False):
        final_score = self.get_final_sel_scores(occ_grid, start, goal, samples, fixed_env)
        best_indice = torch.argmax(final_score)
        return best_indice

    def select_from_samples_heuristic(self, occ_grid, start, goal, samples, fixed_env=False):
        final_score = self.get_final_sel_scores(occ_grid, start, goal, samples, fixed_env)
        sel_good_indices = torch.nonzero(final_score >= 0.5).view(-1)

        sel_filtered_samples_t = samples[sel_good_indices]
        # print(filtered_samples_t.shape, sel_filtered_samples_t.shape)
        # sel_filtered_samples_list = sel_filtered_samples_t.cpu().numpy()[:, :self.dim].tolist()

        # if self.visualize:
        #     utils.visualize_nodes_local(occ_grid_np, sel_filtered_samples_list, v, g,
        #         show=False, save=True, file_name = osp.join(self.log_dir, "sel_valid_samples_viz_{}.png".format(self.i)))

        sel_num_good_samples = sel_filtered_samples_t.shape[0]

        if sel_num_good_samples > 0:
            goal_t = goal.unsqueeze(0).repeat(sel_num_good_samples, 1)
            # print(sel_filtered_samples_t.shape, goal_t.shape)

            # We should use rs path length. But it might take too much time
            diff = sel_filtered_samples_t[:, :self.dim] - goal_t
            diff[:, 3:] *= 0.125
            dist = torch.linalg.norm(diff, dim=-1).view(-1)

            sel_best_filtered_indice = torch.argmin(dist)
            sel_best_indice = sel_good_indices[sel_best_filtered_indice]
        else:
            sel_best_indice = torch.argmax(final_score)

        return sel_best_indice


class DiscriminativeSampler(nn.Module):
    def __init__(self, state_dim, occ_grid_dim=[1, 40, 40], selector_path=None, linkpos_dim=12, device="cuda"):
        super().__init__()
        self.occ_grid_dim = occ_grid_dim

        # self.col_model = ColCheckerSmall(state_dim + linkpos_dim, occ_grid_dim)
        self.sel_model = SelectorSmall(state_dim + linkpos_dim, state_dim + linkpos_dim + 1, occ_grid_dim)

        if selector_path is not None:
            self.sel_model.load_state_dict(torch.load(selector_path))

        self.col_model.to(device)
        self.sel_model.to(device)
        self.col_model.eval()
        self.sel_model.eval()
        # self.col_model = torch.jit.trace(self.col_model)
        # self.sel_model = torch.jit.trace(self.sel_model)

        self.col_model = torch.jit.optimize_for_inference(torch.jit.script(self.col_model))
        self.sel_model = torch.jit.optimize_for_inference(torch.jit.script(self.sel_model))

    def forward(self, occ_grid, start, goal, sample, fixed_env: bool = True):
        col_score = self.col_model(occ_grid, start, sample, fixed_env)
        sel_score = self.sel_model(occ_grid, start, goal, sample, fixed_env)
        sel_score = col_score * sel_score
        return col_score, sel_score

    def get_col_score(self, occ_grid, start, sample):
        col_score = self.col_model(occ_grid, start, sample, fixed_env=True)
        return col_score

    def get_sel_score(self, occ_grid, start, goal, samples, fixed_env=True):
        sel_score = self.sel_model(occ_grid, start, goal, samples, fixed_env=fixed_env)
        return sel_score

    def get_final_sel_scores(self, occ_grid, start, goal, samples, fixed_env=False):
        col_scores = self.col_model(occ_grid, start, samples, fixed_env)
        sel_scores = self.sel_model(occ_grid, start, goal, samples, fixed_env)
        sel_scores = col_scores * sel_scores

        return sel_scores

    def select_from_samples(self, occ_grid, start, goal, samples, fixed_env=False):
        final_score = self.get_final_sel_scores(occ_grid, start, goal, samples, fixed_env)
        best_indice = torch.argmax(final_score)
        return best_indice

    def select_from_samples_heuristic(self, occ_grid, start, goal, samples, fixed_env=False):
        final_score = self.get_final_sel_scores(occ_grid, start, goal, samples, fixed_env)
        sel_good_indices = torch.nonzero(final_score >= 0.5).view(-1)

        sel_filtered_samples_t = samples[sel_good_indices]
        # print(filtered_samples_t.shape, sel_filtered_samples_t.shape)
        # sel_filtered_samples_list = sel_filtered_samples_t.cpu().numpy()[:, :self.dim].tolist()

        # if self.visualize:
        #     utils.visualize_nodes_local(occ_grid_np, sel_filtered_samples_list, v, g,
        #         show=False, save=True, file_name = osp.join(self.log_dir, "sel_valid_samples_viz_{}.png".format(self.i)))

        sel_num_good_samples = sel_filtered_samples_t.shape[0]

        if sel_num_good_samples > 0:
            goal_t = goal.unsqueeze(0).repeat(sel_num_good_samples, 1)
            # print(sel_filtered_samples_t.shape, goal_t.shape)

            # We should use rs path length. But it might take too much time
            diff = sel_filtered_samples_t[:, :self.dim] - goal_t
            diff[:, 3:] *= 0.125
            dist = torch.linalg.norm(diff, dim=-1).view(-1)

            sel_best_filtered_indice = torch.argmin(dist)
            sel_best_indice = sel_good_indices[sel_best_filtered_indice]
        else:
            sel_best_indice = torch.argmax(final_score)

        return sel_best_indice


class SelectModelSmallV3(nn.Module):
    def __init__(self, state_dim, goal_dim, occ_grid_size=40, h_dim=512):
        super().__init__()
        self.state_dim = state_dim
        self.h_dim = h_dim
        self.goal_dim = goal_dim

        self.feature_extractor = CNNEncoder(1, (1, occ_grid_size, occ_grid_size))
        occ_feat_size = self.feature_extractor.flat_fts
        print("occ feat size = {}".format(occ_feat_size))

        # self.feature_extractor = UNet(1, state_dim)
        # feat_size = 100 * state_dim

        self.layers = nn.Sequential(
            nn.Linear(state_dim*2 + self.goal_dim + occ_feat_size, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, 2),
            nn.Sigmoid()
        )

        # self.apply(init_weight)

    def forward(self, occ_grid, start, goal, samples, fixed_env=True):
        """
        occ_grid: bs x 4 x occ_dims
        start : bs x dim
        goal: bs x goal_dim
        samples: bs x dim
        """
        # print(occ_grid.shape, start.shape, goal.shape, samples.shape)
        start_time = time.perf_counter()
        bs = occ_grid.shape[0]
        f = self.feature_extractor(occ_grid)
        # torch.cuda.synchronize() # wait for mm to finish
        # end_time = time.perf_counter()
        # print("1", end_time - start_time)

        num_of_sample = 1
        if fixed_env:
            num_of_sample = samples.shape[1]
            f = f.unsqueeze(1)
            f = f.repeat(1, num_of_sample, 1).view(bs * num_of_sample, -1)  # (bs * N) * f_dim
            start = start.unsqueeze(1).repeat(1, num_of_sample, 1).view(bs * num_of_sample, -1)  # (bs * N) * dim
            goal = goal.unsqueeze(1).repeat(1, num_of_sample, 1).view(bs * num_of_sample, -1)  # (bs * N) * dim
            samples = samples.view(bs * num_of_sample, -1)  # (bs * N) * dim

        x = torch.cat((f, start, goal, samples), dim=-1)  # (bs * N) * (|f| + dim * 2 + goal_dim)
        # torch.cuda.synchronize() # wait for mm to finish
        # end_time = time.perf_counter()
        # print("2", end_time - start_time)

        y = self.layers(x)

        # torch.cuda.synchronize() # wait for mm to finish
        # end_time = time.perf_counter()
        # print("3", end_time - start_time)

        # forward pass
        y = y.view(bs, num_of_sample, 2)

        torch.cuda.synchronize()  # wait for mm to finish
        end_time = time.perf_counter()
        print("4", end_time - start_time)

        return y


if __name__ == "__main__":
    import time

    model = SelectModelSmall(8, 40)
    # model = SelectModelSmallV2(20, 21, 40)
    # model.eval()
    # model.to("cuda")
    occ_grid = torch.zeros((1, 1, 40, 40), device="cuda")
    start = torch.zeros((1, 20), device="cuda")
    goal = torch.zeros((1, 21), device="cuda")
    samples = torch.zeros((1, 1250, 20), device="cuda")

    with torch.no_grad():
        for _ in range(10):
            scores = model.get_final_sel_scores(occ_grid, start, goal, samples, fixed_env=True)

        start_time = time.time()
        for _ in range(1000):
            start_time_1 = time.time()
            scores = model.get_final_sel_scores(occ_grid, start, goal, samples, fixed_env=True)
            # scores = model(occ_grid, start, goal, samples, fixed_env=False)
            torch.cuda.synchronize()  # wait for mm to finish
            end_time_1 = time.time()
            print("5", end_time_1 - start_time_1)

    # torch.cuda.synchronize() # wait for mm to finish
    end_time = time.time()
    print((end_time - start_time) / 1000)
