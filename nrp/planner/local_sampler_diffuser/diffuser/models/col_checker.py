import torch
import torch.nn as nn

class CNNEncoderSmall(nn.Module):
    def __init__(self, output_size, input_size=(1, 10, 10)):
        super(CNNEncoderSmall, self).__init__()

        self.input_size = input_size
        self.channel_mult = 16

        #convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.channel_mult*1, kernel_size=4, stride=2, padding=1), # out = (16, 5, 5)
            nn.LeakyReLU(),
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 3, 2, 1), # out = (32, 3, 3)
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 3, 2, 1), # out = (64, 2, 2)
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        # print("here")
        self.flat_fts = self.get_flat_fts(self.conv)
        # print(self.flat_fts)
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
    def __init__(self, state_dim, occ_grid_size=100, h_dim=512):
        super().__init__()
        self.state_dim = state_dim
        self.h_dim = h_dim

        self.feature_extractor = CNNEncoderSmall(1, (1, occ_grid_size, occ_grid_size))
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

    def forward_trajectory(self, occ_grid, samples):
        """
        occ_grid: bs x 1 x occ_dims x occ_dims
        samples: bs x horizon x dim
        """
        # print(occ_grid.shape, start.shape, samples.shape)

        bs = occ_grid.shape[0]
        f = self.feature_extractor(occ_grid)

        num_of_sample = samples.shape[1]
        f = f.unsqueeze(1)
        f = f.repeat(1, num_of_sample, 1).view(bs * num_of_sample, -1) # (bs * N) * f_dim

        # start = start.unsqueeze(1).repeat(1, num_of_sample, 1).view(bs * num_of_sample, -1) # (bs * N) * dim
        start = torch.cat((samples[:, 0:1, :], samples[:, :-1, :]), dim=1)
        start = start.view(bs * num_of_sample, -1) # (bs * N) * dim
        samples = samples.view(bs * num_of_sample, -1)  # (bs * N) * dim

        x = torch.cat((f, start, samples), dim=-1) # (bs * N) * (|f| + dim * 2 + goal_dim)
        # print(x.shape)
        x = self.layers(x)
        y = self.head(x)

        # forward pass
        y = y.view(bs, num_of_sample)

        return y
    
    def forward(self, occ_grid, start, sample):
        """
        occ_grid: bs x 1 x occ_dims x occ_dims
        samples: bs x dim
        samples: bs x dim
        """
        # print(occ_grid.shape, start.shape, samples.shape)

        bs = occ_grid.shape[0]
        f = self.feature_extractor(occ_grid)

        num_of_sample = 1

        x = torch.cat((f, start, sample), dim=-1) # (bs * N) * (|f| + dim * 2 + goal_dim)
        # print(x.shape)
        x = self.layers(x)
        y = self.head(x)

        # forward pass
        y = y.view(bs, num_of_sample)

        return y