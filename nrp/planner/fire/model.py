import torch
import torch.nn as nn

class FireModel(nn.Module):
    def __init__(self, robot_dim=11, proj_dim=36):
        super(FireModel, self).__init__()
        self.channel_mult = 16
        self.h_dim = 1024
        self.num_linear = 2

        self.occ_grid_feature_extractor = nn.Sequential(
            nn.Conv3d(4, self.channel_mult * 1, kernel_size=3),
            nn.BatchNorm3d(self.channel_mult * 1, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool3d(2),
            nn.Conv3d(self.channel_mult * 1, self.channel_mult * 1, kernel_size=3),
            nn.BatchNorm3d(self.channel_mult * 1, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(self.channel_mult * 1, self.channel_mult * 1, kernel_size=3),
            nn.BatchNorm3d(self.channel_mult * 1, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Flatten(),
            nn.Linear(2000, 128)
        )

        # occ_feat_size = self.occ_grid_feature_extractor.output_size
        occ_feat_size = 128
        print("occ feat size = {}".format(occ_feat_size))

        self.q_feature_extractor = nn.Sequential(
            nn.Linear(robot_dim + proj_dim + 3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
        )

        self.head = nn.Linear(16 + occ_feat_size, 8)

    def forward(self, occ_grid, occ_grid_center, q_target, q_proj):
        occ_grid_f = self.occ_grid_feature_extractor(occ_grid)
        # print(q_target.shape, q_proj.shape, occ_grid_center.shape)
        # print(q_target.dtype, q_proj.dtype, occ_grid_center.dtype)
        q_f = self.q_feature_extractor(torch.cat((q_target, q_proj, occ_grid_center), axis=-1))

        # print(occ_grid_f.shape, q_f.shape)
        f = torch.cat((occ_grid_f, q_f), axis=-1)
        z = self.head(f)
        return z
