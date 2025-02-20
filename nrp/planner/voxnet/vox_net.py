import torch
import torch.nn as nn

class VoxNetEncoderGlobal(nn.Module):
    def __init__(self, in_channels, output_size=2048, input_size=[4, 150, 150, 20], kernel_size=3):
        super().__init__()
        self.channel_mult = 16
        self.input_size = input_size
        self.output_size = output_size

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, self.channel_mult * 1, kernel_size=[7, 7, 3], stride=2),
            nn.BatchNorm3d(self.channel_mult * 1, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(self.channel_mult * 1,  self.channel_mult * 2, kernel_size=[5, 5, 3], stride=1),
            nn.BatchNorm3d(self.channel_mult * 2, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool3d(2),
            nn.Conv3d(self.channel_mult * 2, self.channel_mult * 4, kernel_size=[5, 5, 3], stride=1),
            nn.BatchNorm3d(self.channel_mult * 4, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(self.channel_mult * 4, self.channel_mult * 8, kernel_size=[5, 5, 1], stride=1),
            nn.BatchNorm3d(self.channel_mult * 8, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Flatten()
        )

        self.flat_fts = self.get_flat_fts(self.conv)
        print(self.flat_fts)

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
        x = self.linear(x)
        return x

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.flat_fts)
        return self.linear(x)

class VoxNetEncoder(nn.Module):
    def __init__(self, in_channels, output_size=4096, input_size=(4, 40, 40, 20), kernel_size=3):
        super().__init__()
        self.channel_mult = 32
        self.input_size = input_size
        self.output_size = output_size

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, self.channel_mult * 1, kernel_size=5, stride=2),
            nn.BatchNorm3d( self.channel_mult * 1, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d( self.channel_mult * 1,  self.channel_mult * 1, kernel_size=3, stride=1),
            nn.BatchNorm3d(self.channel_mult * 1, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool3d(2),
            # nn.Conv3d(self.channel_mult * 2, self.channel_mult * 4, kernel_size, stride=1),
            # nn.BatchNorm3d(self.channel_mult * 4, eps=1e-4),
            # nn.LeakyReLU(0.1, True),
            # nn.Conv3d(self.channel_mult * 4, self.channel_mult * 8, kernel_size, stride=1, padding=kernel_size // 2),
            # nn.BatchNorm3d(self.channel_mult * 8, eps=1e-4),
            # nn.LeakyReLU(0.1, True),
            nn.Flatten()
        )

        self.flat_fts = self.get_flat_fts(self.conv)
        print(self.flat_fts)

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
        return self.linear(x)

class VoxNetEncoderMagic(nn.Module):
    def __init__(self, in_channels, input_size=(4, 40, 40, 20)):
        super().__init__()
        self.channel_mult = 64
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, self.channel_mult * 1, kernel_size=3),
            nn.BatchNorm3d(self.channel_mult * 1, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool3d(2),
            nn.Conv3d(self.channel_mult * 1, self.channel_mult * 1, kernel_size=3),
            nn.BatchNorm3d(self.channel_mult * 1, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool3d(2),
            nn.Conv3d(self.channel_mult * 1, self.channel_mult * 1, kernel_size=3),
            nn.BatchNorm3d(self.channel_mult * 1, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Flatten()
        )

        self.output_size = self.get_flat_fts(self.conv)
        print(self.output_size)

    def get_flat_fts(self, fts):
        tmp = torch.ones(1, *self.input_size)
        f = fts(tmp).view(1, -1)
        return f.shape[1]

    def extract_feat(self, x):
        x = self.conv(x)
        x = x.view(-1, self.output_size)
        return x

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.output_size)
        return x