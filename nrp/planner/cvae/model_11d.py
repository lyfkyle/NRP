import torch
import torch.nn as nn
import os.path as osp

CUR_DIR = osp.dirname(osp.abspath(__file__))

class VoxNetEncoder(nn.Module):
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

class Encoder(nn.Module):
    def __init__(self, latent_size, context_size, state_size, h_dim=1024):
        super().__init__()
        print("Instantiating encoder...")
        self.feature_extractor = VoxNetEncoder(in_channels=4, input_size=[4, 150, 150, 20])
        occ_feat_size = self.feature_extractor.output_size
        print("occ feat size = {}".format(occ_feat_size))

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_size + context_size + occ_feat_size, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
        )

        self.linear_means = nn.Linear(h_dim, latent_size)
        self.linear_log_var = nn.Linear(h_dim, latent_size)

    def forward(self, sample, occ_grid, context):
        f = self.feature_extractor.extract_feat(occ_grid)
        x1 = torch.cat((sample, context, f), dim = -1)
        x2 = self.layers(x1)

        means = self.linear_means(x2)
        log_vars = self.linear_log_var(x2)

        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, latent_size, context_size, state_size, h_dim=1024):
        super().__init__()
        print("Instantiating decoder...")
        self.feature_extractor = VoxNetEncoder(in_channels=4, input_size=[4, 150, 150, 20])
        occ_feat_size = self.feature_extractor.output_size
        print("occ feat size = {}".format(occ_feat_size))

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_size + context_size + occ_feat_size, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
        )

        self.recon_layer = nn.Linear(h_dim, state_size)

    def forward(self, z, occ_grid, c):
        f = self.feature_extractor.extract_feat(occ_grid)
        x1 = torch.cat((z, c, f), dim = -1)
        x2 = self.layers(x1)
        y = self.recon_layer(x2)
        return y

class VAE(nn.Module):
    def __init__(self, latent_size, context_size, state_size):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(latent_size, context_size, state_size)
        self.decoder = Decoder(latent_size, context_size, state_size)

    def forward(self, x, occ_grid, c):
        means, log_var = self.encoder(x, occ_grid, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, occ_grid, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z, occ_grid, c):
        recon_x = self.decoder(z, occ_grid, c)
        return recon_x

    def sample(self, num_samples, occ_grid, state):
        z = torch.randn(num_samples, self.latent_size, device=occ_grid.device)
        occ_grid_tmp = occ_grid.repeat(num_samples, 1, 1, 1, 1)
        state_tmp = state.repeat(num_samples, 1)
        samples = self.inference(z, occ_grid_tmp, state_tmp)
        return samples

class VAEInference(nn.Module):
    def __init__(self, latent_size, context_size, state_size):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(latent_size, context_size, state_size)
        self.decoder = Decoder(latent_size, context_size, state_size)

    def forward(self, num_samples: int, occ_grid, state):
        z = torch.randn(num_samples, self.latent_size, device=occ_grid.device)
        occ_grid_tmp = occ_grid.repeat(num_samples, 1, 1, 1, 1)
        state_tmp = state.repeat(num_samples, 1)
        samples = self.inference(z, occ_grid_tmp, state_tmp)
        return samples

    def inference(self, z, occ_grid, c):
        recon_x = self.decoder(z, occ_grid, c)
        return recon_x
