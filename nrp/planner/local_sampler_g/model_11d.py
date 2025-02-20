import torch
import torch.nn as nn
import os.path as osp

from nrp.planner.voxnet.vox_net import VoxNetEncoderMagic, VoxNetEncoderGlobal

CUR_DIR = osp.dirname(osp.abspath(__file__))


class Encoder(nn.Module):
    def __init__(self, latent_size, context_size, state_size, h_dim=1024, global_mode=False):
        super().__init__()
        print("Instantiating encoder...")
        if not global_mode:
            self.feature_extractor = VoxNetEncoderMagic(in_channels=4)
        else:
            self.feature_extractor = VoxNetEncoderGlobal(in_channels=4)
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
        x1 = torch.cat((sample, context, f), dim=-1)
        x2 = self.layers(x1)

        means = self.linear_means(x2)
        log_vars = self.linear_log_var(x2)

        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, latent_size, context_size, state_size, h_dim=1024, global_mode=False):
        super().__init__()
        print("Instantiating decoder...")
        if not global_mode:
            self.feature_extractor = VoxNetEncoderMagic(in_channels=4)
        else:
            self.feature_extractor = VoxNetEncoderGlobal(in_channels=4)
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
        x1 = torch.cat((z, c, f), dim=-1)
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


class VAEGlobal(nn.Module):
    def __init__(self, latent_size, context_size, state_size):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(latent_size, context_size, state_size, global_mode=True)
        self.decoder = Decoder(latent_size, context_size, state_size, global_mode=True)

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


class GenerativeSampler(nn.Module):
    def __init__(self, latent_size, context_size, state_size, global_mode=False):
        super().__init__()
        print("Instantiating sampler... global_mode={}".format(global_mode))
        self.latent_size = latent_size
        self.encoder = Encoder(latent_size, context_size, state_size, global_mode=global_mode)
        self.decoder = Decoder(latent_size, context_size, state_size, global_mode=global_mode)

    def forward(self, num_samples: int, occ_grid, state):
        if num_samples == -1:
            bs = occ_grid.shape[0]
            z = torch.randn(bs, self.latent_size, device=occ_grid.device)
            occ_grid_tmp = occ_grid
            state_tmp = state
        else:
            z = torch.randn(num_samples, self.latent_size, device=occ_grid.device)
            occ_grid_tmp = occ_grid.repeat(num_samples, 1, 1, 1, 1)
            state_tmp = state.repeat(num_samples, 1)
        samples = self.inference(z, occ_grid_tmp, state_tmp)
        return samples

    def inference(self, z, occ_grid, c):
        recon_x = self.decoder(z, occ_grid, c)
        return recon_x


class PointNetEncoder(nn.Module):
    def __init__(self, latent_size, context_size, state_size, h_dim=1024, global_mode=False):
        super().__init__()
        print("Instantiating encoder...")
        self.feature_extractor = PointNetPlusPlus(normal_channel=False)
        occ_feat_size = 1024

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

    def forward(self, sample, pc, context):
        f, _ = self.feature_extractor(pc)
        x1 = torch.cat((sample, context, f), dim=-1)
        x2 = self.layers(x1)

        means = self.linear_means(x2)
        log_vars = self.linear_log_var(x2)

        return means, log_vars


class PointNetDecoder(nn.Module):
    def __init__(self, latent_size, context_size, state_size, h_dim=1024, global_mode=False):
        super().__init__()
        print("Instantiating decoder...")
        self.feature_extractor = PointNetPlusPlus(normal_channel=False)
        occ_feat_size = 1024
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

    def forward(self, z, pc, c):
        f, _ = self.feature_extractor(pc)
        x1 = torch.cat((z, c, f), dim=-1)
        x2 = self.layers(x1)
        y = self.recon_layer(x2)
        return y


class PointNetVAE(nn.Module):
    def __init__(self, latent_size, context_size, state_size):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = PointNetEncoder(latent_size, context_size, state_size)
        self.decoder = PointNetDecoder(latent_size, context_size, state_size)

    def forward(self, x, pc, c):
        means, log_var = self.encoder(x, pc, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, pc, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z, pc, c):
        recon_x = self.decoder(z, pc, c)
        return recon_x

    def sample(self, num_samples, pc, state):
        z = torch.randn(num_samples, self.latent_size, device=pc.device)
        occ_grid_tmp = pc.repeat(num_samples, 1, 1, 1, 1)
        state_tmp = state.repeat(num_samples, 1)
        samples = self.inference(z, occ_grid_tmp, state_tmp)
        return samples
