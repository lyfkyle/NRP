import torch
import torch.nn as nn
import os.path as osp

CUR_DIR = osp.dirname(osp.abspath(__file__))

class CNN_Encoder(nn.Module):
    def __init__(self, output_size, input_size=(1, 40, 40)):
        super(CNN_Encoder, self).__init__()

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
            nn.Conv2d(self.channel_mult*4, self.channel_mult*8, 4, 2, 1), # out = (128, 1, 1)
            nn.BatchNorm2d(self.channel_mult*8),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        self.flat_fts = self.get_flat_fts(self.conv)
        print("flat_fts_size:", self.flat_fts)

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

class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, input_size=(1, 10, 10)):
        super(CNN_Decoder, self).__init__()
        self.input_height = input_size[1]
        self.input_width = input_size[2]
        self.input_dim = embedding_size
        self.channel_mult = 16
        self.output_channels = 1
        self.fc_output_dim = 128 # 512

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_output_dim),
            nn.BatchNorm1d(self.fc_output_dim),
            nn.LeakyReLU()
        )

        self.deconv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.fc_output_dim, self.channel_mult*4, 4, 2, 1, bias=False), # out =(64, 2, 2)
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2, 3, 2, 1, bias=False), # out =(32, 3, 3)
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*1, 3, 2, 1, bias=False), # out =(16, 5, 5)
            nn.BatchNorm2d(self.channel_mult*1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.channel_mult*1, self.output_channels, 4, 2, 1, bias=False), # out =(1, 10, 10)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.fc_output_dim, 1, 1)
        x = self.deconv(x)
        return x

class AE(nn.Module):
    def __init__(self, embedding_size, occ_grid_dim):
        super(AE, self).__init__()
        print(embedding_size, occ_grid_dim)
        self.encoder = CNN_Encoder(embedding_size, input_size=(1, occ_grid_dim, occ_grid_dim))
        self.decoder = CNN_Decoder(embedding_size, input_size=(1, occ_grid_dim, occ_grid_dim))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

class Encoder(nn.Module):
    def __init__(self, occ_grid_dim, latent_size, context_size, input_size, feat_extractor_model_path=None):
        super().__init__()
        print("Instantiating encoder...")
        self.ae = AE(16, occ_grid_dim)
        if feat_extractor_model_path is not None:
            print("using pretrained feature extractor : {}".format(feat_extractor_model_path))
            self.ae.load_state_dict(torch.load(feat_extractor_model_path))
        self.feature_extractor = self.ae.encoder
        occ_feat_size = self.feature_extractor.flat_fts

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size + context_size + occ_feat_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.linear_means = nn.Linear(512, latent_size)
        self.linear_log_var = nn.Linear(512, latent_size)

    def forward(self, sample, occ_grid, context):
        f = self.feature_extractor.extract_feat(occ_grid)
        x1 = torch.cat((sample, context, f), dim = -1)
        x2 = self.layers(x1)

        means = self.linear_means(x2)
        log_vars = self.linear_log_var(x2)

        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, occ_grid_dim, latent_size, context_size, out_size, feat_extractor_model_path=None):
        super().__init__()
        print("Instantiating decoder...")
        self.ae = AE(16, occ_grid_dim)
        if feat_extractor_model_path is not None:
            print("using pretrained feature extractor : {}".format(feat_extractor_model_path))
            self.ae.load_state_dict(torch.load(feat_extractor_model_path))
        self.feature_extractor = self.ae.encoder
        occ_feat_size = self.feature_extractor.flat_fts

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_size + context_size + occ_feat_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.recon_layer = nn.Linear(512, out_size)

    def forward(self, z, occ_grid, c):
        f = self.feature_extractor.extract_feat(occ_grid)
        x1 = torch.cat((z, c, f), dim = -1)
        x2 = self.layers(x1)
        y = self.recon_layer(x2)
        return y

class VAE(nn.Module):
    def __init__(self, occ_grid_dim, latent_size, context_size, state_size, feat_extractor_model_path=None):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(occ_grid_dim, latent_size, context_size, state_size, feat_extractor_model_path)
        self.decoder = Decoder(occ_grid_dim, latent_size, context_size, state_size, feat_extractor_model_path)

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
        occ_grid_tmp = occ_grid.repeat(num_samples, 1, 1, 1)
        state_tmp = state.repeat(num_samples, 1)
        samples = self.inference(z, occ_grid_tmp, state_tmp)
        return samples

class GenerativeSampler(nn.Module):
    def __init__(self, occ_grid_dim, latent_size, context_size, state_size):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(occ_grid_dim, latent_size, context_size, state_size)
        self.decoder = Decoder(occ_grid_dim, latent_size, context_size, state_size)

    def forward(self, num_samples:int, occ_grid, state):
        z = torch.randn(num_samples, self.latent_size, device=occ_grid.device)
        occ_grid_tmp = occ_grid.repeat(num_samples, 1, 1, 1)
        state_tmp = state.repeat(num_samples, 1)
        samples = self.inference(z, occ_grid_tmp, state_tmp)
        return samples

    def inference(self, z, occ_grid, c):
        recon_x = self.decoder(z, occ_grid, c)
        return recon_x
