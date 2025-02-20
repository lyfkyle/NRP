import torch.nn as nn
import torch
from pytorch3d.transforms import Transform3d

class FkModel(nn.Module):
    def __init__(self, state_dim, output_dim):
        super().__init__()
        self.state_dim = state_dim
        self.h_dim = 256
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(state_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.LeakyReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(self.h_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.output_dim)
        )

    def forward(self, state):
        x = self.layers(state)
        y = self.head(x)
        return y

class ProxyFkTorch():
    def __init__(self, robot_dim, linkpos_dim, fkmodel_path, device):
        self.d = device
        self.dtype = torch.float
        self.fk_model = FkModel(robot_dim, linkpos_dim)
        self.fk_model.load_state_dict(torch.load(fkmodel_path))
        self.fk_model.eval()
        self.fk_model.to(device)
        self.scripted_model = torch.jit.script(self.fk_model)

        self.robot_dim = robot_dim
        self.linkpos_dim = linkpos_dim

    def set_device(self, device):
        self.d = device
        self.fk_model.to(device)

    @torch.no_grad()
    def get_link_positions(self, robot_state):
        # N = robot_state.shape[0]
        # th_batch = torch.rand(N, len(self.chain.get_joint_parameter_names()), dtype=self.dtype, device=self.d)
        robot_state = robot_state.to(self.d)

        # we want to forward the base offset but inverse the rotation. So we pass in the negative base offset
        bs = robot_state.shape[0]
        base_offset = torch.zeros((bs, 3), dtype=self.dtype, device=self.d)
        base_offset[:, :2] = -robot_state[:, :2] # So we pass in the negative base offset
        transform3d = Transform3d(dtype=self.dtype, device=self.d)
        transform3d = transform3d.translate(base_offset).rotate_axis_angle(robot_state[:, 2], axis="Z", degrees=False).inverse()
        # print(transform3d.get_matrix())

        robot_state_t = torch.zeros_like(robot_state).view(-1, self.robot_dim)
        robot_state_t[:, 3:] = robot_state[:, 3:]
        linkpos = self.scripted_model(robot_state_t)

        linkpos = linkpos.view(bs, -1, 3)
        linkpos = transform3d.transform_points(linkpos)
        linkpos = linkpos.view(bs, -1)

        return linkpos