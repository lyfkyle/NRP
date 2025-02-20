import os.path as osp
import sys
import pickle
import torch

from nrp.env.fetch_11d import utils
from nrp.env.fetch_11d.maze import Fetch11DEnv

CUR_DIR = osp.dirname(osp.abspath(__file__))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = osp.join(CUR_DIR, "dataset")

fk = utils.FkTorch(device)
env = Fetch11DEnv(gui=False)

high = env.robot.get_joint_higher_bounds()
low = env.robot.get_joint_lower_bounds()

low_t = torch.tensor(low, device=device)
high_t = torch.tensor(high, device=device)
low_t[:3] = 0
high_t[:3] = 0

num_to_sample = 100000
batch_size = 1000
data_cnt = 0
while data_cnt < num_to_sample:
    samples_t = torch.rand((batch_size, 11), device=device)
    samples_t = samples_t * (high_t - low_t) + low_t

    linkpos_t = fk.get_link_positions(samples_t)
    samples = samples_t.detach().cpu().numpy().tolist()
    linkpos = linkpos_t.detach().cpu().numpy().tolist()

    dataset = []
    for i in range(batch_size):
        dataset.append([samples[i], linkpos[i]])

    print("Saving dataset {} to {}".format(data_cnt, data_cnt + batch_size))
    for idx in range(batch_size):
        file_path = osp.join(data_dir, "data_{}.pkl".format(data_cnt + idx))
        with open(file_path, 'wb') as f:
            # print("Dumping to {}".format(file_path))
            pickle.dump(dataset[idx], f)

    data_cnt += len(dataset)


