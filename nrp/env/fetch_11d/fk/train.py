import os.path as osp
import pickle
from torch.utils.tensorboard import SummaryWriter
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

CUR_DIR = osp.dirname(osp.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = osp.join(CUR_DIR, "dataset")

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

class MyDataset(Dataset):
    def __init__(self, dataset_size, transform=None, target_transform=None, device="cpu"):
        print("Instantiating dataset...")
        self.device = device
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        file_path = osp.join(data_dir, "data_{}.pkl".format(idx))
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        robot_state, linkpos = data

        robot_state = torch.Tensor(robot_state)
        linkpos = torch.Tensor(linkpos)

        return robot_state, linkpos

robot_dim = 11
linkpos_dim = 24
data_size = 100000
bs = 128
epoch = 20
lr = 0.0001
train_num = 0

model_name = "model_fk_v2"
model_path = osp.join(CUR_DIR, "models/{}.pt".format(model_name))

writer = SummaryWriter(comment = '_{}'.format(model_name))
model = FkModel(robot_dim, linkpos_dim)
model.to(device)
mse_loss = nn.MSELoss(reduction="sum")

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15000, verbose=True, factor=0.5)

best_loss = float('inf')
dataset = MyDataset(data_size, None, None)
train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size
print("train size: {}. test_size: {}".format(train_size, test_size))
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_set, batch_size=bs, shuffle=True)
eval_dataloader = DataLoader(val_set, batch_size=1, shuffle=True)

for epoch_num in range(epoch):
    for data in train_dataloader:
        robot_state, linkpos = data

        robot_state = robot_state.to(device)
        linkpos = linkpos.to(device)

        # Perform forward pass
        pred_linkpos = model(robot_state)

        # loss
        loss = mse_loss(pred_linkpos, linkpos)

        # Zero the gradients
        optimizer.zero_grad()

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        scheduler.step(loss)

        # Print statistics
        if train_num % 100 == 0:
            print("Learning rate: {}".format(optimizer.param_groups[0]['lr']))
            print('Loss after epoch %d, mini-batch %5d, loss: %.3f' % (epoch_num, train_num, loss.item()))
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], train_num)
            writer.add_scalar('col_loss/train', loss.item(), train_num)

        train_num += 1

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), osp.join(CUR_DIR, "models/{}_best.pt".format(model_name)))

    torch.save(model.state_dict(), model_path)
    print("saved session to ", model_path)

model.load_state_dict(torch.load(model_path))
model.eval()
total_loss = 0
avg_time = 0
with torch.no_grad():
    for data in eval_dataloader:
        robot_state, linkpos = data

        robot_state = robot_state.to(device)
        linkpos = linkpos.to(device)

        # Perform forward pass
        start_time = time.time()
        pred_linkpos = model(robot_state)
        end_time = time.time()
        avg_time += end_time - start_time

        # print(robot_state, pred_linkpos, linkpos)

        # loss
        loss = mse_loss(pred_linkpos, linkpos)

        total_loss += loss.item()

total_loss /= test_size
avg_time /= test_size
print(total_loss, avg_time)


