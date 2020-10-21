import torch
import torch.nn as nn
from torchsummary import summary


class GCN(nn.Module):
    def __init__(self, C, H, F):
        super(GCN, self).__init__()
        self.relu = nn.ReLU()
        self.soft_max = nn.Softmax()
        self.fc1 = nn.Linear(C, H)
        self.fc2 = nn.Linear(H, F)

    def forward(self, hatA, X):
        y = torch.bmm(hatA, X)
        y = self.fc1(y)
        y = self.relu(y)
        y = torch.bmm(hatA, y)
        y = self.fc2(y)
        y = self.soft_max(y)
        return y


class DQN(nn.Module):
    def __init__(self, ipt_len):
        super(DQN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(ipt_len, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)

        self.BN1 = nn.BatchNorm1d(32)
        self.BN2 = nn.BatchNorm1d(64)
        self.BN3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.fc1(x)
        x = self.BN1(x)
        self.relu(x)

        x = self.fc2(x)
        x = self.BN2(x)
        self.relu(x)

        x = self.fc3(x)
        x = self.BN3(x)
        self.relu(x)

        x = self.fc4(x)
        x = self.BN2(x)
        self.relu(x)

        x = self.fc5(x)
        x = self.BN1(x)
        self.relu(x)

        x = self.fc6(x)
        self.relu(x)

        return x


if __name__ == '__main__':
    m = GCN(3, 10, 5)
    m1 = DQN(5)
    # summary(m, input_size=([(20, 20), (20, 3)]), batch_size=3, device="cpu")
    summary(m1, (5,), device="cpu")