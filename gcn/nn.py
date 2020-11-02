import torch
import torch.nn as nn
from torchsummary import summary


class GCN(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, out_len):
        super(GCN, self).__init__()
        hidden_len = 32
        self.ipt_hidden = Layer(in_channel, hidden_channel, hidden_len)
        # self.hidden_out =

    def forward(self, x):
        x = self.ipt_hidden(x)
        x = self.hidden_out(x)
        return x

# class GCN(nn.Module):
#     def __init__(self, C, H, F):
#         super(GCN, self).__init__()
#         self.relu = nn.ReLU()
#         self.soft_max = nn.Softmax(-1)
#         self.fc1 = nn.Linear(C, H)
#         self.fc2 = nn.Linear(H, F)
#
#     def forward(self, hatA, X):
#         y = torch.bmm(hatA, X)
#         y = self.fc1(y)
#         y = self.relu(y)
#         y = torch.bmm(hatA, y)
#         y = self.fc2(y)
#         y = self.soft_max(y)
#         return y


# class GCN(nn.Module):
#     def __init__(self, N, M, in_channel, hidden_channel, L1, out_channel, L2):
#         super(GCN, self).__init__()
#         self.N = N
#         self.M = M
#         self.in_channel = in_channel
#         self.hidden_channel = hidden_channel
#         self.out_channel = out_channel
#         self.L1 = L1
#         self.L2 = L2
#         self.relu = nn.ReLU()
#         self.soft_max = nn.Softmax()
#         self.layer1 = Layer(in_channel, hidden_channel, L1)
#         self.layer2 = Layer(hidden_channel, out_channel, L2)
#
#     def forward(self, hatA, X):
#         """
#         :param hatA: (B, N, N)
#         :param X: (B, N, C_in, 1)
#         :return:
#         """
#         y = torch.bmm(hatA, X.reshape(-1, self.N, self.in_channel))
#         y = self.layer1(y.reshape(-1, self.in_channel, 1)).reshape(-1, self.N, self.hidden_channel, self.L1)
#         y = self.relu(y)
#
#         y = torch.bmm(hatA, y.reshape(y.shape[0], self.N, -1))
#         y = self.layer2(y.reshape(-1, self.hidden_channel, self.L1))
#         return y


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


class Layer(nn.Module):
    def __init__(self, in_channel, out_channel, L):
        super(Layer, self).__init__()
        self.conv = nn.ConvTranspose1d(in_channel, out_channel, L)

    def forward(self, x):
        x = self.conv(x)
        return x


if __name__ == '__main__':
    m = GCN(3, 10, 3, 10)
    m1 = DQN(5)
    m2 = Layer(3, 3, 32)
    # summary(m, input_size=(3, 1), device="cpu", batch_size=5)
    # summary(m1, (5,), device="cpu")
    summary(m2, input_size=(3, 1), device="cpu")
