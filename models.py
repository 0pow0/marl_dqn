import torch.nn as nn
from torchsummary import summary


class Encoder(nn.Module):
    def __init__(self, K, M, L):
        super(Encoder, self).__init__()
        self.in_shape = 1 + 2 * M + 2 * K * M
        self.out_shape = L

        self.fc1 = nn.Linear(self.in_shape, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.out_shape)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class DQN(nn.Module):
    """
    :param N: amount of uav
    :param K: steps
    :param L: dim of out in embedding
    :param M: amount of cities
    """

    def __init__(self, N, K, L, M):
        super(DQN, self).__init__()

        self.N = N
        self.K = K
        self.L = L
        self.M = M

        # with encoder
        # self.len_state = 1 + 2 * M + 2 * K * M + (N - 1) * L
        # without encoder
        self.len_state = 1 + 2 * M + 2 * K * M + (N - 1) * (1 + 2 * M + 2 * K * M)

        self.out1 = 32
        self.out2 = 64
        self.out3 = 128
        self.out4 = 256

        self.ReLu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.out1, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=self.out1, out_channels=self.out2, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=self.out2, out_channels=self.out3, kernel_size=4)
        self.conv4 = nn.Conv1d(in_channels=self.out3, out_channels=self.out4, kernel_size=4)

        self.convBN1 = nn.BatchNorm1d(self.out1)
        self.convBN2 = nn.BatchNorm1d(self.out2)
        self.convBN3 = nn.BatchNorm1d(self.out3)
        self.convBn4 = nn.BatchNorm1d(self.out4)

        self.convPool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(((self.len_state - 1) // 2) * self.out1, 2 * self.M)

    def forward(self, x):
        # print(x.shape)

        x = self.conv1(x)
        x = self.convBN1(x)
        x = self.ReLu(x)

        # x = self.conv2(x)
        # x = self.convBN2(x)
        # x = self.ReLu(x)
        #
        # x = self.convPool(x)
        #
        # x = self.conv3(x)
        # x = self.convBN3(x)
        # x = self.ReLu(x)
        #
        # x = self.conv4(x)
        # x = self.convBn4(x)
        # x = self.ReLu(x)

        x = self.convPool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    m = DQN(3, 20, 100, 20)
    summary(m, input_size=(1, 1041), device="cpu")
