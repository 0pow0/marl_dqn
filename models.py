import torch
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
        self.len_state = 1 + 1 + 2 * M + 2 * M + (N - 1) * (1 + 1 + 2 * M + 2 * M)

        self.out1 = 32
        self.out2 = 64
        self.out3 = 128
        self.out4 = 256

        self.ReLu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.out1, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=self.out1, out_channels=self.out2, kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=self.out2, out_channels=self.out3, kernel_size=5)
        self.conv4 = nn.Conv1d(in_channels=self.out3, out_channels=self.out4, kernel_size=5)
        self.conv5 = nn.Conv1d(in_channels=self.out4, out_channels=self.out3, kernel_size=3)
        self.conv6 = nn.Conv1d(in_channels=self.out3, out_channels=self.out2, kernel_size=3)
        self.conv7 = nn.Conv1d(in_channels=self.out2, out_channels=self.out1, kernel_size=3)
        self.conv8 = nn.Conv1d(in_channels=self.out1, out_channels=1, kernel_size=4, stride=1)

        self.convBN1 = nn.BatchNorm1d(self.out1)
        self.convBN2 = nn.BatchNorm1d(self.out2)
        self.convBN3 = nn.BatchNorm1d(self.out3)
        self.convBn4 = nn.BatchNorm1d(self.out4)

        self.convPool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.len_state, self.out4)
        self.fc1 = nn.Linear(self.out4, self.out3)
        self.fc2 = nn.Linear(self.out3, 2 * self.M)
        # self.fc1 = nn.Linear(((self.len_state - 1) // 2) * self.out1, 2 * self.M)

    def forward(self, x):

        # print(x.shape)
        x = self.fc(x)
        x = self.ReLu(x)
        x = self.fc1(x)
        x = self.ReLu(x)
        x = self.fc2(x)
        # x = self.conv1(x)
        # x = self.convBN1(x)
        # x = self.ReLu(x)

        # x = self.convPool(x)
        # x = self.flatten(x)
        # x = self.fc(x)
        x = torch.squeeze(x)
        return x


if __name__ == '__main__':
    m = DQN(3, 10, 100, 20)
    summary(m, input_size=(1, 1323), device="cpu")

"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv1d-1             [-1, 32, 2519]             192
       BatchNorm1d-2             [-1, 32, 2519]              64
              ReLU-3             [-1, 32, 2519]               0
            Conv1d-4             [-1, 64, 2515]          10,304
       BatchNorm1d-5             [-1, 64, 2515]             128
              ReLU-6             [-1, 64, 2515]               0
         MaxPool1d-7              [-1, 64, 502]               0
            Conv1d-8             [-1, 128, 498]          41,088
       BatchNorm1d-9             [-1, 128, 498]             256
             ReLU-10             [-1, 128, 498]               0
           Conv1d-11             [-1, 256, 494]         164,096
      BatchNorm1d-12             [-1, 256, 494]             512
             ReLU-13             [-1, 256, 494]               0
        MaxPool1d-14              [-1, 256, 97]               0
           Conv1d-15              [-1, 128, 93]         163,968
      BatchNorm1d-16              [-1, 128, 93]             256
             ReLU-17              [-1, 128, 93]               0
           Conv1d-18               [-1, 64, 89]          41,024
      BatchNorm1d-19               [-1, 64, 89]             128
             ReLU-20               [-1, 64, 89]               0
           Conv1d-21               [-1, 32, 85]          10,272
      BatchNorm1d-22               [-1, 32, 85]              64
             ReLU-23               [-1, 32, 85]               0
           Conv1d-24                [-1, 1, 40]             225
================================================================
Total params: 432,577
Trainable params: 432,577
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 10.78
Params size (MB): 1.65
Estimated Total Size (MB): 12.44
----------------------------------------------------------------
"""
