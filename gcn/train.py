from utils import load_data, arg_parser
from gcn.nn import GCN, DQN
import torch.optim as optim
from gcn.core import Env
import time

args = arg_parser()

# total number of agent
M = 3
# total number of cities
N = 20
# vertex state channels
C = 3
# GCN hidden embedding state: (N, H)
H = 20
# GCN embedding state: (N, F)
F = 10
# total number of steps
S = 10
# budget
BUDGET = 80

# load data
data_loader = load_data(args.len_dataset, args.connectivity_path, args.task_path,
                        args.city_path, args.reward_path, args.destination_path)
data = list(data_loader)

# model and optimizer
g_model = GCN(C, H, F)
q_model = DQN(F)
optimizer = optim.Adam([{'params': g_model.parameters()},
                        {'params': q_model.parameters()}],
                       lr=args.lr),

# environment
# conn: (N, N)
# rewards( vertex features): (N, C)
env = Env(N, M, S,
          {"conn": data[0]["conn"].squeeze(),
           "rewards": data[0]["rewards"].squeeze().T},
          BUDGET)


def train(epoch):
    t = time.time()
    q_model.train()
    g_model.train()
    optimizer.zero_grad()






print()






