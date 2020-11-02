from utils import load_data, arg_parser
from gcn.nn import GCN, DQN, Layer
import torch.optim as optim
from gcn.core import Env
import time
import random
from memory import ReplayMemoryWithSet
import torch
import math

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
F = 3
# total number of steps
STEPS = 10
# budget
BUDGET = 80
# Replay Memory Capacity
CAPACITY = 1000
# epsilon greedy
EPS_START = 0.9
EPS_END = 0.05
# if epoch = EPS_DECAY, P = 0.363
EPS_DECAY = 20000

# load data
data_loader = load_data(args.len_dataset, args.connectivity_path, args.task_path,
                        args.city_path, args.reward_path, args.destination_path)
data = list(data_loader)

# model and optimizer
g_model = Layer(3, 3, 32)
q_model = DQN(33)
optimizer = optim.Adam([{'params': g_model.parameters()},
                        {'params': q_model.parameters()}],
                       lr=args.lr)

# environment
# conn: (N, N)
# rewards( vertex features): (N, C)
env = Env(N, M, STEPS,
          {"conn": data[0]["conn"].squeeze(),
           "rewards": data[0]["rewards"].squeeze().T,
           "tasks": data[0]["tasks"].squeeze()},
          BUDGET)

g_model.cuda()
q_model.cuda()

# Graph Laplacian matrix
waveA = env.G.adj().float().cuda()
waveD = env.G.deg(waveA).float().cuda()
# (20, 20)
hatA = torch.chain_matmul(torch.inverse(torch.sqrt(waveD)),
                          waveA,
                          torch.inverse(torch.sqrt(waveD)))

# Replay Memory
memory = ReplayMemoryWithSet(1000)


def forwward_step(step, eps_threshold):
    # (20, 3)
    X = env.G.X().float().cuda()
    # (20, 3, 32)
    X_theta = g_model(X.unsqueeze(-1))
    # (20, 96)
    Z = torch.mm(hatA, X_theta.reshape(X_theta.shape[0], -1))
    Z = Z.reshape(Z.shape[0], M, -1)
    agent_states = []
    for agent in range(M):
        state = env.agents[agent].state()
        state = torch.nn.functional.normalize(state, dim=0)
        agent_states.append(state.unsqueeze(1))
    agent_states = torch.cat(agent_states, dim=1).float().cuda()
    agent_states = torch.cat((Z, agent_states), dim=-1)
    Q = q_model(agent_states.reshape(-1, agent_states.shape[-1]))
    Q = Q.reshape(N, M, -1)
    actions = []
    for agent in range(M):
        p = random.random()
        if p > eps_threshold:
            action = Q[:, agent, :].max(0)[1].item()
            actions.append(action)
        else:
            actions.append(random.randint(0, N-1))

    print()


def train(epoch):
    t = time.time()
    q_model.train()
    g_model.train()
    optimizer.zero_grad()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * epoch / EPS_DECAY)
    for step in range(STEPS):
        forwward_step(step, eps_threshold)
    print()


train(0)






