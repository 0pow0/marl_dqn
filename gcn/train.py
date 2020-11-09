from utils import load_data, arg_parser
from gcn.nn import GCN, DQN, Layer
import torch.optim as optim
from gcn.core import Env
import time
import random
from memory import ReplayMemoryWithSet, ExtendGraphTransition
import torch
import math


args = arg_parser()

# total number of agent
M = 3
# total number of cities
N = 20
# vertex state channels
C = 3
# length of GCN embedding state
L = 32
# Batch
B = args.batch_size
# GCN hidden embedding state: (N, H)
H = 20
# GCN embedding state: (N, F)
F = 3
# Gamma
GAMMA = args.gamma
# total number of steps
STEPS = 10
# budget
BUDGET = 80
# Replay Memory Capacity
CAPACITY = 100
# epsilon greedy
EPS_START = 0.9
EPS_END = 0.05
# if epoch = EPS_DECAY, P = 0.363
EPS_DECAY = 2000

# load data
data_loader = load_data(args.len_dataset, args.connectivity_path, args.task_path,
                        args.city_path, args.reward_path, args.destination_path)
data = list(data_loader)

# model and optimizer
g_model = Layer(3, 3, L)
q_model = DQN(L+1)
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
memory = ReplayMemoryWithSet(CAPACITY)


def forward_step(step, eps_threshold):
    # (20, 3)
    X = env.G.X()
    agent_states = []
    remain_budgets = []
    for agent in range(M):
        state, remain_budget = env.agents[agent].state()
        state = torch.nn.functional.normalize(state, dim=0)
        agent_states.append(state.unsqueeze(1))
        remain_budgets.append(remain_budget)
    agent_states = torch.cat(agent_states, dim=1)

    g_input = X.float().cuda()
    cuda_agent_states = agent_states.float().cuda()
    # (20, 3, 32)
    X_theta = g_model(g_input.unsqueeze(-1))
    # (20, 96)
    Z = torch.mm(hatA, X_theta.reshape(X_theta.shape[0], -1))
    Z = Z.reshape(Z.shape[0], M, -1)
    q_input = torch.cat((Z, cuda_agent_states), dim=-1)
    Q = q_model(q_input.reshape(-1, q_input.shape[-1]))
    Q = Q.reshape(N, M, -1)

    actions = []
    # epsilon-greedy
    for agent in range(M):
        p = random.random()
        if p > eps_threshold:
            action = Q[:, agent, :].max(0)[1].item()
            actions.append(action)
        else:
            actions.append(random.randint(0, N-1))
    reward = env.step(actions)
    next_X = env.G.X()
    next_agent_states = []
    for agent in range(M):
        next_state, _ = env.agents[agent].state()
        next_state = torch.nn.functional.normalize(next_state, dim=0)
        next_agent_states.append(next_state.unsqueeze(1))
    next_agent_states = torch.cat(next_agent_states, dim=1)

    transitions = set()
    for agent in range(M):
        transitions.add(ExtendGraphTransition(state_agent=agent_states[:, agent, :],
                                              next_state_agent=next_agent_states[:, agent, :],
                                              feature_graph=X,
                                              next_feature_graph=next_X,
                                              reward=reward[agent],
                                              action=actions[agent],
                                              agent_idx=agent,
                                              remain_budget=remain_budgets[agent]))
    memory.push_all(list(transitions))
    return torch.tensor(reward)


def optimize(samples):
    batch = ExtendGraphTransition(*zip(*samples))
    batch_remain_budgets = batch.remain_budget
    batch_actions = batch.action
    batch_reward = torch.tensor(batch.reward)
    # (B, 20, 3)
    batch_feature_graph = torch.stack(batch.feature_graph)
    batch_next_feature_graph = torch.stack(batch.next_feature_graph)
    # (B, N, 1)
    batch_state_agent = torch.stack(batch.state_agent)
    batch_next_state_agent = torch.stack(batch.next_state_agent)
    batch_agent_idx = torch.tensor(batch.agent_idx)

    cuda_batch_state_agent = batch_state_agent.float().cuda()
    g_input = batch_feature_graph.reshape(-1, M, 1).float().cuda()
    # (B, N, M*32)
    X_theta = g_model(g_input).reshape(B, N, -1)
    Z = torch.bmm(hatA.unsqueeze(0).expand(B, -1, -1), X_theta).reshape(B, N, M, -1)
    batch_graph_embedding = Z[[i for i in range(B)], :, batch_agent_idx, :]
    # (B, N, L+1)
    q_input = torch.cat([batch_graph_embedding, cuda_batch_state_agent], dim=-1)
    # (B, N, 1)
    Q = q_model(q_input.reshape(-1, q_input.shape[-1])).reshape(B, N, -1)
    # (B, 1)
    Q = Q[[i for i in range(B)], batch_actions, :]

    cuda_batch_next_state_agent = batch_next_state_agent.float().cuda()
    cuda_next_g_input = batch_next_feature_graph.float().cuda().reshape(-1, M, 1)
    # (B, N, M*32)
    next_X_theta = g_model(cuda_next_g_input).reshape(B, N, -1)
    # (B, M, M, L)
    next_Z = torch.bmm(hatA.unsqueeze(0).expand(B, -1, -1), next_X_theta).reshape(B, N, M, -1)
    # (B, N, L)
    next_batch_graph_embedding = next_Z[[i for i in range(B)], :, batch_agent_idx, :]
    # (B, N, L+1)
    next_q_input = torch.cat([next_batch_graph_embedding, cuda_batch_next_state_agent], dim=-1)
    # (B, N, 1)
    next_Q = q_model(next_q_input.reshape(-1, next_q_input.shape[-1])).reshape(B, N, -1)
    # (B, 1)
    next_Q = next_Q.max(dim=1)[0]

    batch_remain_budgets = torch.tensor(batch_remain_budgets).unsqueeze(-1).float().cuda()
    batch_remain_budgets = batch_remain_budgets / BUDGET
    batch_reward = batch_reward.float().cuda()
    Y = batch_reward.unsqueeze(-1)
    Y[Y != -1] += (next_Q[Y != -1] * GAMMA * batch_remain_budgets[Y != -1])
    Y[Y == -1] = 0.0

    loss = torch.nn.functional.mse_loss(Q, Y)
    loss.backward()
    optimizer.step()
    print(' Loss : {0}'.format(loss.item()))


def train(epoch):
    env.reset()
    t = time.time()
    q_model.train()
    g_model.train()
    optimizer.zero_grad()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * epoch / EPS_DECAY)
    rewards = 0.0
    for step in range(STEPS):
        reward = forward_step(step, eps_threshold)
        reward[reward == -1] = 0.0
        rewards += reward.sum(0).item()
    print('\nEpoch: {}'.format(epoch),
          ' Rewards: {}'.format(rewards))
    if epoch % 3 == 0 and epoch > 0:
        samples = memory.sample(args.batch_size)
        optimize(samples)


for epoch in range(args.epochs):
    train(epoch)





