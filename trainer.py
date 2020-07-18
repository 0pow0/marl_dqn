import random
import math
from models import Encoder, DQN
from env import Env
import torch
import torch.nn as nn
import torch.optim
from memory import Transition


class Trainer(object):
    def __init__(self, args, n_agents, n_cities, device, data_loader):
        self.n_agents = n_agents
        self.n_cities = n_cities
        self.envs = []
        self.device = device
        self.steps_done = 0

        self.args = args
        self.Encoder = Encoder(K=args.steps, M=self.n_cities, L=args.len_encoder).to(self.device)
        self.DQN = DQN(N=self.n_agents, K=args.steps, L=args.len_encoder, M=n_cities).to(self.device)
        self.target_DQN = DQN(N=self.n_agents, K=args.steps, L=args.len_encoder, M=n_cities).to(self.device)
        self.target_DQN.load_state_dict(self.DQN.state_dict())
        self.target_DQN.eval()

        self.data_loader = data_loader
        self.iter_data = iter(data_loader)
        self.n_envs = args.n_envs
        self.idx_env = -1
        self.env = None

        self.EPS_START = self.args.eps_start
        self.EPS_END = self.args.eps_end
        self.EPS_DECAY = self.args.eps_decay

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.RMSprop(self.DQN.parameters(), lr=args.lr)

    def calc_loss(self, samples):

        batch = Transition(*zip(*samples))
        # State(Batch, 1, state_len)
        state_batch = torch.cat(batch.state).reshape(self.args.batch_size, 1, -1)
        next_state_batch = torch.cat(batch.next_state).reshape(self.args.batch_size, 1, -1)
        # action(Batch, 2)
        action_batch = torch.cat(batch.action)
        action_batch = torch.cat([e[1].reshape(1) if e[0] == 0
                                  else (e[1]+self.n_cities).reshape(1) for e in action_batch])
        action_batch = action_batch.reshape(-1, 1)
        # reward(Batch)
        reward_batch = torch.cat(batch.reward)

        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        # Q(batch, 2*n_cities)
        Q = self.DQN(state_batch).gather(1, action_batch)
        Q_next = self.DQN(next_state_batch).max(dim=1)[0].detach()
        Q_expected = (Q_next * self.args.gamma) + reward_batch
        Q_expected = Q_expected.reshape(-1, 1)

        loss = self.criterion(Q, Q_expected)

        return loss

    def reset(self):
        for env in self.envs:
            env.reset()

    def gen_env(self):
        for data in self.data_loader:
            self.envs.append(Env(n_agents=self.n_agents, n_cities=self.n_cities, steps=self.args.steps,
                                 conn=data["conn"], tasks=data["tasks"], cities=data["cities"],
                                 rewards=data["rewards"], destinations=data["destinations"],
                                 budget=self.args.budget))

    def select_action(self, state, need_eps=True):
        actions = []
        for i_env in range(self.n_envs):
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                            math.exp(-1. * self.steps_done / self.EPS_DECAY)
            actions_ = []

            for i in range(self.n_agents):
                p = random.random()
                with torch.no_grad():
                    Q = self.DQN(state[i_env][i].reshape(1, 1, -1))
                    if self.envs[i_env].world.agents[i].at_city is not None:
                        mask = [1 if e != -1 else 0 for e in self.envs[i_env].world.agents[i].task[0]] * 2
                        mask = torch.tensor(mask, device=self.device, requires_grad=False, dtype=torch.bool).reshape(1, -1)
                        Q = [Q[0][i] if mask[0][i] else float('-inf') for i in range(len(mask[0]))]
                        Q = torch.tensor(Q, device=self.device, dtype=torch.float).reshape(1, -1)
                    q = Q.reshape(2, -1).max(1)

                if p > eps_threshold or (need_eps is False):
                    if q[0][0] > q[0][1]:
                        action = torch.tensor([0], device=self.device, requires_grad=False)
                        action = torch.cat((action, q[1][0].reshape(1, ).long()))
                    else:
                        action = torch.tensor([1], device=self.device, requires_grad=False)
                        action = torch.cat((action, q[1][1].reshape(1, ).long()))
                    actions_.append(action.detach().cpu())
                else:
                    action = [random.choice([0, 1]), random.randint(0, self.n_cities - 1)]
                    actions_.append(torch.tensor(action, requires_grad=False))

            actions.append(actions_)
        return actions

    def input(self):
        ipts = []
        for env in self.envs:
            ipts.append(env.input().reshape(self.n_agents, -1))
        ipts = torch.cat(ipts, dim=0).reshape((self.n_envs, self.n_agents, -1))
        return ipts

    def step(self, need_eps=True):
        self.steps_done += 1
        for env in self.envs:
            env.steps_done += 1
        # x (n_envs, n_agents, 1+2*n_cities+n_cities+2*steps*n_cities)
        x = self.input()
        # phi (n_envs, n_agents, (n_agents-1) * encode_len)
        x = x.to(device=self.device)
        phi = self.Encoder(x)
        phi_ = []
        for i in range(self.n_agents):
            if i == 0:
                phi_.append(phi[:, 1:].reshape(-1, 1, (self.n_agents-1) * self.args.len_encoder))
            elif i == self.n_agents-1:
                phi_.append(phi[:, 0:i].reshape(-1, 1, (self.n_agents-1) * self.args.len_encoder))
            else:
                phi_.append(torch.cat((phi[:, 0:i], phi[:, i+1:]), dim=1).reshape(-1, 1, (self.n_agents-1) * self.args.len_encoder))
        phi = torch.cat(phi_, dim=1)
        # CUDA s (n_envs, n_agents, 1+2*n_cities+n_cities+2*steps*n_cities + (n_agents-1) * encode_len)
        s = torch.cat((x, phi), dim=-1)
        # epsilon-greedy
        # actions: list, (n_env, n_agents); Q: tensor(n_envs, n_agent, 2*n_cities)
        actions = self.select_action(s, need_eps=need_eps)
        # collect rewards
        rewards = []
        for i in range(self.n_envs):
            reward = self.envs[i].step(actions[i])
            if reward == -1:
                rewards.append(torch.cat([torch.tensor(0.0).reshape(1)]*self.n_agents))
            else:
                rewards.append(torch.cat(reward))
        rewards = torch.cat(rewards).reshape(self.n_envs, self.n_agents)
        # state_{t+1}
        x_tp1 = self.input()
        x_tp1 = x_tp1.to(device=self.device)
        phi_tp1 = self.Encoder(x_tp1)
        phi_tp1_ = []
        for i in range(self.n_agents):
            with torch.no_grad():
                if i == 0:
                    phi_tp1_.append(phi_tp1[:, 1:].reshape(-1, 1, (self.n_agents - 1) * self.args.len_encoder))
                elif i == self.n_agents - 1:
                    phi_tp1_.append(phi_tp1[:, 0:i].reshape(-1, 1, (self.n_agents - 1) * self.args.len_encoder))
                else:
                    phi_tp1_.append(torch.cat((phi_tp1[:, 0:i], phi_tp1[:, i + 1:]), dim=1).reshape(-1, 1, (
                                self.n_agents - 1) * self.args.len_encoder))
        phi_tp1 = torch.cat(phi_tp1_, dim=1)
        s_tp1 = torch.cat((x_tp1, phi_tp1), dim=-1)
        s = s.detach().cpu()
        s_tp1 = s_tp1.detach().cpu()
        # initial Transition tuple
        res = []
        for i in range(self.n_envs):
            for j in range(self.n_agents):
                res.append(Transition(state=s[i][j], action=actions[i][j].reshape(1, 2), next_state=s_tp1[i][j],
                                      reward=rewards[i][j].reshape(1), ID=self.envs[i].world.agents[j].ID,
                                      from_=self.envs[i].world.agents[j].at_city.ID
                                      if self.envs[i].world.agents[j].at_city is not None else -1))
        return res
