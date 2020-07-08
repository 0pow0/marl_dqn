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

        self.device = device

        self.args = args
        self.Encoder = Encoder(K=args.steps, M=self.n_cities, L=args.len_encoder).to(self.device)
        self.DQN = DQN(N=self.n_agents, K=args.steps, L=args.len_encoder, M=n_cities).to(self.device)

        self.data_loader = data_loader
        self.iter_data = iter(data_loader)
        self.n_envs = len(data_loader)
        self.idx_env = -1
        self.env = None

        self.EPS_START = self.args.eps_start
        self.EPS_END = self.args.eps_end
        self.EPS_DECAY = self.args.eps_decay

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.DQN.parameters(), lr=args.lr)

    def calc_loss(self, samples):

        temp_Q = []
        temp_Q_next = []

        for i in range(len(samples)):
            action = samples[i].action
            reward = samples[i].reward.to(self.device)
            action_idx = action[0] * self.n_cities + action[1]

            temp_Q.append(samples[i].Q[action_idx].reshape(1))
            temp_Q_next.append((samples[i].Q_next.max() * self.args.gamma + reward).reshape(1).cuda(self.device))

        Q = torch.cat(temp_Q).float().cuda(self.device)
        Q_next = torch.cat(temp_Q_next).float().cuda(self.device)

        loss = self.criterion(Q, Q_next)

        return loss

    def gen_env(self):
        data = next(self.iter_data)
        self.idx_env += 1
        self.env = Env(n_agents=self.n_agents, n_cities=self.n_cities,
                       steps=self.args.steps, conn=data["conn"], tasks=data["tasks"], cities=data["cities"],
                       rewards=data["rewards"], destinations=data["destinations"],
                       budget=self.args.budget)

    def select_action(self, state, need_eps=True):
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.env.steps_done / self.EPS_DECAY)
        actions = []
        qs = []
        for i in range(self.n_agents):
            p = random.random()
            with torch.enable_grad():
                Q = self.DQN(state[i].reshape(1, 1, -1))
                if self.env.world.agents[i].at_city is not None:
                    mask = [1 if e != -1 else 0 for e in self.env.world.agents[i].task[0]] * 2
                    mask = torch.tensor(mask, device=self.device, requires_grad=False).reshape(1,-1)
                    Q = mask * Q
                q = Q.reshape(2, -1).max(1)
            qs.append(Q)
            if p > eps_threshold or (need_eps is False):
                if q[0][0] > q[0][1]:
                    action = torch.tensor([0], device=self.device, requires_grad=False)
                    action = torch.cat((action, q[1][0].reshape(1, ).long()))
                else:
                    action = torch.tensor([1], device=self.device, requires_grad=False)
                    action = torch.cat((action, q[1][1].reshape(1, ).long()))
                actions.append(action)
            else:
                action = [random.choice([0, 1]), random.randint(0, self.n_cities - 1)]
                actions.append(torch.tensor(action, device=self.device, requires_grad=False))
        qs = torch.cat(qs)
        return actions, qs

    def step(self, need_eps=True):
        self.env.steps_done += 1
        x = self.env.input().reshape(self.n_agents, -1).cuda(self.device)
        phi = []
        for i in range(self.n_agents):
            with torch.no_grad():
                phi.append(self.Encoder(x[i]))
        # after encoding
        n = torch.cat(phi, dim=0).reshape(self.n_agents, -1)

        # state
        s = []
        for i in range(self.n_agents):
            ni = torch.cat((n[0:i], n[i + 1:])).reshape(-1)
            s.append(torch.cat((x[i], ni)))
        s = torch.cat(s).reshape(self.n_agents, -1)
        # epsilon-greedy
        actions, Q = self.select_action(s, need_eps=need_eps)
        # collect rewards
        rewards = self.env.step(actions)
        if rewards == -1:
            return "done"
        # state_{t+1}
        x_tp1 = self.env.input().reshape(self.n_agents, -1).cuda(self.device)
        phi_tp1 = []
        for i in range(self.n_agents):
            with torch.no_grad():
                phi_tp1.append(self.Encoder(x_tp1[i]))
        n_tp1 = torch.cat(phi_tp1, dim=0).reshape(self.n_agents, -1)
        s_tp1 = []
        for i in range(self.n_agents):
            ni = torch.cat((n_tp1[0:i], n_tp1[i + 1:])).reshape(-1)
            s_tp1.append(torch.cat((x_tp1[i], ni)))
        s_tp1 = torch.cat(s_tp1).reshape(self.n_agents, -1)
        _, Q_next = self.select_action(s_tp1, need_eps=need_eps)
        # initial Transition tuple
        res = []
        for i in range(self.n_agents):
            res.append(Transition(state=s[i], action=actions[i], next_state=s_tp1[i],
                                  reward=rewards[i], Q=Q[i], Q_next=Q_next[i],
                                  ID=self.env.world.agents[i].ID,
                                  from_=self.env.world.agents[i].at_city.ID
                                  if self.env.world.agents[i].at_city is not None else -1))
        return res
