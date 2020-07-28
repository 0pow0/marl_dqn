import datetime
import os
from models import DQN, Encoder
from env import Env
import torch


class Evaluator(object):
    def __init__(self, args, n_agents, n_cities, device, data_loader):
        self.args = args
        self.n_agents = n_agents
        self.n_cities = n_cities
        self.n_envs = args.n_envs
        self.device = device
        self.data_loader = data_loader
        self.len_state = 1 + 2 * n_cities + 2 * args.steps * n_cities

        self.DQN = DQN(N=self.n_agents, K=args.steps, L=args.len_encoder, M=n_cities).to(self.device)
        self.DQN.load_state_dict(torch.load(self.args.load_from_main_checkpoint, map_location=torch.device(device)))
        self.DQN.eval()

        self.envs = []
        self.log_dir = args.log_dir_path + "/" + str(datetime.date.today())
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

    def gen_env(self):
        for data in self.data_loader:
            self.envs.append(Env(n_agents=self.n_agents, n_cities=self.n_cities, steps=self.args.steps,
                                 conn=data["conn"], tasks=data["tasks"], cities=data["cities"],
                                 rewards=data["rewards"], destinations=data["destinations"],
                                 budget=self.args.budget))

    def step(self, env_idx):
        x = self.envs[env_idx].input().reshape(self.n_agents, 1, -1)
        x = x.to(self.device)
        # with encoder
        # phi = self.Encoder(x)
        # without encoder
        phi = x
        phi_ = []
        for i in range(self.n_agents):
            if i == 0:
                phi_.append(phi[1:].reshape(-1, 1, (self.n_agents-1) * self.len_state))
            elif i == self.n_agents-1:
                phi_.append(phi[0:i].reshape(-1, 1, (self.n_agents-1) * self.len_state))
            else:
                phi_.append(torch.cat((phi[0:i], phi[i+1:]), dim=1)
                            .reshape(-1, 1, (self.n_agents-1) * self.len_state))
        phi = torch.cat(phi_)
        s = torch.cat((x, phi), dim=-1)
        actions = []
        for i in range(self.n_agents):
            with torch.no_grad():
                Q = self.DQN(s[i].reshape(1, 1, -1))
                if self.envs[env_idx].world.agents[i].at_city is not None:
                    mask = [1 if e != -1 else 0 for e in self.envs[env_idx].world.agents[i].task[0]] * 2
                    mask = torch.tensor(mask, device=self.device, requires_grad=False, dtype=torch.bool).reshape(1, -1)
                    Q = [Q[0][i] if mask[0][i] else float('-inf') for i in range(len(mask[0]))]
                    Q = torch.tensor(Q, device=self.device, dtype=torch.float).reshape(1, -1)
                q = Q.reshape(2, -1).max(1)
            if q[0][0] > q[0][1]:
                action = torch.tensor([0], device=self.device, requires_grad=False)
                action = torch.cat((action, q[1][0].reshape(1, ).long()))
            else:
                action = torch.tensor([1], device=self.device, requires_grad=False)
                action = torch.cat((action, q[1][1].reshape(1, ).long()))
            actions.append(action.detach().cpu())
        froms = [self.envs[env_idx].world.agents[j].at_city.ID
                 if self.envs[env_idx].world.agents[j].at_city is not None else -1
                 for j in range(self.n_agents)]
        rewards = self.envs[env_idx].step(actions)
        if rewards == -1:
            rewards = torch.cat([torch.tensor(0.0).reshape(1)]*self.n_agents)
        else:
            rewards = torch.cat(rewards)
        return rewards, actions, froms


