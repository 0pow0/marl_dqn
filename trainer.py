import random
import math
from models import Encoder, DQN
from env import Env
import torch
import torch.nn as nn
import torch.optim
import numpy as np
from memory import Transition


class Trainer(object):
    def __init__(self, args, n_agents, n_cities, device, data_loader):
        self.n_agents = n_agents
        self.n_cities = n_cities
        self.envs = []
        self.device = device
        self.steps_done = 0
        self.len_state = 1 + 2 * n_cities + 2 * args.steps * n_cities

        self.args = args
        # self.Encoder = Encoder(K=args.steps, M=self.n_cities, L=args.len_encoder).to(self.device)
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
        self.mse = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.DQN.parameters(), lr=args.lr)

        self.adam = torch.optim.Adam(self.DQN.parameters(), lr=args.lr, weight_decay=0.0001)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.adam, step_size=1000, gamma=0.1)

    def calc_loss(self, samples):

        batch = Transition(*zip(*samples))
        # State(Batch, 1, state_len)
        state_batch = torch.cat(batch.state).reshape(self.args.batch_size, 1, -1)
        next_state_batch = torch.cat(batch.next_state).reshape(self.args.batch_size, 1, -1)
        # action(Batch, 2)
        action_batch = torch.cat(batch.action)
        action_batch = torch.cat([e[1].reshape(1) if e[0] == 0
                                  else (e[1] + self.n_cities).reshape(1) for e in action_batch])
        action_batch = action_batch.reshape(-1, 1)
        # reward(Batch)
        reward_batch = torch.cat(batch.reward)

        # next_mask(Batch, 2*n_cities)
        # next_mask_batch = torch.cat(batch.next_mask).reshape(self.args.batch_size, -1)

        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        # Q(batch, 2*n_cities)
        Q = self.DQN(state_batch).gather(1, action_batch)

        # Q_next = self.DQN(next_state_batch).detach().cpu()
        Q_next = self.DQN(next_state_batch).detach().cpu()
        Q_next = Q_next.max(dim=1)[0].to(self.device)

        Q_expected = [(Q_next[i] * self.args.gamma + reward_batch[i][1]).reshape(1)
                      if reward_batch[i][0] == 1 else torch.tensor(0.0, device=self.device).reshape(1) for i in
                      range(len(samples))]
        Q_expected = torch.cat(Q_expected)
        Q_expected = Q_expected.reshape(-1, 1)

        # loss = self.criterion(Q, Q_expected)
        loss = self.mse(Q, Q_expected)

        return loss

    def reset(self):
        self.steps_done = 0
        for env in self.envs:
            env.reset()

    def gen_env(self):
        for data in self.data_loader:
            self.envs.append(Env(n_agents=self.n_agents, n_cities=self.n_cities, steps=self.args.steps,
                                 conn=data["conn"], tasks=data["tasks"], cities=data["cities"],
                                 rewards=data["rewards"], destinations=data["destinations"],
                                 budget=self.args.budget))

    def act(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        for i_env in range(self.n_envs):
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                            math.exp(-1. * self.steps_done / self.EPS_DECAY)
            states_ = []
            actions_ = []
            rewards_ = []
            next_states_ = []
            for i in range(self.n_agents):
                state = self.envs[i_env].input().reshape(self.n_agents, -1)
                tmp_ = torch.cat([state[i].reshape(-1), state[0:i].reshape(-1), state[i + 1:].reshape(-1)])
                state = tmp_.reshape(1, -1).to(self.device)
                p = random.random()
                cur_agent = self.envs[i_env].world.agents[i]
                # reachable_cities = {}
                # for city in self.envs[i_env].world.cities:
                #     if cur_agent.budget >= cur_agent.task[0][city.ID] != -1:
                #         reachable_cities[city.ID] = 1
                # if len(reachable_cities) == 0:
                #     actions_.append(torch.tensor([-1, -1]))
                #     continue
                if p > eps_threshold:
                    with torch.no_grad():
                        Q = self.DQN(state.reshape(1, 1, -1)).reshape(1, -1)
                        # tmp = [0] * (self.n_cities * 2)
                        # for city_idx in range(self.n_cities):
                        #     if city_idx in reachable_cities.keys():
                        #         tmp[city_idx] = Q[0][city_idx]
                        #         tmp[city_idx+20] = Q[0][city_idx+20]
                        #     else:
                        #         tmp[city_idx] = float('-inf')
                        #         tmp[city_idx+20] = float('-inf')
                        # Q = torch.tensor(tmp, device=self.device, dtype=torch.float).reshape(1, -1)
                        q = Q.reshape(2, -1).max(1)
                    if q[0][0] > q[0][1]:
                        action = torch.tensor([0], device=self.device, requires_grad=False)
                        action = torch.cat((action, q[1][0].reshape(1, ).long()))
                    else:
                        action = torch.tensor([1], device=self.device, requires_grad=False)
                        action = torch.cat((action, q[1][1].reshape(1, ).long()))
                    actions_.append(action.detach().cpu())
                else:
                    action = [random.choice([0, 1]), random.randint(0, self.n_cities - 1)]
                    # action = [random.choice([0, 1]), random.choice(list(reachable_cities.keys()))]
                    actions_.append(torch.tensor(action, requires_grad=False))

                reward = self.envs[i_env].step_one_agent(action, i)
                if reward[0] == -1:
                    reward = (-1, torch.tensor(0.0).reshape(1))

                next_state = self.envs[i_env].input().reshape(1, -1)

                states_.append(state.detach().cpu())
                rewards_.append(torch.tensor(reward, dtype=torch.float).reshape(1, 2))
                next_states_.append(next_state.detach().cpu())

            states.append(states_)
            actions.append(actions_)
            rewards.append(rewards_)
            next_states.append(next_states_)
        return states, actions, rewards, next_states

    def input(self):
        ipts = []
        for env in self.envs:
            ipts.append(env.input().reshape(self.n_agents, -1))
        ipts = torch.cat(ipts, dim=0).reshape((self.n_envs, self.n_agents, -1))
        return ipts

    def step(self, need_eps=True):
        # epsilon-greedy
        # actions: list, (n_env, n_agents); Q: tensor(n_envs, n_agent, 2*n_cities)
        states, actions, rewards, next_states = self.act()
        # initial Transition tuple
        res = []
        for i in range(self.n_envs):
            for j in range(self.n_agents):
                res.append(
                    Transition(state=states[i][j], action=actions[i][j].reshape(1, 2), next_state=next_states[i][j],
                               reward=rewards[i][j]))
        self.steps_done += 1
        return res
