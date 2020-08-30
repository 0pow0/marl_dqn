import gym
import torch
from core.World import World


class Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        self.steps_done = 0
        # -1 represents has not been picked

        self.n_agents = kwargs["n_agents"]
        self.n_cities = kwargs["n_cities"]
        self.steps = kwargs["steps"]
        self.conn = kwargs["conn"]
        self.tasks = kwargs["tasks"]
        self.cities = kwargs["cities"]
        self.rewards = kwargs["rewards"]
        self.destinations = kwargs["destinations"]
        self.budget = kwargs["budget"]

        self.world = World(self.n_agents, self.n_cities, self.steps, self.conn.clone(),
                           self.tasks.clone(), self.cities.clone(), self.rewards.clone(), self.destinations.clone(),
                           self.budget)
        self.terminal = self.steps
        self.last_pick = [-1] * self.world.n_cities
        self.last_pick_distance = [-1] * self.world.n_cities

    def step_one_agent(self, action, agent_idx):
        if self.steps_done >= self.terminal:
            self.close()
            # return -1 to stop iteration
            return -1, 0

        reward = self.world.agents[agent_idx].step(action, self.world.cities[action[1]], action[1], self.steps_done)
        if action[0] == 1 and reward[0] is not -1:
            if self.last_pick[action[1]] == -1:
                self.last_pick[action[1]] = reward[1]
                self.last_pick_distance[action[1]] = self.world.agents[agent_idx].distance
            elif reward[1] >= 0:
                if self.world.agents[agent_idx].distance <= self.last_pick_distance[action[1]]:
                    reward = (1, reward[1] - self.last_pick[action[1]])
                else:
                    reward = (1, 0)
        if agent_idx == self.n_agents-1:
            self.steps_done += 1
        return reward

    def step(self, action):
        if self.steps_done >= self.terminal:
            self.close()
            # return -1 to stop iteration
            return -1

        rewards = [0] * self.world.n_agents

        for i in range(self.world.n_agents):
            # no reachable city, stay and no action
            # if action[i][0] == -1:
            #     rewards[i] = 0
            #     continue
            # reward could be:
            # 0, (action=visit but not pick)
            # >0 (no penalty from agent if agent never visited that before)
            # <0 (get penalty from agent if agent visit same city that this agent has visited before)
            reward = self.world.agents[i].step(action[i], self.world.cities[action[i][1]],
                                               action[i][1], self.steps_done)
            if action[i][0] == 1 and reward[0] is not -1:
                if self.last_pick[action[i][1]] == -1:
                    self.last_pick[action[i][1]] = reward[1]
                    self.last_pick_distance[action[i][1]] = self.world.agents[i].distance
                elif reward[1] >= 0:
                    if self.world.agents[i].distance <= self.last_pick_distance[action[i][1]]:
                        reward = (1, reward[1] - self.last_pick[action[i][1]])
                    else:
                        reward = (1, 0)
            rewards[i] = reward[1]

        for i in range(len(rewards)):
            rewards[i] = torch.tensor([rewards[i]], dtype=torch.float)
        self.steps_done += 1
        return rewards

    def step_eval(self, action):
        if self.steps_done >= self.terminal:
            self.close()
            # return -1 to stop iteration
            return -1

        rewards = [0] * self.world.n_agents

        for i in range(self.world.n_agents):
            if action[i][0] == -1:
                rewards[i] = 0
                continue
            # reward could be:
            # 0, (action=visit but not pick)
            # >0 (no penalty from agent if agent never visited that before)
            # <0 (get penalty from agent if agent visit same city that this agent has visited before)
            reward = self.world.agents[i].step(action[i], self.world.cities[action[i][1]],
                                               action[i][1], self.steps_done)
            reward = list(reward)
            if reward[1] < 0:
                reward[1] = 0
                continue
            if action[i][0] == 1 and reward[0] is not -1:
                if self.last_pick[action[i][1]] == -1:
                    self.last_pick[action[i][1]] = reward[1]
                    self.last_pick_distance[action[i][1]] = self.world.agents[i].distance
                elif reward[1] >= 0:
                    if self.world.agents[i].distance <= self.last_pick_distance[action[i][1]]:
                        reward = (1, reward[1] - self.last_pick[action[i][1]])
                    else:
                        reward = (1, 0)
            rewards[i] = reward[1]

        for i in range(len(rewards)):
            rewards[i] = torch.tensor([rewards[i]], dtype=torch.float)
        self.steps_done += 1
        return rewards

    def reset(self):
        self.steps_done = 0
        self.terminal = self.steps
        self.last_pick = [-1] * self.world.n_cities
        self.last_pick_distance = [-1] * self.world.n_cities
        self.world = World(self.n_agents, self.n_cities, self.steps, self.conn.clone(),
                           self.tasks.clone(), self.cities.clone(), self.rewards.clone(), self.destinations.clone(),
                           self.budget)

    def render(self, mode='human'):
        pass

    def input(self):
        x = []
        for uav in self.world.agents:
            if type(uav.budget) == int:
                uav.budget = torch.tensor(uav.budget).float()
            uav.budget = uav.budget.float()
            uav.reward = uav.reward.float()
            uav.task = uav.task.float()
            uav.history = uav.history.float()
            ipt = uav.input_()
            ipt = torch.cat(ipt)
            x.append(ipt)
        return torch.cat(x)

