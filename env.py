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

        self.world = World(self.n_agents, self.n_cities, self.steps, self.conn,
                           self.tasks, self.cities, self.rewards, self.destinations,
                           self.budget)
        self.budget = self.get_budget()
        self.terminal = self.steps
        self.last_pick = [-1] * self.world.n_cities

    def step(self, action):
        self.budget = self.get_budget()
        if not self.check():
            self.close()
            # return -1 to stop iteration
            return -1
        # measure distance if go to same city
        rewards = [0] * self.world.n_agents
        for i in range(self.world.n_agents):
            if action[i][0] == 1 and self.last_pick[action[i][1]] != -1:
                rewards[i] = -self.last_pick[action[i][1]]
        for i in range(self.world.n_agents):
            rewards[i] += self.world.agents[i].step(action[i], self.world.cities[action[i][1]],
                                                    action[i][1], self.steps_done)
        for i in range(len(rewards)):
            rewards[i] = torch.tensor([rewards[i]], dtype=torch.float)

        return rewards

    def reset(self):
        self.steps_done = 0
        self.terminal = self.steps
        self.last_pick = [-1] * self.world.n_cities
        self.world = World(self.n_agents, self.n_cities, self.steps, self.conn,
                           self.tasks, self.cities, self.rewards, self.destinations,
                           self.budget)
        self.budget = self.get_budget()

    def render(self, mode='human'):
        pass

    def get_budget(self):
        budget = 0
        for uav in self.world.agents:
            budget += uav.budget
        return budget

    def check(self):
        return True if self.steps_done < self.terminal and self.budget != 0 \
            else False

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
