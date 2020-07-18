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
        self.remain_budget = self.get_budget()
        self.terminal = self.steps
        self.last_pick = [-1] * self.world.n_cities
        self.last_pick_distance = [-1] * self.world.n_cities

    def step(self, action):
        self.remain_budget = self.get_budget()
        if not self.check():
            self.close()
            # return -1 to stop iteration
            return -1

        rewards = [0] * self.world.n_agents

        for i in range(self.world.n_agents):
            task = self.world.agents[i].task
            reward = self.world.agents[i].step(action[i], self.world.cities[action[i][1]],
                                               action[i][1], self.steps_done)
            if action[i][0] == 1:
                if self.last_pick[action[i][1]] == -1:
                    self.last_pick[action[i][1]] = reward
                    self.last_pick_distance[action[i][1]] = task[0][action[i][1]]
                else:
                    if task[0][action[i][1]] <= self.last_pick_distance[action[i][1]]:
                        if reward - self.last_pick[action[i][1]] > 0:
                            reward = reward - self.last_pick[action[i][1]]
                        else:
                            reward = 0
                    else:
                        reward = 0
            rewards[i] = reward

        for i in range(len(rewards)):
            rewards[i] = torch.tensor([rewards[i]], dtype=torch.float)

        # self.last_pick = [-1] * self.world.n_cities
        return rewards

    def reset(self):
        self.steps_done = 0
        self.terminal = self.steps
        self.last_pick = [-1] * self.world.n_cities
        self.last_pick_distance = [-1] * self.world.n_cities
        self.world = World(self.n_agents, self.n_cities, self.steps, self.conn,
                           self.tasks, self.cities, self.rewards, self.destinations,
                           self.budget)
        self.remain_budget = self.get_budget()

    def render(self, mode='human'):
        pass

    def get_budget(self):
        budget = 0
        for uav in self.world.agents:
            budget += uav.budget
        return budget

    def check(self):
        return True if self.steps_done < self.terminal and self.remain_budget != 0 \
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

