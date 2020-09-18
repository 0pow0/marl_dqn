from core.City import City
import torch

class Agent(object):
    def __init__(self, ID, task, reward, n_cities, n_UAVs, history, at_city=None, budget=80):
        self.ID = ID
        # M
        self.n_cities = n_cities
        # N
        self.n_UAVs = n_UAVs
        # scalar
        self.budget = budget
        # (1, M)
        self.task = task.reshape(1, -1)
        # (1, M)
        self.reward = reward.reshape(1, -1)
        # (2, M)
        self.history = history
        # (N-1, L)
        self.partner = None
        self.at_city = at_city
        self.distance = 0.0

    """
    action guarantee city are reachable and within enough budget
    :return reward, could be 0, reward, -reward(penalty)
    """

    """
    Action are promised to be feasible, which means reachable from current city with enough budget
    """
    def step(self, action, go_city: City, city_idx, step):
        if self.budget < self.task[0][city_idx] or self.task[0][city_idx] == -1:
            return -1, 0

        self.budget -= self.task[0][city_idx]
        self.distance += self.task[0][city_idx]
        if action[0] == 0:
            reward = 0
            self.history[0][city_idx] = self.distance
        else:
            visited = False
            if self.history[1][city_idx] != 0:
                visited = True
            if not visited:
                reward = self.reward[0][city_idx].clone()
                self.history[1][city_idx] = self.distance
                self.reward[0][city_idx] = -(self.reward[0][city_idx].clone())
            else:
                reward = self.reward[0][city_idx].clone()
        self.task = go_city.conn.reshape(1, -1)
        self.at_city = go_city
        return 1, reward

    def input_(self):
        input_ = [self.budget.flatten(), torch.tensor(self.distance).reshape(1), self.task.flatten(),
                  self.reward.flatten(), self.history.flatten()]
        return input_
