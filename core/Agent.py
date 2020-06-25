from core.City import City


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
        # (2, K, M)
        self.history = history
        # (N-1, L)
        self.partner = None

        self.at_city = at_city

    def step(self, action, go_city: City, city_idx, step):
        if self.budget == 0:
            return 0
        if self.task[0][city_idx] == -1:
            return 0

        self.budget -= 1
        if action[0] == 0:
            reward = 0
            self.history[0][step][city_idx] = 1
        else:
            reward = self.reward[0][city_idx]
            self.history[1][step][city_idx] = 1
        self.task = go_city.conn.reshape(1, -1)
        return reward

    def input_(self):
        input_ = [self.budget.flatten(), self.task.flatten(),
                  self.reward.flatten(), self.history.flatten()]
        return input_




