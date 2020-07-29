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
        self.distance = 0

    """
    action guarantee city are reachable and within enough budget
    :return reward, could be 0, reward, -reward(penalty)
    """
    def step(self, action, go_city: City, city_idx, step):

        self.budget -= self.task[0][city_idx]
        self.distance += self.task[0][city_idx]
        if action[0] == 0:
            reward = 0
            self.history[0][step][city_idx] = 1
        else:
            if self.history[1][step][city_idx] != 1:
                reward = self.reward[0][city_idx].clone()
                self.history[1][step][city_idx] = 1
                self.reward[0][city_idx] = -reward
            else:
                reward = self.reward[0][city_idx].clone()
        self.task = go_city.conn.reshape(1, -1)
        self.at_city = go_city
        return reward

    def input_(self):
        input_ = [self.budget.flatten(), self.task.flatten(),
                  self.reward.flatten(), self.history.flatten()]
        return input_




