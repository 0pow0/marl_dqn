from gcn.vGraph import vGraph
import torch
from copy import deepcopy


class Agent(object):
    def __init__(self, init_budget, cost, pos=-1):
        self.budget = init_budget
        self.pos = pos
        self.cost = cost
        self.reward = 0.0
        self.path = []

    def step(self, **action):
        if self.budget < self.cost[action["city"]] or self.cost[action["city"]] == -1:
            return -1
        self.budget -= self.cost[action["city"]]
        self.pos = action["city"]
        self.reward += action["reward"]
        self.path.append(action["city"])
        self.cost = action["cost"]
        return 1

    def state(self):
        state = torch.full((self.cost.shape[0], 1), self.budget)
        self.cost[self.cost == -1.0] = 100.0
        state = state.sub(self.cost.reshape(-1, 1))
        # state = torch.cat((state, self.cost.reshape(-1, 1).float()), dim=1)
        return state, self.budget


class Env(object):
    def __init__(self, n_cities, n_agents, steps, data, budget):
        self.n_cities = n_cities
        self.n_agents = n_agents
        self.budget = budget
        self.terminate = steps
        self.rounds = 0
        self.data = data
        self.agents = []
        for i in range(n_agents):
            self.agents.append(Agent(self.budget, deepcopy(data["tasks"][i])))
        self.G = vGraph(self.n_cities, deepcopy(data["conn"]), deepcopy(data["rewards"]))

    def step(self, actions):
        res = []
        for i in range(self.n_agents):
            res.append(self.step_one_agent(i, actions[i]))
        return res

    def step_one_agent(self, agent_idx, action):
        conn = self.G.vertexes[action].neighbors
        reward = self.G.vertexes[action].features[agent_idx]
        rtn = self.agents[agent_idx].step(cost=conn, reward=reward, city=action)
        if rtn == 1:
            reward = reward.clone()
            self.G.vertexes[action].features.fill_(0.0)
            return reward
        else:
            return -1

    def reset(self):
        self.rounds = 0
        self.G = vGraph(self.n_cities, deepcopy(self.data["conn"]), deepcopy(self.data["rewards"]))
        for i in range(self.n_agents):
            self.agents.append(Agent(self.budget, deepcopy(self.data["tasks"][i])))
