from gcn.vGraph import vGraph


class Agent(object):
    def __init__(self, init_budget, pos=-1):
        self.budget = init_budget
        self.pos = pos
        self.reward = 0.0
        self.path = []

    def step(self, **action):
        self.budget -= action["cost"]
        self.pos = action["city"]
        self.reward += action["reward"]
        self.path.append(action["city"])


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
            self.agents.append(Agent(self.budget))
        self.G = vGraph(self.n_cities, data["conn"], data["rewards"])

    def step(self, actions):
        for i in range(self.n_agents):
            self.agents[i].step(actions[i])

    def step_one_agent(self, agent_idx, action):
        self.agents[agent_idx].step(action)

    def reset(self):
        self.rounds = 0
        self.G = vGraph(self.n_cities, self.data["conn"], self.data["rewards"])
        for i in range(self.n_agents):
            self.agents.append(Agent(self.budget))