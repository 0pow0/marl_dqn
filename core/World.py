from core.Agent import Agent
from core.City import City
import torch


class World(object):
    """
    conn:             shape: (1, 20, 20)
    tasks:            shape: (1, 3, 20)
    cities:           shape: (1, 20, 2)
    rewards:          shape: (1, 3, 20)
    destinations:     shape: (1, 1, 20)
    """
    def __init__(self, n_agents, n_cities, steps, conn, tasks, cities, rewards, dest, budget=80):
        self.n_agents = n_agents
        self.n_cities = n_cities

        self.agents = []
        self.cities = []

        for i in range(self.n_agents):
            self.agents.append(Agent(i, tasks[0][i], rewards[0][i], n_cities,
                                     n_agents, torch.full((2, n_cities), 0),
                                     budget=budget))

        for i in range(self.n_cities):
            self.cities.append(City(i, cities[0][i], conn[0][i], dest[0][0][i]))

