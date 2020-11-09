from collections import namedtuple
import random
import torch

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

GraphTransition = namedtuple('GraphTransition', ('state_agent', 'feature_graph',
                                                 'next_state_agent', 'next_feature_graph',
                                                 'reward', 'action',
                                                 'agent_idx', 'remain_budget'))


class ExtendGraphTransition(GraphTransition):
    def __eq__(self, other):
        if not isinstance(other, ExtendGraphTransition):
            return False
        return (torch.all(torch.eq(self.state_agent, other.state_agent)) and
                torch.all(torch.eq(self.next_state_agent, other.next_state_agent)) and
                torch.all(torch.eq(self.feature_graph, other.feature_graph)) and
                torch.all(torch.eq(self.next_feature_graph, other.next_feature_graph)))

    def __repr__(self):
        return "Reward{0} Action{1} " \
               "state_agent{2} next_state_agent{3} " \
               "feature_graph{4} next_feature_graph{5}".format(self.reward, self.action,
                                                               self.state_agent, self.next_state_agent,
                                                               self.feature_graph, self.next_feature_graph)

    def __hash__(self):
        return hash(self.__repr__())


class ExtendTransition(Transition):
    def __eq__(self, other):
        if not isinstance(other, ExtendTransition):
            return False
        return torch.all(torch.eq(self.state, other.state)) and torch.all(torch.eq(self.next_state, other.next_state))

    def __repr__(self):
        return "Budget{0} Reward{1} Conn{2} History{3} Action{4}".format(self.state[0], self.state[1], self.state[2],
                                                                         self.state[3], self.action)

    def __hash__(self):
        return hash(self.__repr__())


class ReplayMemoryWithSet(object):
    def __init__(self, capacity):
        self.memory = set()
        self.capacity = capacity

    def __len__(self):
        return len(self.memory)

    def push_all(self, source: list):
        for e in source:
            while len(self) > self.capacity:
                self.memory.pop()
            self.memory.add(e)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def push_all(self, source: list):
        for e in source:
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = e
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def __len__(self):
        return len(self.memory)
