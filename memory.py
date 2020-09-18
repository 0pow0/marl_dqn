from collections import namedtuple
import random
import torch

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


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
