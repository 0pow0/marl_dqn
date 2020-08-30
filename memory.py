from collections import namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


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

