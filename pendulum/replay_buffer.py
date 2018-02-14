from collections import deque
import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, transition):
        if len(self.buffer) > self.buffer_size:
            self.buffer.popleft()
        self.buffer.append(transition)

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, len(self))
        return zip(**batch)

    def clear(self):
        self.buffer.clear()


