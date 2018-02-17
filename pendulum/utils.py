import numpy as np

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_dim, action_max, mu = 0, theta = 0.15, sigma = 0.2):
        self.action_dim = action_dim
        self.action_max = action_max
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X * self.action_max

def copy_params(target_network, source_network):
    for target_params, source_params in zip(target_network.parameters(), source_network.parameters()):
        target_params.data.copy_(source_params.data)