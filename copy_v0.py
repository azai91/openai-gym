import gym
import numpy as np

D = 1
H = 8
C = 4

model = {}
model['W1'] = np.random.randn(H,D)
model['W2'] = np.random.randn(C, H)



def forward_pass(x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0
    y = np.dot(model['W2'], h)
    return y

def backward_pass(h, x, dy):
    dW2 = np.dot(dy, h).ravel()
    dh = np.outer(dy, model['W2'])
    dh[h <= 0] = 0
    dW1 = np.dot(dh, x)
    return {'W1': dW1, 'W2': dW2}