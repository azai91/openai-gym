import gym
import numpy as np

D = 1
H = 8
C = 4

model = {}
model['W1'] = np.random.randn(H,D)
model['W2'] = np.random.randn(C, H)

LEFT = 0
RIGHT = 1
WRITE = 1
NOT_WRITE = 0
NUMBER_OF_LETTERS = 4

# output is a vector
"""
[Direction, Write, A, B, C, D, E]
"""

render = True

env = gym.make('Copy-v0')
observation = env.reset()

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]


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


def get_actions(aprob):
    direction_prob = sigmoid(aprob[0])
    write_prob = sigmoid(aprob[1])
    lprobs = np.softmax(aprob[2:])

    direction = RIGHT if np.random.uniform() < direction_prob else LEFT
    to_write = WRITE if np.random.uniform() < write_prob else NOT_WRITE
    letter_to_write = np.random.choice(NUMBER_OF_LETTERS, 1, lprobs)[0]

    ddirection = 1 - direction_prob if direction == RIGHT else -direction_prob
    dto_write = 1 - write_prob if to_write == WRITE else -write_prob
    mask = np.zeros(NUMBER_OF_LETTERS)
    mask[letter_to_write] = 1
    dletters = mask - lprobs
    action = np.concatenate((np.array([direction, to_write]), mask))
    dlog = np.concatenate((np.array([ddirection, dto_write ]), dletters))

    return action, dlog

xs, hs, dlogps, drs = [], [], [], []
reward_sum = 0
episode_number = 0


while True:
    if render: env.render()

    x = observation
    aprob, h = forward_pass(x)
    action, dlog = get_actions(aprob)


    # output is total number of actions

    # might want to scale actions between 0 and 1

    xs.append(x)
    hs.append(h)
    dlogps.append(dlog)

    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward)

    if done:
        episode_number = 0
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)

        xs, hs, dlogps, drs = [], [], [], []

        # discounted_epr = discount_rewards(epr)
        epdlogp *= epr
        grad = backward_pass(eph, epdlogp)






