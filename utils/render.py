from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os

files = ['cartpole/pg_adv.npy', 'cartpole/pg_vanilla.npy']


def transform(data, smooth):
    return np.mean(data.reshape(-1, smooth), axis=1)

def render_plot(data_dict):
    for file,data in data_dict.items():
        plt.plot(range(len(data)),data,label=file)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--smooth', type=int, default=100)

    args = parser.parse_args()

    data_dict = {}
    for file in args.files:
        data = np.load(os.path.join('data', file))
        smoothed = transform(data, args.smooth)
        data_dict[file] = smoothed

    render_plot(data_dict)