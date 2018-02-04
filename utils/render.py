# from argparse import ArgumentParser
#
#
# parser = ArgumentParser()
#
# parser

import matplotlib.pyplot as plt
import numpy as np

files = ['cartpole/pg_adv.npy','cartpole/pg_vanilla.npy']

for file in files:
    y = np.load(file)[::100]
    x = range(len(y))
    plt.plot(x,y, label=file)
    print(np.max(y))

plt.show()

