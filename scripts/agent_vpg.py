
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from itertools import cycle, count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

import subprocess
import os.path
import tempfile
import random
import base64
import glob
import time
import json
import gym
import io
import gc
from gym import wrappers

LEAVE_PRINT_EVERY_N_SECS = 30
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
RESULTS_DIR = os.path.join('..', 'results')
SEEDS = (12, 34, 56, 78, 90)

plt.style.use('fivethirtyeight')
params = {
    'figure.figsize': (15, 8),
    'font.size': 24,
    'legend.fontsize': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
}
pylab.rcParams.update(params)
np.set_printoptions(suppress=True)
