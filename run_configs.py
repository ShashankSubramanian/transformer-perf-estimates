import os
import sys
import time
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
import pprint

pd.options.display.max_columns = None
pd.options.display.max_rows = None
from modules import *
from execution import *
import json
import argparse
import models

from mpi4py import MPI

if __name__ == '__main__':
    # parsers
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='gpt3_1T', type=str, help="model type")
    parser.add_argument("--global_batch_size", default=4096, type=int, help="gobal batch size")
    parser.add_argument("--parallel_strat", default='1d', type=str, help="tensor parallelization strategy")
    args = parser.parse_args()

    model_str = args.model
    model = models.models[args.model]
    
    # set model hyperparams
    l = model['l']
    e = model['e']
    f = 4 * e
    model['f'] = f
    h = model['h']
    depth = model['depth']
    print('model is {}'.format(model))

    # set data hyperparms
    if model_str == 'gpt3_1T':
        total_tokens = 1 * 10**12
    else:
        total_tokens = 300 * 10**9
    num_samples = total_tokens / l
    if model_str == 'vit_era5':
        num_ep = 80
        num_samples = 350000 * num_ep
    print('training on {} samples'.format(num_samples))

    global_batch_size = args.global_batch_size
    systems = ['A100-NVL4', 'A100-NVL8', 'A100-NVL64', 'H200-NV4', 'H200-NVL8', 'H200-NVL64', 'B200-NVL4', 'B200-NVL8', 'B200-NVL64']
    config_names = ['A100', 'A100', 'A100', 'H200', 'H200', 'H200', 'B200', 'B200', 'B200']
    nvlink_sizes = [4, 8, 64, 4, 8, 64, 4, 8, 64]
    n_gpus = 2**np.array([i for i in range(2,15)])
    n_sys = len(systems)

    t0 = time.time()

    rank = MPI.COMM_WORLD.rank
    n_proc = MPI.COMM_WORLD.size
    print(rank, n_proc)
    assert n_proc == n_sys
    systems = [systems[rank]]
    config_names = [config_names[rank]]
    nvlink_sizes = [nvlink_sizes[rank]]
    print('rank {} is doing {}'.format(rank, systems))

    if args.parallel_strat == '1d':
        for sidx, sys_str in enumerate(systems):
            with open('systems/config-' + config_names[sidx] + '.json', 'r') as file:
                system = json.load(file)
            nvs_list = [nvlink_sizes[sidx]]
            print(sys_str, nvs_list)
            plots = []
            for nvs in nvs_list:
                t = []
                conf = []
                start = None
                system['nvlink_size'] = nvs
                configs = execute_1d(model, n_gpus, global_batch_size=global_batch_size, system=system, verbose=False, nlargest=10)
                for s,config in enumerate(configs):
                    if len(config) > 0: # check feasibility
                        if not start and start != 0:
                            start = s
                        conf.append(config)
                        t.append([c[0] for c in config])
                t_max = [tm[0] for tm in t]
                t_min = [tm[-1] for tm in t]
                n_gpus = n_gpus[start:]
                configs = configs[start:]
                plots.append((nvs, t_max, t_min, n_gpus, configs))
            np.save('outputs/exec_1d_{}_{}.npy'.format(model_str, sys_str), np.array(plots, dtype=object))

    print('time for script = {}'.format(time.time() - t0))
