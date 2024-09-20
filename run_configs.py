import os
import sys
import time
import numpy as np
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
    parser.add_argument("--parallel_strat", default='1d', type=str, help="tensor parallelization strategy: 1d, 2d, 2d-seq")
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

    global_batch_size = args.global_batch_size

    # what systems?
    systems = ['A100-NVS4', 'A100-NVS8', 'A100-NVS64', 'H200-NVS4', 'H200-NVS8', 'H200-NVS64', 'B200-NVS4', 'B200-NVS8', 'B200-NVS64']
    # which config to use?
    config_names = ['A100', 'A100', 'A100', 'H200', 'H200', 'H200', 'B200', 'B200', 'B200']
    # set the nvlink sizes (overwrite the config)
    nvlink_sizes = [4, 8, 64, 4, 8, 64, 4, 8, 64]
    # how many gpus?
    n_gpus = 2**np.array([i for i in range(2,15)])
    n_sys = len(systems)
    print(n_sys)

    t0 = time.time()

    rank = MPI.COMM_WORLD.rank
    n_proc = MPI.COMM_WORLD.size
    print("running script on {} procs, printing from rank {}".format(rank, n_proc))
    if n_proc > 1:
        assert n_proc == n_sys, 'please use as many procs as systems'
        systems = [systems[rank]]
        config_names = [config_names[rank]]
        nvlink_sizes = [nvlink_sizes[rank]]
        print('rank {} is doing {}'.format(rank, systems))

    # what parallel strat?
    if args.parallel_strat == '1d':
        execute_fn = execute_1d
    elif args.parallel_strat == '2d': # summa
        execute_fn = execute_2d
    elif args.parallel_strat == '2d-seqp': # context parallel
        execute_fn = execute_seqp
    else:
        assert False, 'parallel strat not valid'

    os.makedirs("./outputs", exist_ok=True)

    for sidx, sys_str in enumerate(systems):
        with open('systems/config-' + config_names[sidx] + '.json', 'r') as file:
            system = json.load(file)
        nvs_list = [nvlink_sizes[sidx]] # overwrite nvs
        print(sys_str, nvs_list)
        plots = []
        for nvs in nvs_list:
            t = []
            conf = []
            start = None
            system['nvlink_size'] = nvs
            configs = execute_fn(model, n_gpus, global_batch_size=global_batch_size, system=system, verbose=False, nlargest=10)
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
        np.save('outputs/exec_{}_{}_{}.npy'.format(args.parallel_strat, model_str, sys_str), np.array(plots, dtype=object))

    print('time for script = {}'.format(time.time() - t0))
