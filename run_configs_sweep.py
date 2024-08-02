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

    # set data hyperparms
    if model_str == 'gpt3_1T':
        total_tokens = 1 * 10**12
    else:
        total_tokens = 300 * 10**9
    num_samples = total_tokens / l
    if 'vit_era5' in model_str:
        num_ep = 80
        num_samples = 350000 * num_ep
    print('training on {} samples'.format(num_samples))

    global_batch_size = args.global_batch_size
    config1 = 'A100'
    config2 = 'B200'
    with open('systems/config-' + config1 + '.json', 'r') as file:
        system1  = json.load(file)
    with open('systems/config-' + config2 + '.json', 'r') as file:
        system2 = json.load(file)

    nconfs = 16
    hbm_capacity_list = np.linspace(system1['hbm_capacity'],system2['hbm_capacity'],nconfs//2)
    hbm_bandwidth_list = np.linspace(system1['hbm_bandwidth'],system2['hbm_bandwidth'],nconfs//2)
    matrix_flops_list = np.linspace(system1['matrix_flops_fp16'],system2['matrix_flops_fp16'],nconfs)
    vector_flops_list = np.linspace(system1['vector_flops_fp16'],system2['vector_flops_fp16'],nconfs)


    params = []
    for i in range(nconfs//2):
        for j in range(nconfs):
            bw = hbm_bandwidth_list[i]
            cap = hbm_capacity_list[i]
            matrix_flops = matrix_flops_list[j]
            vector_flops = vector_flops_list[j]
            params.append((bw,cap,matrix_flops,vector_flops))

    n_gpus = 2**np.array([i for i in range(2,15)])
    n_params = len(params)

    t0 = time.time()

    rank = MPI.COMM_WORLD.rank
    n_proc = MPI.COMM_WORLD.size
    print(rank, n_proc)
    assert n_proc == n_params
    params = [params[rank]]
    print('rank {} is doing {}'.format(rank, params))

    if args.parallel_strat == '1d':
        execute_fn = execute_1d
    elif args.parallel_strat == '2d':
        execute_fn = execute_2d
    elif args.parallel_strat == '2d-seqp':
        execute_fn = execute_seqp
    else:
        assert False, 'parallel strat not valid'

    for sidx, p in enumerate(params):
        with open('systems/config-B200.json', 'r') as file:
            system = json.load(file)
        system['hbm_bandwidth'] = p[0]
        system['hbm_capacity'] = p[1]
        system['matrix_flops_fp16'] = p[2]
        system['vector_flops_fp16'] = p[3]
        sys_str = 'B200-{}'.format(str(rank))
        nvs = 8

        plots = []
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
        np.save('outputs/sweep/sw_exec_{}_{}_{}.npy'.format(args.parallel_strat, model_str, sys_str), np.array(plots, dtype=object))

    print('time for script = {}'.format(time.time() - t0))
