import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
import pprint
from execution import execute_1d, execute_2d, execute_seqp
import json
import argparse

gpt3_xl = {'l': 2048, 'e': 2048, 'h': 16, 'depth': 24}
gpt3 = {'l': 2048, 'e': 12288, 'h': 96, 'depth': 96}
gpt3_1T = {'l': 2048, 'e': 25600, 'h': 160, 'depth': 128}
gpt3_3T = {'l': 2048, 'e': 25600, 'h': 160, 'depth': 128*3}
vit_era5 = {'l': 64800*4, 'e': 4096, 'h': 32, 'depth': 32}

models = {'gpt3_xl': gpt3_xl,
          'gpt3': gpt3,
          'gpt3_1T': gpt3_1T,
          'gpt3_3T': gpt3_3T,
          'vit_era5': vit_era5}


def compute_stats(system, hbm_capacity, hbm_bandwidth, 
                  matrix_flops, vector_flops32, vector_flops16,i,j,
                  global_batch_size, config_type, n_gpus, nvs_list, model, model_str, exec_model, nlargest, verbose):
    fname='outputs_new/exec_'+str(i)+'_'+str(j)+'_'+exec_model+'_model_'+str(model_str)+'_config_'+config_type+'.npz'
    if not os.path.exists(fname):
        if exec_model == '1d':
            execute_function = execute_1d
        elif exec_model == '2d':
            execute_function = execute_2d
        elif exec_model == 'seqp':
            execute_function = execute_seqp
        stats={}

        system['hbm_capacity'] = hbm_capacity
        system['hbm_bandwidth'] = hbm_bandwidth
        system['matrix_flops_fp16'] = matrix_flops
        system['vector_flops_fp32'] = vector_flops32
        system['vector_flops_fp16'] = vector_flops16
    
        for nvs in nvs_list:
            t = []
            conf = []
            start = None
            system['nvlink_size'] = nvs
            configs = execute_function(model, n_gpus, global_batch_size=global_batch_size, system=system, verbose=verbose, nlargest=10)
            for s,config in enumerate(configs):
                if len(config) > 0: # check feasibility
                    if not start and start != 0:
                        start = s
                    conf.append(config)
                    t.append([c[0] for c in config])
        # print(t)

            t_max = [tm[0] for tm in t]
            t_min = [tm[-1] for tm in t]
            ngpus = n_gpus[start:]
            confs = configs[start:]

            stats[nvs]=[t_min,t_max,ngpus,confs]
      
        np.savez(fname, stats=stats, global_batch_size=global_batch_size)

def parse_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--global_batch_size", default=4096, type=int, help="gobal batch size")#4096
    parser.add_argument("--config_1", default='A100', type=str, help="the gpu type")
    parser.add_argument("--config_2", default='next', type=str, help="the gpu type")
    parser.add_argument("--ngpus_index", default=15, type=int, help="the max number of gpus for the system as a power of 2")#15
    parser.add_argument("--nv_index", default=5, type=float, help="the max number of gpus for NVLink domain as a power of 2")#7
    parser.add_argument("--LLM_model", default='vit_era5', type=str, help="LLM model")
    parser.add_argument("--parallel_option", default='1d', type=str, help="tensor parallelization strategy")
    parser.add_argument("--top_n", default=10, type=int, help="how many top results to store")
    parser.add_argument("--nconfs", default=25, type=int, help="how many configurations to compute")
    parser.add_argument("--verbose", default=False, type=bool, help="whether to print results for intermediate steps")
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args(); 
    
    global_batch_size = args.global_batch_size
    n_gpus = 2**np.array([i for i in range(2,args.ngpus_index)])
    nvs_list = list(2**np.array([i for i in range(1,args.nv_index)]))
    config_1=args.config_1
    config_2=args.config_2
    model_str = args.LLM_model
    model = models[model_str]

    l = model['l']
    e = model['e']
    f = 4 * e
    model['f'] = f

    #num_samples = total_tokens / l
    
    exec_model = args.parallel_option
    nlargest = args.top_n
    verbose = args.verbose
    nconfs = args.nconfs
    
    with open('systems/config-'+config_1+'.json', 'r') as file:
        system1 = json.load(file)
        
    with open('systems/config-'+config_2+'.json', 'r') as file:
        system2 = json.load(file)

    #print(system)
    
    hbm_capacity_list=np.linspace(system1['hbm_capacity'],system2['hbm_capacity'],nconfs)
    hbm_bandwidth_list=np.linspace(system1['hbm_bandwidth'],system2['hbm_bandwidth'],nconfs)
    matrix_flops_list=np.linspace(system1['matrix_flops_fp16'],system2['matrix_flops_fp16'],nconfs)
    vector_flops32_list=np.linspace(system1['vector_flops_fp32'],system2['vector_flops_fp32'],nconfs)
    vector_flops16_list=np.linspace(system1['vector_flops_fp16'],system2['vector_flops_fp16'],nconfs)

    for i in range(nconfs):
        for j in range(nconfs):
            compute_stats(system1, hbm_capacity_list[i], hbm_bandwidth_list[i], 
                  matrix_flops_list[j], vector_flops32_list[j], vector_flops16_list[j], i,j,
                  global_batch_size, config_1, n_gpus, nvs_list, model, model_str, exec_model, nlargest, verbose)
            print(i,j)

if __name__ == '__main__':
    main()
