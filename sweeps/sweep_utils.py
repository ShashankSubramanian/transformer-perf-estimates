import os
import sys
import numpy as np
import json


def sweep_bw_capacity(sys_str = 'A100'):
    ''' create an array of capacities and bandwidths to sweep over '''
    with open('../systems/config-{}.json'.format(sys_str), 'r') as file:
        system = json.load(file)

    nconfs_capacity = 32  # how many capacities
    nconfs_bandwidth = 28 # how many bandwidths
    hbm_capacity_list = np.linspace(system['hbm_capacity'],1000.0,nconfs_capacity)
    hbm_bandwidth_list = np.linspace(500.0,16000.0,nconfs_bandwidth)

    numx = len(hbm_capacity_list)
    numy = len(hbm_bandwidth_list)

    params = []
    for i in range(numx):
        for j in range(numy):
            cap = hbm_capacity_list[i]
            bw = hbm_bandwidth_list[j]
            params.append((bw,cap))

    return hbm_capacity_list, hbm_bandwidth_list, params

def sweep_flops_cap_bw():
    ''' create an array of flops and capacity to sweep over.
        Also sweep bws and vector flops. Typically they also
        scale similarly as the capacity and matrix flops.
        the values are swept so that 4 gpus are sampled,
        including a future B200-next'''

    config1 = 'A100'
    config2 = 'H200'
    config3 = 'B200'
    config4 = 'B200-next' # possible future B200

    with open('../systems/config-' + config1 + '.json', 'r') as file:
        system1  = json.load(file)

    with open('../systems/config-' + config2 + '.json', 'r') as file:
        system2 = json.load(file)

    with open('../systems/config-' + config3 + '.json', 'r') as file:
        system3 = json.load(file)

    with open('../systems/config-' + config4 + '.json', 'r') as file:
        system4 = json.load(file)

    nconfs = 8

    hbm_capacity_list12 = np.linspace(system1['hbm_capacity'],system2['hbm_capacity'],nconfs//2)
    hbm_bandwidth_list12 = np.linspace(system1['hbm_bandwidth'],system2['hbm_bandwidth'],nconfs//2)
    matrix_flops_list12 = np.linspace(system1['matrix_flops_fp16'],system2['matrix_flops_fp16'],nconfs)
    vector_flops_list12 = np.linspace(system1['vector_flops_fp16'],system2['vector_flops_fp16'],nconfs)

    hbm_capacity_list23 = np.linspace(system2['hbm_capacity'],system3['hbm_capacity'],nconfs//2)
    hbm_bandwidth_list23 = np.linspace(system2['hbm_bandwidth'],system3['hbm_bandwidth'],nconfs//2)
    matrix_flops_list23 = np.linspace(system2['matrix_flops_fp16'],system3['matrix_flops_fp16'],nconfs)
    vector_flops_list23 = np.linspace(system2['vector_flops_fp16'],system3['vector_flops_fp16'],nconfs)

    hbm_capacity_list34 = np.linspace(system3['hbm_capacity'],system4['hbm_capacity'],nconfs//2)
    hbm_bandwidth_list34 = np.linspace(system3['hbm_bandwidth'],system4['hbm_bandwidth'],nconfs//2)
    matrix_flops_list34 = np.linspace(system3['matrix_flops_fp16'],system4['matrix_flops_fp16'],nconfs)
    vector_flops_list34 = np.linspace(system3['vector_flops_fp16'],system4['vector_flops_fp16'],nconfs)

    hbm_capacity_list = np.concatenate([hbm_capacity_list12[:-1],hbm_capacity_list23[:-1],hbm_capacity_list34])
    hbm_bandwidth_list = np.concatenate([hbm_bandwidth_list12[:-1],hbm_bandwidth_list23[:-1],hbm_bandwidth_list34])
    matrix_flops_list = np.concatenate([matrix_flops_list12[:-1],matrix_flops_list23[:-1],matrix_flops_list34])
    vector_flops_list = np.concatenate([vector_flops_list12[:-1],vector_flops_list23[:-1],vector_flops_list34])

    numx=len(hbm_capacity_list)
    numy=len(matrix_flops_list)

    params = []
    for i in range(numx):
        for j in range(numy):
            bw = hbm_bandwidth_list[i]
            cap = hbm_capacity_list[i]
            matrix_flops = matrix_flops_list[j]
            vector_flops = vector_flops_list[j]
            params.append((bw,cap,matrix_flops,vector_flops))

    return hbm_capacity_list, hbm_bandwidth_list, matrix_flops_list, vector_flops_list, params
