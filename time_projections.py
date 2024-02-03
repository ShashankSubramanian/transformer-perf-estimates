import numpy as np

def comm_type_ops (n, comm_type):
    match comm_type:
        case 'allreduce':
            result = 2*(n-1)/n
        case 'reducescatter':
            result = (n-1)/n
        case 'allgather':
            result = (n-1)/n
        case 'broadcast':
            result = 1
        case 'reduce':
            result 1


def get_time_flops(flops, use_tensor=True, system={}):
    hardware_flops = system['matrix_flops_fp16'] if use_tensor else system['vector_flops_fp16']
    t_flops = flops / hardware_flops
    return t_flops * 10**3

def get_time_mem(mem, system={}):
    hbm_bandwidth = system['hbm_bandwidth']
    t_mem = mem / hbm_bandwidth
    return t_mem * 10**3

def get_time_comm(vol, n_gpus=4, comm_type='allreduce', topology='nvlink', empirical=False, system={}):
    # effective_vol / effective_bandwidht: need an analytical model
    if vol != vol:
        t_comm = 0
    else:
        if topology == 'nvlink': # use nvlink
            t_comm  = vol / system['nvlink_bandwidth'] * comm_type_ops(n,comm_type)
            t_latency = np.log2(n_gpus) * system['nvlink_latency']
        elif topology == 'ib':
            t_comm  = vol / system['ib_bandwidth'] * comm_type_ops(n,comm_type)
            t_latency = np.log2(n_gpus) * system['ib_latency']
        else:
            t_comm = 0
            t_latency =0
    return (t_comm + t_latency)* 10**3

def get_total_time(t_comp, t_comm, system={}, use_max=False):
    if use_max:
        return max(t_comp, t_comm) # overlap
    else:
        return t_comp + t_comm

def get_topology(n_gpus, system={}):
    if n_gpus != n_gpus:
        return None
    has_nvlink = system['nvlink_size'] > 0
    if has_nvlink:
        topology =  ('nvlink' if n_gpus <= system['nvlink_size'] else 'ib')
    else:
        topology = 'ib'
    return topology
