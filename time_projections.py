import numpy as np
import json

with open('./measurements/comm/nccl-benchmark-21209538.out', 'r') as data:
    bwdata = json.load(data)

def nccl_bw_sigmoid(op, message_size, gpus_per_node=4, num_nodes=2):
    '''
    Computes expected bandwidth based on sigmoid fits to empirical measurements on Perlmutter
    Inputs:
        op: one of 'allreduce', 'allgather', 'reducescatter'
        message_size: communication volume in bytes (int/float or np.array)
        gpus_per_node: number of GPUs in the node (or nvlink island)
        num_nodes: number of nodes (or nvlink islands)
    Returns:
        bw: bandwidth in GB/s
    '''
    gpusper_tag = 'gpus_per_node_%d'%gpus_per_node
    nn_tag = 'num_nodes_%d'%num_nodes
    height, slope, xcenter, ycenter = bwdata[op][gpusper_tag][nn_tag]
    x = np.log2(message_size*2**30)
    bw = height / (1 + np.exp(-slope * (x - xcenter))) + ycenter
    return bw

def get_time_flops(flops, use_tensor=True, system={}):
    hardware_flops = system['matrix_flops_fp16'] if use_tensor else system['vector_flops_fp16']
    t_flops = flops / hardware_flops
    return t_flops * 10**3

def get_time_mem(mem, system={}):
    hbm_bandwidth = system['hbm_bandwidth']
    t_mem = mem / hbm_bandwidth
    return t_mem * 10**3

def get_time_comm(vols, n_gpuss=4, comm_types='allreduce', topologys='nvlink', empirical=False, system={}):
    # TODO: incomplete
    if isinstance(vols, list):
        for vol,n_gpus,comm_type,topology in zip(vols,n_gpuss,comm_types,topologys):
            if empirical: assert system['nvlink_size'] <= 4, 'Empirical comms measurements not intended for nvlink_size > 4 currently'
            if topology == 'nvlink': # use nvlink
                t_comm  +=  vol / nccl_bw_sigmoid(comm_type, vol, system['nvlink_size'], num_nodes=1) if empirical else vol / system['nvlink_bandwidth']
            elif topology == 'ib':
                t_comm  += vol / nccl_bw_sigmoid(comm_type, vol, system['nvlink_size'], num_nodes=n_gpus//system['nvlink_size']) if empirical else vol / system['ib_bandwidth'] 
            else:
                t_comm = 0
    else:
        vol = vols
        n_gpus = n_gpuss
        comm_type = comm_types
        topology = topologys
        if np.isnan(vol): return 0
        if empirical: assert system['nvlink_size'] <= 4, 'Empirical comms measurements not intended for nvlink_size > 4 currently'
        if topology == 'nvlink': # use nvlink
            t_comm  =  vol / nccl_bw_sigmoid(comm_type, vol, system['nvlink_size'], num_nodes=1) if empirical else vol / system['nvlink_bandwidth']
        elif topology == 'ib':
            t_comm  = vol / nccl_bw_sigmoid(comm_type, vol, system['nvlink_size'], num_nodes=n_gpus//system['nvlink_size']) if empirical else vol / system['ib_bandwidth'] 
        else:
            t_comm = 0
    return t_comm * 10**3

def get_total_time(t_comp, t_comm, system={}, use_max=False):
    if use_max:
        return max(t_comp, t_comm) # overlap
    else:
        return t_comp + t_comm

def get_topology(n_gpus, system={}):
    if np.isnan(n_gpus): return None
    has_nvlink = system['nvlink_size'] > 1
    if has_nvlink:
        topology =  ('nvlink' if n_gpus <= system['nvlink_size'] else 'ib')
    else:
        topology = 'ib'
    return topology
