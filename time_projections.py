import numpy as np
import json

def nccl_bw_sigmoid(op, message_size, bwdata, gpus_per_node=4, num_nodes=2):
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

