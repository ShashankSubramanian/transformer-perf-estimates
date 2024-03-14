import numpy as np
import json
from time_projections import nccl_bw_sigmoid

class Estimates():
    def __init__(self, config_path='config.json', system = None):
        if system is None:
            with open(config_path, 'r') as file:
                system = json.load(file)
        self.system = system
        self.element_size = system['element_size']
        self.mask_element_size = system['mask_element_size']
        self.flops_units = system['flops_units']
        if self.system['empirical']:
            with open('./measurements/comm/nccl-benchmark-21209538.out', 'r') as data:
                self.bwdata = json.load(data)
        else:
            self.bwdata = None

    def set_stats(self,
                  name = 'nnlayer',
                  use_tensor_cores = True,
                  flops_fwd = 0,
                  mem_fwd = 0,
                  activation_buffer = 0, # store for bwd pass
                  weights_mem = 0,
                  weights_grad_mem = 0,
                  comm_fwd = 0, 
                  comm_fwd_type = 'none',
                  comm_fwd_size = 0,
                  comm_fwd_topology = 'none',
                  comm_fwd_nv = 0,
                  flops_bwd = 0,
                  mem_bwd = 0,
                  comm_bwd = 0, 
                  comm_bwd_type = 'none',
                  comm_bwd_size = 0,
                  comm_bwd_topology = 'none',
                  comm_bwd_nv = 0):
        self.name = name
        self.use_tensor_cores = use_tensor_cores
        self.flops_fwd = flops_fwd
        self.mem_fwd = mem_fwd
        self.activation_buffer = activation_buffer
        self.weights_mem = weights_mem
        self.weights_grad_mem = weights_grad_mem
        self.comm_fwd = comm_fwd
        self.comm_fwd_type = comm_fwd_type
        self.comm_fwd_size = comm_fwd_size
        self.comm_fwd_topology = comm_fwd_topology
        self.comm_fwd_nv = comm_fwd_nv
        self.flops_bwd = flops_bwd
        self.mem_bwd = mem_bwd
        self.comm_bwd = comm_bwd
        self.comm_bwd_type = comm_bwd_type
        self.comm_bwd_size = comm_bwd_size
        self.comm_bwd_topology = comm_bwd_topology
        self.comm_bwd_nv = comm_bwd_nv

        # what to see 
        self.stats = {"name": self.name,
                      "weights_mem": self.weights_mem,
                      "weights_grad_mem": self.weights_grad_mem,
                      "flops_fwd": self.flops_fwd,
                      "activation_buffer": self.activation_buffer,
                      "comm_fwd": self.comm_fwd,
                      "comm_fwd_type": self.comm_fwd_type,
                      "flops_bwd": self.flops_bwd,
                      "comm_bwd": self.comm_bwd,
                      "comm_bwd_type": self.comm_bwd_type}

    def get_stats(self):
        return self.stats

    def get_time(self, flops, mem, comm, nvlink_size, comm_size, comm_type, comm_topology):
#         print("name,sizes = ",self.name,nvlink_size,comm_size)
        t_comp = self.get_time_flops(flops)
        t_mem = self.get_time_mem(mem)
        intensity = t_comp / t_mem
        t_comm = self.get_time_comm(comm, nvlink_size, comm_size, comm_type, comm_topology)  
        return max(t_comp, t_mem) + t_comm, t_comm, intensity

    def compute_time(self):
        self.stats['t_fwd'], self.stats['t_fwd_comm'], self.stats['intensity_fwd'] = self.get_time(self.flops_fwd, self.mem_fwd, 
                                                                                     self.comm_fwd, self.comm_fwd_nv, \
                                                                                                   self.comm_fwd_size, \
                                                                                     self.comm_fwd_type, self.comm_fwd_topology)
        self.stats['t_bwd'], self.stats['t_bwd_comm'], self.stats['intensity_bwd'] = self.get_time(self.flops_bwd, self.mem_bwd, 
                                                                                     self.comm_bwd, self.comm_bwd_nv, \
                                                                                                   self.comm_bwd_size, \
                                                                                     self.comm_bwd_type, self.comm_bwd_topology)
        self.stats['t'] = self.stats['t_fwd'] + self.stats['t_bwd']


    def get_time_flops(self, flops):
        """
            input:
                flops   = number of floating point operations
                
            output:
                t_flops = time for computations in seconds
        
        """
        hardware_flops = self.system['matrix_flops_fp16'] if self.use_tensor_cores else self.system['vector_flops_fp16']
        t_flops = flops / hardware_flops
        return t_flops

    def get_time_mem(self, mem):
        """ 
            input: 
                mem      = memory in GB 
            output:
                t_mem    = time in secods
        """
        hbm_bandwidth = self.system['hbm_bandwidth']
        t_mem = mem / hbm_bandwidth
        return t_mem

    def get_time_comm(self, vol, nvlink_size ,n_gpus, comm_type, topology):
        """
            input:
                vol           = communication volume in GB
                nvlink_size   = number of gpus with NVLINK all-to-all connectivity
                n_gpus        = number of gpus used for communication
                comm_type     = type of communication, e.g. 'allreduce'
                topology      = 'nvlink', 'ib', 'mixed' when both are present
            
            output:
                t_comm        = time for communication in seconds
        """
        
        if np.isnan(vol) or vol == 0: return 0 # if no communication volume return 0
        
        
        assert nvlink_size <= self.system["nvlink_size"], "nvlink size of "+str(nvlink_size)+" can not be larger than "+str(self.system["nvlink_size"]) # for 1D they must be equal, for 2D we give some flexibility
        
        if self.system['empirical']:  # use empirical fits
#             print("empirical!")
            assert nvlink_size <= 4, 'Empirical comms measurements not intended for nvlink_size > 4 currently'
            if topology == 'nvlink': 
                t_comm  =  vol / nccl_bw_sigmoid(comm_type, vol, self.bwdata, nvlink_size, num_nodes=1) 
            elif topology == 'ib':
                t_comm  = vol / nccl_bw_sigmoid(comm_type, vol, self.bwdata, nvlink_size, num_nodes=n_gpus//nvlink_size) 
            elif topology == 'mixed':
                t_comm = 0 #TODO empirical for 'mixed' topology
        elif self.system['formula']: # use formulae
#             print("formula!")
            comm_params = self.system["ib_latency"], self.system["nvlink_latency"], \
                          self.system["ib_bandwidth"], self.system["nvlink_bandwidth"]
            t_comm = self.comm_time_formula(comm_type, vol, topology, comm_params, nvlink_size, n_gpus)
        else: # just take ratio of coomunication volume to bandwidth 
            if topology == 'nvlink': 
                t_comm  = vol / self.system['nvlink_bandwidth']
            elif topology == 'ib':
                t_comm  = vol / self.system['ib_bandwidth'] 
            elif topology == 'mixed':
                t_comm = vol / self.system['ib_bandwidth'] 
        return t_comm


    def comm_time_formula(self, op, message_size, topology, comm_params, nvlink_size, system_size):
        """
        Computes expected time for communication based on analytical expressions
        Inputs:
            op              = communication type: 'allreduce', 'allgather', 'reducescatter', 'gather', 'broadcast'
            message_size    = communication volume in bytes (int/float)
            topology        = 'nvlink', 'ib', or 'mixed'
            nvlink_size     = number of GPUs in the node (or nvlink island)
            system_size     = number of nodes (must be equal to or higher than gpus_per_node)
        Returns:
            time            = communication time in seconds
        """
        latency_ib, latency_nvlink, bandwidth_ib, bandwidth_nvlink = comm_params
        inv_band_ib = 1.0/bandwidth_ib
        inv_band_nvlink = 1.0/bandwidth_nvlink
#         print(latency_ib, latency_nvlink, bandwidth_ib, bandwidth_nvlink)
        match topology:
            case 'nvlink':
                r = 1
                assert nvlink_size >= system_size, 'nvlink size of '+str(nvlink_size)+\
                      ' should be greater than or equal to system size of '+str(system_size)
                nvlink_size = system_size
                latency_ib = 0
                inv_band_ib = 0
            case 'mixed':
                r = np.ceil(system_size / nvlink_size) #TODO account for a node being partially used
            case 'ib':
                assert nvlink_size == 1
#                 latency_nvlink = 0
#                 bandwidth_nvlink = 0
                r  = system_size
                latency_nvlink = 0
                inv_band_nvlink = 0
            
         
        match op:
            case 'allreduce':
                time = 2*np.log2(max(r+2,1))*(latency_ib + message_size*inv_band_ib) \
                        + (np.log2(max(nvlink_size-1,1))*latency_nvlink + (nvlink_size-1)*message_size*inv_band_nvlink)
            case 'allgather':
                message = message_size/system_size
                time = np.log2(max(r,1))*latency_ib + np.log2(max(nvlink_size,1))*latency_nvlink \
                       + r*(nvlink_size-1)*message*inv_band_nvlink \
                       + (system_size-1 - r*(nvlink_size-1))*message*inv_band_ib
            case 'reducescatter':
                message = message_size/system_size
                time = np.log2(max(r,1))*latency_ib + np.log2(max(nvlink_size,1))*latency_nvlink \
                       + r*(nvlink_size-1)*message*inv_band_nvlink \
                       + (system_size-1 - r*(nvlink_size-1))*message*inv_band_ib
            case 'gather':
                message = message_size/system_size
                time = np.log2(max(r,1))*latency_ib + np.log2(max(nvlink_size,1))*latency_nvlink \
                       + r*(nvlink_size-1)*message*inv_band_nvlink \
                       + (system_size-1 - r*(nvlink_size-1))*message*inv_band_ib
            case 'broadcast':
#                 print(nvlink_size,bandwidth_nvlink,(system_size - r*nvlink_size),bandwidth_ib)
                time = latency_ib*np.log2(max(system_size - r*nvlink_size,1)) + latency_nvlink*np.log2(max(nvlink_size,1)) \
                        + message_size*(r*nvlink_size*inv_band_nvlink + (system_size - r*nvlink_size)*inv_band_ib)
            case _:
                time = 0
    
        return time
            
