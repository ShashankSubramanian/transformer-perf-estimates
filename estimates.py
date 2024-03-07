import numpy as np
import json
from time_projections import nccl_bw_sigmoid

class Estimates():
    def __init__(self, config_path='config.json'):
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
                  flops_bwd = 0,
                  mem_bwd = 0,
                  comm_bwd = 0, 
                  comm_bwd_type = 'none',
                  comm_bwd_size = 0,
                  comm_bwd_topology = 'none'):
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
        self.flops_bwd = flops_bwd
        self.mem_bwd = mem_bwd
        self.comm_bwd = comm_bwd
        self.comm_bwd_type = comm_bwd_type
        self.comm_bwd_size = comm_bwd_size
        self.comm_bwd_topology = comm_bwd_topology

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

    def get_time(self, flops, mem, comm, comm_size, comm_type, comm_topology):
        t_comp = self.get_time_flops(flops)
        t_mem = self.get_time_mem(mem)
        intensity = t_comp / t_mem
        t_comm = self.get_time_comm(comm, comm_size, comm_type, comm_topology)  
        return max(t_comp, t_mem) + t_comm, t_comm, intensity

    def compute_time(self):
        self.stats['t_fwd'], self.stats['t_fwd_comm'], self.stats['intensity_fwd'] = self.get_time(self.flops_fwd, self.mem_fwd, 
                                                                                     self.comm_fwd, self.comm_fwd_size,
                                                                                     self.comm_fwd_type, self.comm_fwd_topology)
        self.stats['t_bwd'], self.stats['t_bwd_comm'], self.stats['intensity_bwd'] = self.get_time(self.flops_bwd, self.mem_bwd, 
                                                                                     self.comm_bwd, self.comm_bwd_size,
                                                                                     self.comm_bwd_type, self.comm_bwd_topology)
        self.stats['t'] = self.stats['t_fwd'] + self.stats['t_bwd']


    def get_time_flops(self, flops):
        hardware_flops = self.system['matrix_flops_fp16'] if self.use_tensor_cores else self.system['vector_flops_fp16']
        t_flops = flops / hardware_flops
        return t_flops * 10**3

    def get_time_mem(self, mem):
        hbm_bandwidth = self.system['hbm_bandwidth']
        t_mem = mem / hbm_bandwidth
        return t_mem * 10**3

    def get_time_comm(self, vol, n_gpus, comm_type, topology):
        system = self.system
        empirical = system['empirical']
        if np.isnan(vol) or vol == 0: return 0
        if empirical: assert system['nvlink_size'] <= 4, 'Empirical comms measurements not intended for nvlink_size > 4 currently'
        if topology == 'nvlink': # use nvlink
            t_comm  =  vol / nccl_bw_sigmoid(comm_type, vol, self.bwdata, system['nvlink_size'], num_nodes=1) if empirical else vol / system['nvlink_bandwidth']
        elif topology == 'ib':
            t_comm  = vol / nccl_bw_sigmoid(comm_type, vol, self.bwdata, system['nvlink_size'], num_nodes=n_gpus//system['nvlink_size']) if empirical else vol / (system['ib_bandwidth'])
        else:
            t_comm = 0
        return t_comm * 10**3


            
