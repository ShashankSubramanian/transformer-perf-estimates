import numpy as np
import json

class Estimates():
    def __init__(self, system=None):
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
                  comm_bwd_topology = 'none',
                  recompute = False,
                  remat = False):
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
        # TODO: recompute not tested/implemented fully
        self.recompute = recompute # do fwd pass again

        # what to see 
        self.stats = {"name": self.name,
                      "weights_mem": self.weights_mem,
                      "weights_grad_mem": self.weights_grad_mem,
                      "flops_fwd": self.flops_fwd,
                      "activation_buffer": self.activation_buffer * (not remat),
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
        t_mem_exposed = max(t_mem - t_comp, 0)
        return max(t_comp, t_mem) + t_comm, t_comm, t_comp, t_mem_exposed, intensity

    def compute_time(self):
        self.stats['t_fwd'], self.stats['t_fwd_comm'], self.stats['t_fwd_comp'], self.stats['t_fwd_mem'], self.stats['intensity_fwd'] = \
                self.get_time(self.flops_fwd, self.mem_fwd, self.comm_fwd, self.comm_fwd_size, self.comm_fwd_type, self.comm_fwd_topology)

        self.stats['t_bwd'], self.stats['t_bwd_comm'], self.stats['t_bwd_comp'], self.stats['t_bwd_mem'], self.stats['intensity_bwd'] = \
                 self.get_time(self.flops_bwd, self.mem_bwd, self.comm_bwd, self.comm_bwd_size, self.comm_bwd_type, self.comm_bwd_topology)

        if self.recompute:
            self.stats['t_bwd'] += self.stats['t_fwd']
        self.stats['t'] = self.stats['t_fwd'] + self.stats['t_bwd']


    def get_time_flops(self, flops):
        ''' time to execute flops '''
        hardware_flops = self.system['matrix_flops_fp16'] if self.use_tensor_cores else self.system['vector_flops_fp16']
        hardware_flops *= 0.8 # some avg efficiency
        t_flops = 20 * 1E-6 + flops / hardware_flops
        return t_flops

    def get_time_mem(self, mem):
        '' ' time to fetch data from hbm '''
        hbm_bandwidth = self.system['hbm_bandwidth']
        t_mem = mem / hbm_bandwidth
        return t_mem

    def get_time_comm(self, vol, n_gpus, comm_type, topology):
        ''' time for communication 
            comm_type: 'allreduce', 'allgather', 'reducescatter', 'broadcast'
            vol: message size in GB
            topology: number of GPUs in a first bandwidth domain
            n_gpus: total number of GPUs in this comm
        '''
        system = self.system
        empirical = system['empirical']
        if np.isnan(vol) or vol == 0: return 0
        if empirical: assert system['nvlink_size'] <= 4, 'Empirical comms measurements not intended for nvlink_size > 4 currently'

        # get some system configs
        nvs = system['nvlink_size']
        ls = system['ib_latency']
        lf = system['nvlink_latency']
        es = system['ib_eff']
        ef = system['nvlink_eff']
        bs = system['ib_bandwidth'] * es
        bf = system['nvlink_bandwidth'] * ef
        #bf = system['nvlink_bandwidth'] * (topology - 1) * ef
        nic_factor = system['nic_factor'] # number of nics per GPU (can be 0.5 etc)
        assert topology <= nvs, 'you have provisioned more gpus than nvlink domain size for fast comm'

        if n_gpus == 1:
            return 0 # no comms

        # nodes here just means number of nvlink domains: careful
        nodes = n_gpus // topology # note topology controls how many gpus you are using in the nvlink domain
        # ring corrections
        if comm_type == 'p2p':
            assert n_gpus == 2, 'p2p comms is btw pair of gpus'
            if topology == 1: # all ib
                t_comm = ls * (n_gpus - 1) + vol / bs
            else:
                t_comm = lf * (n_gpus - 1) + vol / bf # faster p2p
            return t_comm

        if comm_type not in ['reduce', 'broadcast']:
            correction = (n_gpus - 1) / n_gpus
        else:
            correction = 1

        if topology == 1: # all ib
            t_comm = ls * (n_gpus - 1) + correction * vol / bs
        elif nodes == 1:
            t_comm = lf * (n_gpus - 1) + correction * vol / bf # has nvlink and only one node
        else:
            # multiple rings 
            num_rings = nic_factor * topology # number of nics per node
            t1 = correction * vol / (num_rings * bs)
            t2 = correction * vol / bf
            t_comm = max(t1, t2)
            if comm_type == 'allreduce':
                t_comm *= 2
            t1_l = ls * (nodes - 1)
            t2_l = lf * (n_gpus - nodes)
            t_comm += (t1_l + t2_l) # add latencies

        return t_comm
