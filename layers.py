from estimates import Estimates

class Linear(Estimates):
    def __init__(self, name, b, l, e, f, 
                 has_bias=False,
                 parallelism={'dim1': 1, 'dim2': 1}, 
                 topology={'dim1': 'none', 'dim2': 'none'},
                 system=None):
        """
        nn.Linear layer estimates
        parameters: b: batch size
                    l: seq length
                    e: embedding dim/1st dim
                    f: hidden dim/2nd dim
                    has_bias: whether bias term is added
        
        layer arithmetic: 
            forward pass :  
                Y = X * W + B
                (b,l,f) = (b,l,e) * (e,f) + (1,f)
            backward pass: (L = loss)
                dL/dX = dL/dY * W^T
                (b,l,e) = (b,l,f) * (f,e)
                dL/dW = X^T * dL/dY
                (e,f) = (e, bl) * (bl,f) 
                dL/dB = dL/dY * 1
                (1,f) = \sum_{b,l} (b,l,f)
    
        comments: 
                only 1D parallelism, if f is sharded (dim2) 
                then no comms in fwd, allreduce in bwd, if
                e is sharded (dim1), then allreduce in fwd,
                nothing in bwd
        """

        super().__init__(system=system)
        
        flops_per_mult = 1 * self.flops_units
        flops_per_add = 1 * self.flops_units
        element_size = self.element_size

        m2 = parallelism['dim1']
        m1 = parallelism['dim2']
        t2 = topology['t1']
        t1 = topology['t2']

        # either e is sharded or f depending on which layer; if both use summa or need to implement an allgather
        assert (m1 == 1 or m2 == 1), "error: using 1d parallelism function, both dims of weight matrix cannot be sharded"

        f_local = f // m1
        e_local = e // m2

        ####### forward pass ########
        # total flops
        flops_fwd = b * l * f_local * (e_local * flops_per_mult + (e_local - 1) * flops_per_add)
        if has_bias:
            flops_fwd += b * l * (f_local * flops_per_add)
            
        # total mem
        activation_in_mem = (b * l * e_local) * element_size
        activation_out_mem = (b * l * f_local) * element_size
        activation_buffer = (b * l * e_local) * element_size # store for bwd pass
        weights_mem = (e_local * f_local) * element_size
        if has_bias:
            weights_mem += (1 * f_local) * element_size
        mem_fwd = activation_in_mem + activation_out_mem + weights_mem

        # sync/comm layers
        comm_fwd = (b * l * f) * element_size if m2 > 1 else 0
        comm_fwd_type = 'reducescatter'
        comm_fwd_size = m2
        comm_fwd_topology = t2

        ####### backward pass #######
        xgrad_flops = b * l * e_local * (f_local * flops_per_mult + (f_local - 1) * flops_per_add)
        wgrad_flops = e_local * f_local * (b * l * flops_per_mult + (b * l - 1) * flops_per_add)
        bgrad_flops = b * l * f_local * flops_per_add if has_bias else 0
        flops_bwd = xgrad_flops + wgrad_flops + bgrad_flops

        wgrad_mem = (e_local * f_local) * element_size
        bgrad_mem = (1 * f_local) * element_size if has_bias else 0
        weights_grad_mem = wgrad_mem + bgrad_mem
        xgrad_mem = (b * l * e_local) * element_size 
        ygrad_mem = (b * l * f_local) * element_size  

        num_bwd_ops = 3 if has_bias else 2 # need to bring ygrad_mem multiple times
        mem_bwd = weights_grad_mem + xgrad_mem + num_bwd_ops * ygrad_mem + weights_mem + activation_buffer

        # sync/comm layers
        comm_bwd = (b * l * e) * element_size if m1 > 1 else 0
        comm_bwd_type = 'reducescatter'
        comm_bwd_size = m1
        comm_bwd_topology = t1

        self.set_stats(name,
                       flops_fwd = flops_fwd,
                       use_tensor_cores = True,
                       mem_fwd = mem_fwd,
                       activation_buffer = activation_buffer,
                       weights_mem = weights_mem,
                       weights_grad_mem = weights_grad_mem,
                       comm_fwd = comm_fwd, 
                       comm_fwd_type = comm_fwd_type,
                       comm_fwd_size = comm_fwd_size,
                       comm_fwd_topology = comm_fwd_topology,
                       flops_bwd = flops_bwd,
                       mem_bwd = mem_bwd,
                       comm_bwd = comm_bwd, 
                       comm_bwd_type = comm_bwd_type,
                       comm_bwd_size = comm_bwd_size,
                       comm_bwd_topology = comm_bwd_topology)

    def get_stats(self):
        self.compute_time()
        return self.stats

class LinearSeqp(Estimates):
    def __init__(self, name, b, l, e, f, 
                 has_bias=False,
                 parallelism={'dim1': 1, 'dim2': 1, 'dimseq': 1}, 
                 topology={'dim1': 'none', 'dim2': 'none', 'dimseq': 'none'},
                 system=None):
        """
        nn.Linear layer estimates
        parameters: b: batch size
                    l: seq length
                    e: embedding dim/1st dim
                    f: hidden dim/2nd dim
                    has_bias: whether bias term is added
        
        layer arithmetic: 
            forward pass :  
                Y = X * W + B
                (b,l,f) = (b,l,e) * (e,f) + (1,f)
            backward pass: (L = loss)
                dL/dX = dL/dY * W^T
                (b,l,e) = (b,l,f) * (f,e)
                dL/dW = X^T * dL/dY
                (e,f) = (e, bl) * (bl,f) 
                dL/dB = dL/dY * 1
                (1,f) = \sum_{b,l} (b,l,f)
    
        comments: 
            weights are 1d parallelized and shared in seqp group
            seq is embarassingly parallel in fwd, allreduce in bwd for wgrads
        """

        super().__init__(system=system)
        
        flops_per_mult = 1 * self.flops_units
        flops_per_add = 1 * self.flops_units
        element_size = self.element_size

        m2 = parallelism['dim1']
        m1 = parallelism['dim2']
        mseq = parallelism['dimseq']
        t2 = topology['t1']
        t1 = topology['t2']
        tseq = topology['tseq']


        # either e is sharded or f depending on which layer; if both use summa or need to implement an allgather
        assert (m1 == 1 or m2 == 1), "error: using 1d parallelism function, both dims of weight matrix cannot be sharded"
        f_local = f // m1
        e_local = e // m2

        l_local = l // mseq


        ####### forward pass ########
        # total flops
        flops_fwd = b * l_local * f_local * (e_local * flops_per_mult + (e_local - 1) * flops_per_add)
            
        # total mem
        activation_in_mem = (b * l_local * e_local) * element_size
        activation_out_mem = (b * l_local * f_local) * element_size
        activation_buffer = (b * l_local * e_local) * element_size # store for bwd pass
        weights_mem = (e_local * f_local) * element_size
        mem_fwd = activation_in_mem + activation_out_mem + weights_mem

        # sync/comm layers: no comms for seq
        comm_fwd = (b * l * f) * element_size if m2 > 1 else 0
        comm_fwd_type = 'reducescatter'
        comm_fwd_size = m2
        comm_fwd_topology = t2

        ####### backward pass #######
        xgrad_flops = b * l_local * e_local * (f_local * flops_per_mult + (f_local - 1) * flops_per_add)
        wgrad_flops = e_local * f_local * (b * l_local * flops_per_mult + (b * l_local - 1) * flops_per_add)
        flops_bwd = xgrad_flops + wgrad_flops

        wgrad_mem = (e_local * f_local) * element_size
        weights_grad_mem = wgrad_mem 
        xgrad_mem = (b * l_local * e_local) * element_size 
        ygrad_mem = (b * l_local * f_local) * element_size  

        num_bwd_ops = 2 # need to bring ygrad_mem multiple times
        mem_bwd = weights_grad_mem + xgrad_mem + num_bwd_ops * ygrad_mem + weights_mem + activation_buffer

        # sync/comm layers
        comm_bwd = (b * l * e) * element_size if m1 > 1 else 0
        comm_bwd_type = 'reducescatter'
        comm_bwd_size = m1
        comm_bwd_topology = t1

        self.set_stats(name,
                       flops_fwd = flops_fwd,
                       use_tensor_cores = True,
                       mem_fwd = mem_fwd,
                       activation_buffer = activation_buffer,
                       weights_mem = weights_mem,
                       weights_grad_mem = weights_grad_mem,
                       comm_fwd = comm_fwd, 
                       comm_fwd_type = comm_fwd_type,
                       comm_fwd_size = comm_fwd_size,
                       comm_fwd_topology = comm_fwd_topology,
                       flops_bwd = flops_bwd,
                       mem_bwd = mem_bwd,
                       comm_bwd = comm_bwd, 
                       comm_bwd_type = comm_bwd_type,
                       comm_bwd_size = comm_bwd_size,
                       comm_bwd_topology = comm_bwd_topology)

    def get_stats(self):
        self.compute_time()
        return self.stats

class LinearSumma(Estimates):
    def __init__(self, name, b, l, e, f, 
                 parallelism={'dim1': 1, 'dim2': 1}, 
                 topology={'dim1': 'none', 'dim2': 'none'}, 
                 system=None):
        """
        nn.Linear layer estimates
        parameters: b: batch size
                    l: seq length
                    e: embedding dim/1st dim
                    f: hidden dim/2nd dim
                    has_bias: whether bias term is added
        
        layer arithmetic: 
            forward pass :  
                Y = X * W
                (b,l/m2,f/m1) = (b,l/m2,e/m1) * (e/m2,f/m1) 
                for i=1:nb:
                    broadcast (b,l/m2,e/nb) row-wise, (e/nb,f/m1) column-wise
                    addmm (b,l/m2,e/nb) * (e/nb,f/m1) locally
                
            backward pass :
                dL/dX = dL/dY * W^T
                (b,l/m2,e/m1) = (b,l/m2,f/m1) * (f/m1,e/m2)
                for i=1:nb
                    broadcast slice of W (e/nb,f/m1) column-wise
                    temp = (b,l/m2,f/m1) * (f/m1,e/nb) locally
                    reduce temp to nb block within row
                --------------------------------------------------------------        
                dL/dW = X^T * dL/dY
                (e/m2,f/m1) = (e/m1,bl/m2) * (bl/m2,f/m1) 
                for i=1:nb
                    broadcast slice of X (bl/m2,e/nb) row-wise
                    temp = (e/nb,bl/m2) * (bl/m2,f/m1) locally
                    reduce temp to nb block within col
    
        comments: 
            nb = m if square grid of procs; else it's some small number (?)
        """

        super().__init__(system=system)
        
        flops_per_mult = 1 * self.flops_units
        flops_per_add = 1 * self.flops_units
        element_size = self.element_size

        m2 = parallelism['dim1']
        m1 = parallelism['dim2']
        t2 = topology['t1']
        t1 = topology['t2']
        m1_parallel = (m1 > 1)
        m2_parallel = (m2 > 1)

        l_local = l // m2
        f_local = f // m1
        e_local_1 = e // m1
        e_local_2 = e // m2

        ####### forward pass ########
        # total flops: counts the total flops across all summa itrs (maybe not the best way, but fine for now..)
        flops_fwd = b * l_local * f_local * (e * flops_per_mult + e * flops_per_add) # e outer products and e addns in total
        
        # total mem
        activation_buffer = (b * l_local * e_local_1) * element_size # store for bwd pass
        weights_mem = (e_local_2 * f_local) * element_size
        
        # careful, nb is arbitrarily chosen here
        # panel size e/n_b is some value << e/max(m1,m2)
#        n_b = e // 512 if m1 != m2 else m1
        n_b = self.system['summa_nb']
        self.n_b = n_b
        mem_fwd  = (b * l_local * e + e * f_local + b * l_local * f_local * n_b) * element_size

        # sync/comm layers
        comm_fwd = [m1_parallel * (b * l_local * e) * element_size,
                    m2_parallel * (e * f_local) * element_size] # both inputs are broadcasted in diff dims
        comm_fwd_type = ["broadcast", "broadcast"]
        comm_fwd_size = [m1, m2]
        comm_fwd_topology = [t1, t2]

        ####### backward pass #######
        xgrad_flops = b * l_local * e * (f_local * flops_per_mult + (f_local - 1) * flops_per_add) 
        wgrad_flops = e * f_local * (b * l_local * flops_per_mult + (b * l_local - 1) * flops_per_add)
        flops_bwd = xgrad_flops + wgrad_flops
        
        weights_grad_mem = (e_local_2 * f_local) * element_size
        mem_bwd =  (f_local * e + b * l_local * f_local * n_b + b * l_local * e) * element_size
        mem_bwd +=  (e * b * l_local + b * l_local * f_local * n_b +  e * f_local) * element_size

        # sync/comm layers
        comm_bwd = [m2_parallel * (e * f_local) * element_size, # dl/dx = dl/dy * wt
                    m1_parallel * (b * l_local * e) * element_size, # reduce result
                    m1_parallel * (b * l_local * e) * element_size, # dl/dw = xt * dl/dy
                    m2_parallel * (e * f_local) * element_size] # reduce temp result
        comm_bwd_type = ["broadcast", "reduce", "broadcast", "reduce"]
        comm_bwd_size = [m2, m1, m1, m2]
        comm_bwd_topology = [t2, t1, t1, t2]

        self.set_stats(name,
                       flops_fwd = flops_fwd,
                       use_tensor_cores = True,
                       mem_fwd = mem_fwd,
                       activation_buffer = activation_buffer,
                       weights_mem = weights_mem,
                       weights_grad_mem = weights_grad_mem,
                       comm_fwd = comm_fwd, 
                       comm_fwd_type = comm_fwd_type,
                       comm_fwd_size = comm_fwd_size,
                       comm_fwd_topology = comm_fwd_topology,
                       flops_bwd = flops_bwd,
                       mem_bwd = mem_bwd,
                       comm_bwd = comm_bwd, 
                       comm_bwd_type = comm_bwd_type,
                       comm_bwd_size = comm_bwd_size,
                       comm_bwd_topology = comm_bwd_topology)


    def compute_time_summa(self, flops, mem, comms, comm_sizes, comm_types, comm_tops, verbose=False):
        t_comp = self.get_time_flops(flops)
        t_mem  = self.get_time_mem(mem)
        t_compute = max(t_comp, t_mem)
        intensity = t_comp / t_mem

        t_for_one_comm_set = 0 # typically two broadcasts or one broadcast, one reduce

        for c, comm in enumerate(comms):
            comm_size = comm_sizes[c]
            comm_type = comm_types[c]
            comm_top = comm_tops[c]
#            t_for_nb_comms = self.get_time_comm(comm, comm_size, comm_type, comm_top)  
#            t_for_one_comm_set += t_for_nb_comms / self.n_b 

            # for any comm, divide by n_b for each itr (do it this way because of the comm latencies)
            t_for_one_comm_set += self.get_time_comm(comm / self.n_b, comm_size, comm_type, comm_top)  


        # overlap some comms with compute
        t_comm = t_for_one_comm_set + max(t_for_one_comm_set * self.n_b - t_compute, 0)
        t = t_compute + t_comm
        if verbose:
            print(t_compute, t_comp, t_mem, flops, mem, t_comm, t_for_one_comm_set, t_for_one_comm_set * self.n_b)
        return t, t_comm, intensity

    def compute_time(self): # need to overwrite to implement pipelined/overlapping comms
        # forward time
        self.stats['t_fwd'], self.stats['t_fwd_comm'], self.stats['intensity_fwd'] = self.compute_time_summa(self.flops_fwd,
                                                                                                             self.mem_fwd,
                                                                                                             self.comm_fwd,
                                                                                                             self.comm_fwd_size,
                                                                                                             self.comm_fwd_type,
                                                                                                             self.comm_fwd_topology)
        self.stats['t_bwd'], self.stats['t_bwd_comm'], self.stats['intensity_bwd'] = self.compute_time_summa(self.flops_bwd,
                                                                                                             self.mem_bwd,
                                                                                                             self.comm_bwd,
                                                                                                             self.comm_bwd_size,
                                                                                                             self.comm_bwd_type,
                                                                                                             self.comm_bwd_topology)
        self.stats['t'] = self.stats['t_fwd'] + self.stats['t_bwd']


    def get_stats(self):
        self.compute_time()
        return self.stats

class Bias(Estimates):
    def __init__(self, name, b, l, f, 
                 parallelism={'dim1': 1, 'dim2': 1}, 
                 topology={'dim1': 'none', 'dim2': 'none'},
                 system=None):
        """
        bias layer estimates
        parameters: b: batch size
                    l: seq length
                    f: hidden dim/2nd dim
        
        layer arithmetic: 
            forward pass :  
                Y = Y + B
                (b,l/m2,f/m1) = (b,l/m2,f/m1) + (1,f/m1)

            backward pass :
                dL/dB = dL/dY * 1
                (1,f/m1) = \sum_{b,l/m2} (b,l/m2,f/m1)
                allreduce on m2 dim
    
        comments: 
        """

        super().__init__(system=system)
        
        flops_per_mult = 1 * self.flops_units
        flops_per_add = 1 * self.flops_units
        element_size = self.element_size

        m2 = parallelism['dim1']
        m1 = parallelism['dim2']
        t2 = topology['t1']
        t1 = topology['t2']
        m1_parallel = (m1 > 1)
        m2_parallel = (m2 > 1)


        l_local = l // m2
        f_local = f // m1

        ####### forward pass ########
        weights_mem = (1 * f_local) * element_size
        flops_fwd = b * l_local * (f_local * flops_per_add)
        mem_fwd = (b * l_local * f_local + 1 * f_local) * element_size

        ####### backward pass #######
        weights_grad_mem = (1 * f_local) * element_size
        flops_bwd = b * l_local * f_local * flops_per_add
        mem_bwd = (b * l_local * f_local + 1 * f_local) * element_size

        # comms
        comm_bwd = m2_parallel * (f_local) * element_size # allreduce bias
        comm_bwd_type = "allreduce"
        comm_bwd_size = m2
        comm_bwd_topology = t2

        self.set_stats(name,
                       flops_fwd = flops_fwd,
                       use_tensor_cores = False,
                       mem_fwd = mem_fwd,
                       activation_buffer = 0,
                       weights_mem = weights_mem,
                       weights_grad_mem = weights_grad_mem,
                       flops_bwd = flops_bwd,
                       mem_bwd = mem_bwd,
                       comm_bwd = comm_bwd, 
                       comm_bwd_type = comm_bwd_type,
                       comm_bwd_size = comm_bwd_size,
                       comm_bwd_topology = comm_bwd_topology)

    def get_stats(self):
        self.compute_time()
        return self.stats

class Act(Estimates):
    def __init__(self, name, m, system=None):
        """
        activation function estimates
        parameters: m: size of act
        
                    
        layer arithmetic: 
            forward pass: 
                Y = GELU(X)
            backward pass:
                dL/dX = dL/dY * f(X)
            """

        super().__init__(system=system)
        
        flops_per_mult = 1 * self.flops_units
        flops_per_add = 1 * self.flops_units
        element_size = self.element_size

        # total flops
        flops_fwd = m * flops_per_mult
        
        #total mem
        activation_in_mem = (m) * element_size
        activation_out_mem = (m) * element_size
        activation_buffer = (m) * element_size
        mem_fwd = activation_in_mem + activation_out_mem
        
        ####### backward pass #######
        flops_bwd =  m * flops_per_mult
        activation_grad_mem = 2 * (m) * element_size
        mem_bwd = activation_grad_mem + activation_buffer

        self.set_stats(name,
                       flops_fwd = flops_fwd,
                       use_tensor_cores = False,
                       mem_fwd = mem_fwd,
                       activation_buffer = activation_buffer,
                       weights_mem = 0,
                       weights_grad_mem = 0,
                       flops_bwd = flops_bwd,
                       mem_bwd = mem_bwd)

    def get_stats(self):
        self.compute_time()
        return self.stats

class DropOut(Estimates):
    def __init__(self, name, m, recompute=False, remat=False, system=None):
        """
        dropout function estimates
        parameters: m: size of act
        
                    
        layer arithmetic: 
            forward pass: 
                Y = random_mask(X)
                (b,l,e) = (b,l,e) * (b,l,e)
            backward pass:
                dl/dX = dl/dY * random_mask
        """

        super().__init__(system=system)
        
        flops_per_mult = 1 * self.flops_units
        flops_per_add = 1 * self.flops_units
        element_size = self.element_size
        mask_element_size = self.mask_element_size

        # total flops
        flops_fwd = m * flops_per_mult
        
        #total mem
        activation_in_mem = (m) * element_size
        activation_in_mem += (m) * mask_element_size
        activation_out_mem = (m) * element_size
        activation_buffer = (m) * mask_element_size
        mem_fwd = activation_in_mem + activation_out_mem
        
        ####### backward pass #######
        flops_bwd =  m * flops_per_mult
        activation_grad_mem = 2 * (m) * element_size
        mem_bwd = activation_grad_mem + activation_buffer

        self.set_stats(name,
                       flops_fwd = flops_fwd,
                       use_tensor_cores = False,
                       mem_fwd = mem_fwd,
                       activation_buffer = activation_buffer,
                       weights_mem = 0,
                       weights_grad_mem = 0,
                       flops_bwd = flops_bwd,
                       mem_bwd = mem_bwd,
                       recompute = recompute,
                       remat = remat)

    def get_stats(self):
        self.compute_time()
        return self.stats

class Softmax(Estimates):
    def __init__(self, name, b, h, l1, l2,
                 parallelism={'dim1': 1,}, 
                 topology={'dim1': 'none'}, recompute=False, remat=False, system=None):
        """
        softmax  function estimates
        parameters: b: batch size
                    l1, l2: seq length
                    h: number of attention heads
                    element_size: in MB
                    
        layer arithmetic: 
            forward pass: 
                Y = softmax(X)
                (b,h,l,l) = (b,h,l,l)
            backward pass: 
                dL/dX = Y . dL/dY - Y . sum(Y . dL/dY, axis=-1)
                (b,h,l,l) = (b,h,l,l) . (b,h,l,l) - (b,h,l,l) . (b,h,l,1)
        
        comments: . is pointwise mult
        """

        super().__init__(system=system)
        
        flops_per_mult = 1 * self.flops_units
        flops_per_add = 1 * self.flops_units
        flops_per_exp = 1 * self.flops_units
        element_size = self.element_size

        m = parallelism['dim1']
        t = topology['t1']

        h_local = h // m

        # total flops
        flops_fwd = b * h_local * l1 * l2 * (flops_per_exp + flops_per_mult) + (b * h_local * l1 * (l2 - 1)) * flops_per_add
        
        #total mem
        activation_in_mem =  (b * h_local * l1 * l2) * element_size
        activation_out_mem = (b * h_local * l1 * l2) * element_size
        activation_buffer =  (b * h_local * l1 * l2) * element_size
        mem_fwd = activation_in_mem + activation_out_mem
        
        ####### backward pass #######
        flops_bwd =  (2 * b * h_local * l1 * l2) * flops_per_mult +  (b * h_local * l1 * (l2 - 1)) * flops_per_add + (b * h_local * l1 * l2) * flops_per_add
        activation_grad_mem = 2 * (b * h_local * l1 * l2) * element_size
        mem_bwd = activation_grad_mem + activation_buffer


        self.set_stats(name,
                       flops_fwd = flops_fwd,
                       use_tensor_cores = False,
                       mem_fwd = mem_fwd,
                       activation_buffer = activation_buffer,
                       weights_mem = 0,
                       weights_grad_mem = 0,
                       flops_bwd = flops_bwd,
                       mem_bwd = mem_bwd,
                       recompute = recompute,
                       remat = remat)

    def get_stats(self):
        self.compute_time()
        return self.stats

class Softmax2D(Estimates):
    def __init__(self, name, b, h, l1, l2,
                 parallelism={'dim1': 1, 'dim2': 1}, 
                 topology={'dim1': 'none', 'dim2': 'none'}, recompute=False, remat=False, system=None):
        """
        softmax  function estimates
        parameters: b: batch size
                    l1, l2: seq length
                    h: number of attention heads
                    element_size: in MB
                    
        layer arithmetic: 
            forward pass: 
                Y = softmax(X)
                (b,h,l,l) = (b,h,l,l)
            backward pass: 
                dL/dX = Y . dL/dY - Y . sum(Y . dL/dY, axis=-1)
                (b,h,l,l) = (b,h,l,l) . (b,h,l,l) - (b,h,l,l) . (b,h,l,1)
        
        comments: . is pointwise mult
        """

        super().__init__(system=system)
        
        flops_per_mult = 1 * self.flops_units
        flops_per_add = 1 * self.flops_units
        flops_per_exp = 1 * self.flops_units
        element_size = self.element_size

        m2 = parallelism['dim1']
        m1 = parallelism['dim2']
        t2 = topology['t1']
        t1 = topology['t2']
        m1_parallel = (m1 > 1)
        m2_parallel = (m2 > 1)

        l_local_1 = l1 // m2
        l_local_2 = l2 // m1

        # total flops
        flops_fwd = b * h * l_local_1 * l_local_2 * (flops_per_exp + flops_per_mult) + (b * h * l_local_1 * (l_local_2 - 1)) * flops_per_add
        
        #total mem
        activation_in_mem =  (b * h * l_local_1 * l_local_2) * element_size
        activation_out_mem = (b * h * l_local_1 * l_local_2) * element_size
        activation_buffer =  (b * h * l_local_1 * l_local_2) * element_size
        mem_fwd = activation_in_mem + activation_out_mem

        # sync/comm layers
        comm_fwd = m1_parallel * (b * h * l_local_1) * element_size
        comm_fwd_type = 'allreduce'
        comm_fwd_size = m1
        comm_fwd_topology = t1
        
        ####### backward pass #######
        flops_bwd =  (2 * b * h * l_local_1 * l_local_2) * flops_per_mult +  (b * h * l_local_1 * (l_local_2 - 1)) * flops_per_add + (b * h  * l_local_1 * l_local_2) * flops_per_add
        activation_grad_mem = 2 * (b * h * l_local_1 * l_local_2) * element_size
        mem_bwd = activation_grad_mem + activation_buffer

        # sync/comm layers
        comm_bwd = m1_parallel * (b * h * l_local_1) * element_size
        comm_bwd_type = 'allreduce'
        comm_bwd_size = m1
        comm_bwd_topology = t1

        self.set_stats(name,
                       flops_fwd = flops_fwd,
                       use_tensor_cores = False,
                       mem_fwd = mem_fwd,
                       activation_buffer = activation_buffer,
                       weights_mem = 0,
                       weights_grad_mem = 0,
                       comm_fwd = comm_fwd, 
                       comm_fwd_type = comm_fwd_type,
                       comm_fwd_size = comm_fwd_size,
                       comm_fwd_topology = comm_fwd_topology,
                       flops_bwd = flops_bwd,
                       mem_bwd = mem_bwd,
                       comm_bwd = comm_bwd, 
                       comm_bwd_type = comm_bwd_type,
                       comm_bwd_size = comm_bwd_size,
                       comm_bwd_topology = comm_bwd_topology,
                       recompute = recompute,
                       remat = remat)

    def get_stats(self):
        self.compute_time()
        return self.stats

class LayerNorm(Estimates):
    def __init__(self, name, b, l, e,
                 parallelism={'dim1': 1}, 
                 topology={'dim1': 'none'}, system=None):
        """
        layernorm layer estimates
        parameters: b: batch size
                    l: seq length
                    e: embedding dim/hidden dim
                    element_size: in MB

        layer arithmetic:
            define:
            forward pass:
                m = \avg X
                (b,l) = (b,l,e)
                s = \avg (X - m)^2
                (b,l) = (b,l,e)
                X_hat = (X - m) / s
                (b,l,e) = (b,l,e)
                Y = g * X_hat + b
                (b,l,e) = (e) * (b,l,e) + (e)

            backward pass:
                dL/dg = \sum_{b,l} dL/dY * X_hat
                (e) = \sum (b,l,e) * (b,l,e)
                dL/db = \sum_{b,l} dL/dY
                (e) = \sum (b,l,e)
                dL/dX = 1/s (dL/dY * g - 1/e dL/dY \dot g - 1/e (X_hat \dot (dL/dY * g)) * X_hat) [?]
                (b,l,e) = (b,l,e) * (e) - (b,l,e) . (e) - (b,l,e) . (b,l,e) * (e) * (b,l,e)


        comments: this ref is right: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
        """

        super().__init__(system=system)
        
        flops_per_mult = 1 * self.flops_units
        flops_per_add = 1 * self.flops_units
        element_size = self.element_size

        m = parallelism['dim1']
        t = topology['t1']

        l_local = l // m
        e_local = e # not sharded here

        ####### forward pass ########

        # total flops
        flops_fwd = 2 * b * l_local * e_local * flops_per_add # mean
        flops_fwd += 2 * b * l_local * e_local * flops_per_add + b * l_local * e_local * flops_per_mult # var
        flops_fwd += b * l_local * e_local * flops_per_add +  b * l_local * e_local * flops_per_mult # scale
        flops_fwd += b * l_local * e_local * flops_per_mult + b * l_local * e_local * flops_per_add

        #total mem
        activation_in_mem = (b * l_local * e_local) * element_size
        activation_in_mem += 2 * (b * l_local) * element_size # mean and std
        activation_out_mem = (b * l_local * e_local) * element_size
        activation_buffer = (b * l_local * e_local) * element_size
        weights_mem = 2 * e_local * element_size
        mem_fwd = activation_in_mem + activation_out_mem + weights_mem

        # sync/comm layers
        comm_fwd = (b * l * e) * element_size if m > 1 else 0
        comm_fwd_type = 'allgather'
        comm_fwd_size = m
        comm_fwd_topology = t

        ####### backward pass #######
        # little rough calcs, pretty sure some constant factor is off..
        flops_bwd = b * l_local * e_local * flops_per_mult + 2 * b * l_local * e_local * flops_per_add # g,b
        flops_bwd += (5 * b * l_local * e_local * flops_per_mult) + (4 * b * l_local * (e_local - 1) * flops_per_add)
        activation_grad_mem = 4 * (b * l_local * e_local) * element_size #dldy*3, dldx
        weights_grad_mem = 2 * e_local * element_size
        mem_bwd = activation_grad_mem + weights_grad_mem + activation_buffer

        # sync/comm layers
        comm_bwd = (b * l * e) * element_size if m > 1 else 0
        comm_bwd_type = 'reducescatter'
        comm_bwd_size = m
        comm_bwd_topology = t

        self.set_stats(name,
                       flops_fwd = flops_fwd,
                       use_tensor_cores = False,
                       mem_fwd = mem_fwd,
                       activation_buffer = activation_buffer,
                       weights_mem = weights_mem,
                       weights_grad_mem = weights_grad_mem,
                       comm_fwd = comm_fwd, 
                       comm_fwd_type = comm_fwd_type,
                       comm_fwd_size = comm_fwd_size,
                       comm_fwd_topology = comm_fwd_topology,
                       flops_bwd = flops_bwd,
                       mem_bwd = mem_bwd,
                       comm_bwd = comm_bwd, 
                       comm_bwd_type = comm_bwd_type,
                       comm_bwd_size = comm_bwd_size,
                       comm_bwd_topology = comm_bwd_topology)

    def get_stats(self):
        self.compute_time()
        return self.stats

class LayerNorm2D(Estimates):
    def __init__(self, name, b, l, e,
                 parallelism={'dim1': 1, 'dim2': 1}, 
                 topology={'dim1': 'none', 'dim2': 'none'}, system=None):
        """
        layernorm layer estimates
        parameters: b: batch size
                    l: seq length
                    e: embedding dim/hidden dim
                    element_size: in MB

        layer arithmetic:
            define:
            forward pass:
                m = \avg X
                (b,l) = (b,l,e)
                s = \avg (X - m)^2
                (b,l) = (b,l,e)
                X_hat = (X - m) / s
                (b,l,e) = (b,l,e)
                Y = g * X_hat + b
                (b,l,e) = (e) * (b,l,e) + (e)

            backward pass:
                dL/dg = \sum_{b,l} dL/dY * X_hat
                (e) = \sum (b,l,e) * (b,l,e)
                dL/db = \sum_{b,l} dL/dY
                (e) = \sum (b,l,e)
                dL/dX = 1/s (dL/dY * g - 1/e dL/dY \dot g - 1/e (X_hat \dot (dL/dY * g)) * X_hat) [?]
                (b,l,e) = (b,l,e) * (e) - (b,l,e) . (e) - (b,l,e) . (b,l,e) * (e) * (b,l,e)


        comments: this ref is right: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
        """

        super().__init__(system=system)
        
        flops_per_mult = 1 * self.flops_units
        flops_per_add = 1 * self.flops_units
        element_size = self.element_size

        m2 = parallelism['dim1']
        m1 = parallelism['dim2']
        t2 = topology['t1']
        t1 = topology['t2']
        m1_parallel = (m1 > 1)
        m2_parallel = (m2 > 1)

        l_local = l // m2
        e_local = l // m1

        ####### forward pass ########

        # total flops
        flops_fwd = 2 * b * l_local * e_local * flops_per_add # mean
        flops_fwd += 2 * b * l_local * e_local * flops_per_add + b * l_local * e_local * flops_per_mult # var
        flops_fwd += b * l_local * e_local * flops_per_add +  b * l_local * e_local * flops_per_mult # scale
        flops_fwd += b * l_local * e_local * flops_per_mult + b * l_local * e_local * flops_per_add

        #total mem
        activation_in_mem = (b * l_local * e_local) * element_size
        activation_in_mem += 2 * (b * l_local) * element_size # mean and std
        activation_out_mem = (b * l_local * e_local) * element_size
        activation_buffer = (b * l_local * e_local) * element_size
        weights_mem = 2 * e_local * element_size
        mem_fwd = activation_in_mem + activation_out_mem + weights_mem

        # sync/comm layers
        comm_fwd = m1_parallel * (2 * b * l_local) * element_size
        comm_fwd_type = 'allreduce'
        comm_fwd_size = m1
        comm_fwd_topology = t1

        ####### backward pass #######
        # little rough calcs, pretty sure some constant factor is off..
        flops_bwd = b * l_local * e_local * flops_per_mult + 2 * b * l_local * e_local * flops_per_add # g,b
        flops_bwd += (5 * b * l_local * e_local * flops_per_mult) + (4 * b * l_local * (e_local - 1) * flops_per_add)
        activation_grad_mem = 4 * (b * l_local * e_local) * element_size #dldy*3, dldx
        weights_grad_mem = 2 * e_local * element_size
        mem_bwd = activation_grad_mem + weights_grad_mem + activation_buffer

        # sync/comm layers
        comm_bwd = m1_parallel * (2 * e // m1 + 2 * b * l // m2) * element_size
        comm_bwd_type = 'allreduce'
        comm_bwd_size = m1
        comm_bwd_topology = t1

        self.set_stats(name,
                       flops_fwd = flops_fwd,
                       use_tensor_cores = False,
                       mem_fwd = mem_fwd,
                       activation_buffer = activation_buffer,
                       weights_mem = weights_mem,
                       weights_grad_mem = weights_grad_mem,
                       comm_fwd = comm_fwd, 
                       comm_fwd_type = comm_fwd_type,
                       comm_fwd_size = comm_fwd_size,
                       comm_fwd_topology = comm_fwd_topology,
                       flops_bwd = flops_bwd,
                       mem_bwd = mem_bwd,
                       comm_bwd = comm_bwd, 
                       comm_bwd_type = comm_bwd_type,
                       comm_bwd_size = comm_bwd_size,
                       comm_bwd_topology = comm_bwd_topology)

    def get_stats(self):
        self.compute_time()
        return self.stats

class Logits(Estimates):
    def __init__(self, name, b, l, q, h, 
                 parallelism={'dim1': 1}, 
                 topology={'dim1': 'none'}, recompute=False, system=None):
        """
        logits layer estimates
        parameters: b: batch size
                    l: seq length
                    h: number of attention heads
                    q: embedding dim/h
                    element_size: in MB

        tensor shapes: input tensor: (b,h,l,q)
                    output tensor: (b,h,l,q)

        layer arithmetic:
            forward pass:
                A = Q * K^T
                (b,h,l,l) = (b,h,l,q) * (b,h,q,l)
            backward pass:
                dL/dK = dL/dA^T * Q
                (b,h,l,q) = (b,h,l,l) * (b,h,l,q)
                dL/dQ = dL/dA * K
                (b,h,l,q) = (b,h,l,l) * (b,h,l,q)
        """

        super().__init__(system=system)
        
        flops_per_mult = 1 * self.flops_units
        flops_per_add = 1 * self.flops_units
        element_size = self.element_size

        m = parallelism['dim1']
        t = topology['t1']

        h_local = h // m

        ####### forward pass ########
        # total flops
        flops_fwd = b * h_local * l * l * (q * flops_per_mult + (q - 1) * flops_per_add)

        #total mem
        activation_in_mem =  (b * h_local * l * q) * element_size
        activation_in_mem += (b * h_local * l * q) * element_size
        activation_out_mem = (b * h_local * l * l) * element_size
        activation_buffer = 2 * (b * h_local * l * q) * element_size # Q and K
        mem_fwd = activation_in_mem + activation_out_mem

        ####### backward pass #######
        flops_bwd = 2 * b * h_local * l * q * (l * flops_per_mult + (l - 1) * flops_per_add)
        activation_grad_mem = 2 * (b * h_local * l * q) * element_size
        activation_grad_mem_att = 2 * (b * h_local * l * l) * element_size
        mem_bwd = activation_grad_mem + activation_grad_mem_att + activation_buffer

        self.set_stats(name,
                       flops_fwd = flops_fwd,
                       use_tensor_cores = True,
                       mem_fwd = mem_fwd,
                       activation_buffer = activation_buffer,
                       weights_mem = 0,
                       weights_grad_mem = 0,
                       flops_bwd = flops_bwd,
                       mem_bwd = mem_bwd,
                       recompute = recompute)

    def get_stats(self):
        self.compute_time()
        return self.stats

class Attend(Estimates):
    def __init__(self, name, b, l, q, h, 
                 parallelism={'dim1': 1}, 
                 topology={'dim1': 'none'}, remat=False, system=None):
        """
        attend layer estimates
        parameters: b: batch size
                    l: seq length
                    h: number of attention heads
                    q: embedding dim/h
                    element_size: in MB

        layer arithmetic: 
            forward pass :  
                Y = AV
                (b,h,l,q) = (b,h,l,l) * (b,h,l,q)
            backward pass: (L = loss)
                dL/dA = dL/dY * V^T
                (b,h,l,l) = (b,h,l,q) * (b,h,q,l)
                dL/dV = A^T * dL/dY
                (b,h,l,q) = (b,h,l,l) * (b,h,l,q) 
        """

        super().__init__(system=system)
        
        flops_per_mult = 1 * self.flops_units
        flops_per_add = 1 * self.flops_units
        element_size = self.element_size

        m = parallelism['dim1']
        t = topology['t1']

        h_local = h // m

        ####### forward pass ########
        flops_fwd = b * h_local * l * q * (l * flops_per_mult + (l - 1) * flops_per_add)
            
        # total mem
        activation_in_mem =  (b * h_local * l * l) * element_size
        activation_in_mem += (b * h_local * l * q) * element_size
        activation_out_mem = (b * h_local * l * q) * element_size
        activation_buffer1 =  (b * h_local * l * l) * element_size # store for bwd pass
        activation_buffer2 = (b * h_local * l * q) * element_size # store for bwd pass
        mem_fwd = activation_in_mem + activation_out_mem

        ####### backward pass #######
        flops_bwd = b * h_local * l * l * (q * flops_per_mult + (q - 1) * flops_per_add)
        flops_bwd += b * h_local * l * q * (l * flops_per_mult + (l - 1) * flops_per_add)
        activation_grad_mem = 3 * (b * h_local * l * q) * element_size
        activation_grad_mem_att = (b * h_local * l * l) * element_size
        mem_bwd = activation_grad_mem + activation_grad_mem_att + activation_buffer1 + activation_buffer2
        
        activation_buffer1 *= (not remat) # this buffer is released if remat
        activation_buffer = activation_buffer1 + activation_buffer2

        self.set_stats(name,
                       flops_fwd = flops_fwd,
                       use_tensor_cores = True,
                       mem_fwd = mem_fwd,
                       activation_buffer = activation_buffer,
                       weights_mem = 0,
                       weights_grad_mem = 0,
                       flops_bwd = flops_bwd,
                       mem_bwd = mem_bwd)

    def get_stats(self):
        self.compute_time()
        return self.stats

class LogitsSumma(Estimates):
    def __init__(self, name, b, l, q, h, 
                 parallelism={'dim1': 1, 'dim2': 1}, 
                 topology={'dim1': 'none', 'dim2': 'none'}, recompute=False, system=None):
        """
        logit layer estimates
        parameters: b: batch size
                    l: seq length
                    h: number of attention heads
                    q: embedding dim/h
                    element_size: in MB

        layer arithmetic:
            define: q = e/h
            forward pass:
                A = Q * K^T
                (b,h,l/m2,l/m1) = (b,h,l/m2,q/m1) * (b,h,l/m2,q/m1)^T
            backward pass:
                dL/dK = dL/dA^T * Q
                (b,h,l/m2,q/m1) = (b,h,l/m2,l/m1) * (b,h,l/m2,q/m1)
                dL/dQ = dL/dA * K
                (b,h,l/m2,q/m1) = (b,h,l/m2,l/m1) * (b,h,l/m2,q/m1)
        comments:
        """

        super().__init__(system=system)
        
        flops_per_mult = 1 * self.flops_units
        flops_per_add = 1 * self.flops_units
        element_size = self.element_size

        m2 = parallelism['dim1']
        m1 = parallelism['dim2']
        t2 = topology['t1']
        t1 = topology['t2']
        m1_parallel = (m1 > 1)
        m2_parallel = (m2 > 1)

        l_local_1 = l // m1
        l_local_2 = l // m2
        q_local = q // m1 

        # total flops
        # do nb times
        # broad cast bh,l/nb,q/m1 col-wise
        # temp matmul bh,l/m2,q/m1 * bh,q/m1,l/nb
        # reduce bh,l/m2,l/nb row-wise
        # careful, nb is arbitrarily chosen here
#        nb = l // 512 if m1 != m2 else m1
        nb = self.system['summa_nb']
        self.n_b = nb
        flops_fwd = b * h * l_local_2 * l * (q_local * flops_per_mult + (q_local - 1) * flops_per_add)

        #total mem
        activation_buffer = 2 * (b * h * l_local_2 * q_local) * element_size # Q and K
        weights_mem = 0
        mem_fwd = (b * h * l * q_local + b * h * l_local_2 * q_local * nb + b * h * l_local_2 * l) * element_size

        # sync/comm layers
        comm_fwd = [m2_parallel * (b * h * l * q_local) * element_size,
                     m1_parallel * (b * h * l_local_2 * l) * element_size] 
        comm_fwd_type = ["broadcast", "reduce"]
        comm_fwd_size = [m2, m1]
        comm_fwd_topology = [t2, t1]

        ####### backward pass #######
        # dl/dq:  broadcast bh,l/m2,l/nb rowwise and bh,l/nb,q/m1 colwise, admm locally
        flops_bwd = b * h * l_local_2 * q_local * (l * flops_per_mult + (l - 1) * flops_per_add)
        # dl/dk:  broadcast bh,l/m2,l/nb rowwise 
        flops_bwd += b * h * l * q_local * (l_local_2 * flops_per_mult + (l_local_2 - 1) * flops_per_add)
        mem_bwd = (2 * b * h * l_local_2 * l + 2 * b * h * l * q_local + 2 * b * h * l_local_2 * q_local * nb) * element_size

        # sync/comm layers
        # broadcast bh,l/m2,l/nb rowwise and bh,l/nb,q/m1 colwise, admm locally
        comm_bwd = [m1_parallel * (b * h * l_local_2 * l) * element_size,
                    m2_parallel * (b * h * l * q_local) * element_size,
                    m1_parallel * (b * h * l_local_2 * l) * element_size,
                    m2_parallel * (b * h * l * q_local) * element_size]
        comm_bwd_type = ["broadcast", "broadcast", "broadcast", "reduce"]
        comm_bwd_size = [m1, m2, m1, m2]
        comm_bwd_topology = [t1, t2, t1, t2]

        self.set_stats(name,
                       flops_fwd = flops_fwd,
                       use_tensor_cores = True,
                       mem_fwd = mem_fwd,
                       activation_buffer = activation_buffer,
                       weights_mem = 0,
                       weights_grad_mem = 0,
                       comm_fwd = comm_fwd, 
                       comm_fwd_type = comm_fwd_type,
                       comm_fwd_size = comm_fwd_size,
                       comm_fwd_topology = comm_fwd_topology,
                       flops_bwd = flops_bwd,
                       mem_bwd = mem_bwd,
                       comm_bwd = comm_bwd, 
                       comm_bwd_type = comm_bwd_type,
                       comm_bwd_size = comm_bwd_size,
                       comm_bwd_topology = comm_bwd_topology,
                       recompute = recompute)


    def compute_time_summa(self, flops, mem, comms, comm_sizes, comm_types, comm_tops, verbose=False):
        t_comp = self.get_time_flops(flops)
        t_mem  = self.get_time_mem(mem)
        t_compute = max(t_comp, t_mem)
        intensity = t_comp / t_mem

        t_for_one_comm_set = 0 # typically two broadcasts or one broadcast, one reduce

        for c, comm in enumerate(comms):
            comm_size = comm_sizes[c]
            comm_type = comm_types[c]
            comm_top = comm_tops[c]
#            t_for_nb_comms = self.get_time_comm(comm, comm_size, comm_type, comm_top)  
#            t_for_one_comm_set += t_for_nb_comms / self.n_b 

            # for any comm, divide by n_b for each itr (do it this way because of the comm latencies)
            t_for_one_comm_set += self.get_time_comm(comm / self.n_b, comm_size, comm_type, comm_top)  


        # overlap some comms with compute
        t_comm = t_for_one_comm_set + max(t_for_one_comm_set * self.n_b - t_compute, 0)
        t = t_compute + t_comm
        if verbose:
            print(t_compute, t_comp, t_mem, flops, mem, t_comm, t_for_one_comm_set, t_for_one_comm_set * self.n_b)
        return t, t_comm, intensity

    def compute_time(self): # need to overwrite to implement pipelined/overlapping comms
        # forward time
        self.stats['t_fwd'], self.stats['t_fwd_comm'], self.stats['intensity_fwd'] = self.compute_time_summa(self.flops_fwd,
                                                                                                             self.mem_fwd,
                                                                                                             self.comm_fwd,
                                                                                                             self.comm_fwd_size,
                                                                                                             self.comm_fwd_type,
                                                                                                             self.comm_fwd_topology)
        self.stats['t_bwd'], self.stats['t_bwd_comm'], self.stats['intensity_bwd'] = self.compute_time_summa(self.flops_bwd,
                                                                                                             self.mem_bwd,
                                                                                                             self.comm_bwd,
                                                                                                             self.comm_bwd_size,
                                                                                                             self.comm_bwd_type,
                                                                                                             self.comm_bwd_topology)
        self.stats['t'] = self.stats['t_fwd'] + self.stats['t_bwd']


    def get_stats(self):
        self.compute_time()
        return self.stats

class AttendSumma(Estimates):
    def __init__(self, name, b, l, q, h, 
                 parallelism={'dim1': 1, 'dim2': 1}, 
                 topology={'dim1': 'none', 'dim2': 'none'}, remat=False, system=None):
        """
        attend layer estimates
        parameters: b: batch size
                    l: seq length
                    h: number of attention heads
                    q: embedding dim/h
                    element_size: in MB

                    
        layer arithmetic: 
            forward pass :  
                Y = AV
                (b,h,l,q) = (b,h,l,l) * (b,h,l,q)
            backward pass: (L = loss)
                dL/dA = dL/dY * V^T
                (b,h,l,l) = (b,h,l,q) * (b,h,q,l)
                dL/dV = A^T * dL/dY
                (b,h,l,q) = (b,h,l,l) * (b,h,l,q) 
        comments:
        """

        super().__init__(system=system)
        
        flops_per_mult = 1 * self.flops_units
        flops_per_add = 1 * self.flops_units
        element_size = self.element_size

        m2 = parallelism['dim1']
        m1 = parallelism['dim2']
        t2 = topology['t1']
        t1 = topology['t2']
        m1_parallel = (m1 > 1)
        m2_parallel = (m2 > 1)

        l_local_1 = l // m1
        l_local_2 = l // m2
        q_local = q // m1 

        # total flops
        flops_fwd = b * h * l_local_2 * q_local * (l * flops_per_mult + l * flops_per_add) # l outer products and l addns in total
        
        # total mem
        activation_buffer = (b * h * l_local_2 * l_local_1 * (not remat) + b * h * l_local_2 * q_local) * element_size # store for bwd pass
        weights_mem = 0 
        # careful, nb is arbitrarily chosen here
        nb = self.system['summa_nb']
#        nb = l // 512 if m1 != m2 else m1
        self.n_b = nb
        mem_fwd  = (b * h * l_local_2 * l + b * h * l * q_local + b * h * l_local_2 * q_local * nb) * element_size

        # sync/comm layers
        comm_fwd = [m1_parallel * (b * h * l_local_2 * l) * element_size,
                    m2_parallel * (b * h * l * q_local) * element_size] # both inputs are broadcasted in diff dims
        comm_fwd_type = ["broadcast", "broadcast"]
        comm_fwd_size = [m1, m2]
        comm_fwd_topology = [t1, t2]

        ####### backward pass #######
        # broadcast  bh,l/nb,q/m1 colwise, bh,l/m2,q/m1 * bh,q/m1,l/nb
        agrad_flops = b * h * l_local_2 * l * (q_local * flops_per_mult + (q_local - 1) * flops_per_add) 
        # broadcast  bh,l/m2,l/nb rowsie, bh,l/nb,l/m2 * bh,l/m2,q/m1
        vgrad_flops = b * h * l * q_local * (l_local_2 * flops_per_mult + (l_local_2 - 1) * flops_per_add)
        flops_bwd = agrad_flops + vgrad_flops
        
        mem_bwd = b * h * (l_local_2 * q_local * nb + q_local * l + l_local_2 * l) * element_size
        mem_bwd += b * h * (l_local_2 * q_local * nb + l * l_local_2 + l * q_local) * element_size

        # sync/comm layers
        # broadcast  bh,l/nb,q/m1 colwise, bh,l/m2,q/m1 * bh,q/m1,l/nb
        # broadcast  bh,l/m2,l/nb rowsie, bh,l/nb,l/m2 * bh,l/m2,q/m1
        comm_bwd = [m2_parallel * (b * h * l * q_local) * element_size, # broadcast
                    m1_parallel * (b *  h * l_local_2 * l) * element_size, # reduce result
                    m1_parallel * (b * h * l_local_2 * l) * element_size, # broadcast
                    m2_parallel * (b * h * l * q_local) * element_size] # reduce result
        comm_bwd_type = ["broadcast", "reduce", "broadcast", "reduce"]
        comm_bwd_size = [m2, m1, m1, m2]
        comm_bwd_topology = [t2, t1, t1, t2]

        self.set_stats(name,
                       flops_fwd = flops_fwd,
                       use_tensor_cores = True,
                       mem_fwd = mem_fwd,
                       activation_buffer = activation_buffer,
                       weights_mem = 0,
                       weights_grad_mem = 0,
                       comm_fwd = comm_fwd, 
                       comm_fwd_type = comm_fwd_type,
                       comm_fwd_size = comm_fwd_size,
                       comm_fwd_topology = comm_fwd_topology,
                       flops_bwd = flops_bwd,
                       mem_bwd = mem_bwd,
                       comm_bwd = comm_bwd, 
                       comm_bwd_type = comm_bwd_type,
                       comm_bwd_size = comm_bwd_size,
                       comm_bwd_topology = comm_bwd_topology)


    def compute_time_summa(self, flops, mem, comms, comm_sizes, comm_types, comm_tops, verbose=False):
        t_comp = self.get_time_flops(flops)
        t_mem  = self.get_time_mem(mem)
        t_compute = max(t_comp, t_mem)
        intensity = t_comp / t_mem

        t_for_one_comm_set = 0 # typically two broadcasts or one broadcast, one reduce

        for c, comm in enumerate(comms):
            comm_size = comm_sizes[c]
            comm_type = comm_types[c]
            comm_top = comm_tops[c]
#            t_for_nb_comms = self.get_time_comm(comm, comm_size, comm_type, comm_top)  
#            t_for_one_comm_set += t_for_nb_comms / self.n_b 

            # for any comm, divide by n_b for each itr (do it this way because of the comm latencies)
            t_for_one_comm_set += self.get_time_comm(comm / self.n_b, comm_size, comm_type, comm_top)  


        # overlap some comms with compute
        t_comm = t_for_one_comm_set + max(t_for_one_comm_set * self.n_b - t_compute, 0)
        t = t_compute + t_comm
        if verbose:
            print(t_compute, t_comp, t_mem, flops, mem, t_comm, t_for_one_comm_set, t_for_one_comm_set * self.n_b)
        return t, t_comm, intensity

    def compute_time(self): # need to overwrite to implement pipelined/overlapping comms
        # forward time
        self.stats['t_fwd'], self.stats['t_fwd_comm'], self.stats['intensity_fwd'] = self.compute_time_summa(self.flops_fwd,
                                                                                                             self.mem_fwd,
                                                                                                             self.comm_fwd,
                                                                                                             self.comm_fwd_size,
                                                                                                             self.comm_fwd_type,
                                                                                                             self.comm_fwd_topology)
        self.stats['t_bwd'], self.stats['t_bwd_comm'], self.stats['intensity_bwd'] = self.compute_time_summa(self.flops_bwd,
                                                                                                             self.mem_bwd,
                                                                                                             self.comm_bwd,
                                                                                                             self.comm_bwd_size,
                                                                                                             self.comm_bwd_type,
                                                                                                             self.comm_bwd_topology)
        self.stats['t'] = self.stats['t_fwd'] + self.stats['t_bwd']


    def get_stats(self):
        self.compute_time()
        return self.stats

class LogitsSeqp(Estimates):
    def __init__(self, name, b, l, q, h, 
                 parallelism={'dim1': 1, 'dim2': 1}, 
                 topology={'dim1': 'none', 'dim2': 'none'}, recompute=False, system=None):
        """
        logit layer estimates
        parameters: b: batch size
                    l: seq length
                    h: number of attention heads
                    q: embedding dim/h
                    element_size: in MB

        forward pass:
             A = Q * K^T
             (b,h/m1,l/m2,l) = (b,h/m1,l/m2,q) * (b,h/m1,q,l/m2)
        backward pass:
             dL/dK = dL/dA^T * Q
             (b,h/m1,l/m2,q) = (b,h/m1,l,l/m2) * (b,h/m1,l/m2,q)
             dL/dQ = dL/dA * K
             (b,h/m1,l/m2,q) = (b,h/m1,l/m2,l) * (b,h/m1,l/m2,q)
        comments:
        """

        super().__init__(system=system)
        
        flops_per_mult = 1 * self.flops_units
        flops_per_add = 1 * self.flops_units
        element_size = self.element_size

        m2 = parallelism['dim1']
        m1 = parallelism['dim2']
        t2 = topology['t1']
        t1 = topology['t2']
        m1_parallel = (m1 > 1)
        m2_parallel = (m2 > 1)

        l_local = l // m2
        h_local = h // m1

        # total flops
        flops_fwd = b * h_local * l_local * l * (q * flops_per_mult + (q - 1) * flops_per_add)

        #total mem
        activation_in_mem = (b * h_local * l_local * q) * element_size
        activation_in_mem += (b * h_local * l * q) * element_size
        activation_out_mem = (b * h_local * l_local * l) * element_size
        activation_buffer = 2 * (b * h_local * l_local * q) * element_size # Q and K
        mem_fwd = activation_in_mem + activation_out_mem

        # sync/comm layers
        comm_fwd = m2_parallel * (b * h_local * l * q) * element_size
        comm_fwd_type = "allgather"
        comm_fwd_size = m2
        comm_fwd_topology = t2

        ####### backward pass #######
        flops_bwd = b * h_local * l_local * q * (l * flops_per_mult + (l - 1) * flops_per_add)
        flops_bwd += b * h_local * l * q * (l_local * flops_per_mult + (l_local - 1) * flops_per_add)
        activation_grad_mem = (b * h_local * l_local * q + b * h_local * l * q) * element_size
        activation_grad_mem_att = 2 * (b * h_local * l_local * l) * element_size
        mem_bwd = activation_grad_mem + activation_grad_mem_att + activation_buffer

        comm_bwd = [m2_parallel * (b * h_local * l * q) * element_size,
                    m2_parallel * (b * h_local * l * q) * element_size]
        comm_bwd_type = ["allgather", "reducescatter"]
        comm_bwd_size = [m2, m2]
        comm_bwd_topology = [t2, t2] 

        self.set_stats(name,
                       flops_fwd = flops_fwd,
                       use_tensor_cores = True,
                       mem_fwd = mem_fwd,
                       activation_buffer = activation_buffer,
                       weights_mem = 0,
                       weights_grad_mem = 0,
                       comm_fwd = comm_fwd, 
                       comm_fwd_type = comm_fwd_type,
                       comm_fwd_size = comm_fwd_size,
                       comm_fwd_topology = comm_fwd_topology,
                       flops_bwd = flops_bwd,
                       mem_bwd = mem_bwd,
                       comm_bwd = comm_bwd, 
                       comm_bwd_type = comm_bwd_type,
                       comm_bwd_size = comm_bwd_size,
                       comm_bwd_topology = comm_bwd_topology,
                       recompute = recompute)


    def compute_times(self, flops, mem, comms, comm_sizes, comm_types, comm_tops):
        t_comp = self.get_time_flops(flops)
        t_mem  = self.get_time_mem(mem)
        t_compute = max(t_comp, t_mem)
        intensity = t_comp / t_mem

        t_comm = 0
        if isinstance(comms, list):
            for c, comm in enumerate(comms):
                comm_size = comm_sizes[c]
                comm_type = comm_types[c]
                comm_top = comm_tops[c]
                t_comm += self.get_time_comm(comm, comm_size, comm_type, comm_top)  
        else:
            t_comm = self.get_time_comm(comms, comm_sizes, comm_types, comm_tops)  

        t = t_compute + t_comm
        return t, t_comm, intensity

    def compute_time(self): # overwrite due to diff comm patterns
        # forward time
        self.stats['t_fwd'], self.stats['t_fwd_comm'], self.stats['intensity_fwd'] = self.compute_times(self.flops_fwd,
                                                                                                             self.mem_fwd,
                                                                                                             self.comm_fwd,
                                                                                                             self.comm_fwd_size,
                                                                                                             self.comm_fwd_type,
                                                                                                             self.comm_fwd_topology)
        self.stats['t_bwd'], self.stats['t_bwd_comm'], self.stats['intensity_bwd'] = self.compute_times(self.flops_bwd,
                                                                                                             self.mem_bwd,
                                                                                                             self.comm_bwd,
                                                                                                             self.comm_bwd_size,
                                                                                                             self.comm_bwd_type,
                                                                                                             self.comm_bwd_topology)
        self.stats['t'] = self.stats['t_fwd'] + self.stats['t_bwd']


    def get_stats(self):
        self.compute_time()
        return self.stats

class AttendSeqp(Estimates):
    def __init__(self, name, b, l, q, h, 
                 parallelism={'dim1': 1, 'dim2': 1}, 
                 topology={'dim1': 'none', 'dim2': 'none'}, remat=False, system=None):
        """
        attend layer estimates
        parameters: b: batch size
                    l: seq length
                    h: number of attention heads
                    q: embedding dim/h
                    element_size: in MB

                    
        layer arithmetic: 
            assume h = h/m1
            forward pass :  
                Y = AV
                (b,h,l/m2,q) = (b,h,l/m2,l) * (b,h,l/m2,q)
            backward pass: (L = loss)
                dL/dA = dL/dY * V^T
                (b,h,l/m2,l) = (b,h,l/m2,q) * (b,h,q,l/m2)
                dL/dV = A^T * dL/dY
                (b,h,l/m2,q) = (b,h,l,l/m2) * (b,h,l/m2,q) 
        comments:
        """

        super().__init__(system=system)
        
        flops_per_mult = 1 * self.flops_units
        flops_per_add = 1 * self.flops_units
        element_size = self.element_size

        m2 = parallelism['dim1']
        m1 = parallelism['dim2']
        t2 = topology['t1']
        t1 = topology['t2']
        m1_parallel = (m1 > 1)
        m2_parallel = (m2 > 1)

        l_local = l // m2
        h_local = h // m1
        flops_fwd = b * h_local * l_local * q * (l * flops_per_mult + (l - 1) * flops_per_add)
            
        # total mem
        activation_in_mem = (b * h_local * l_local * l) * element_size
        activation_in_mem += (b * h_local * l * q) * element_size
        activation_out_mem = (b * h_local * l_local * q) * element_size
        activation_buffer = (b * h_local * l_local * l) * element_size * (not remat)# store for bwd pass
        activation_buffer += (b * h_local * l_local * q) * element_size # store for bwd pass
        mem_fwd = activation_in_mem + activation_out_mem

        # sync/comm layers
        comm_fwd = m2_parallel * (b * h_local * l * q) * element_size
        comm_fwd_type = "allgather"
        comm_fwd_size = m2
        comm_fwd_topology = t2

        flops_bwd = b * h_local * l_local * l * (q * flops_per_mult + (q - 1) * flops_per_add)
        flops_bwd += b * h_local * l * q * (l_local * flops_per_mult + (l_local - 1) * flops_per_add)
        activation_grad_mem = 2 * (b * h_local * l_local * q) * element_size + (b * h_local * l_local * q) * element_size
        activation_grad_mem_att = (b * h_local * l_local * l) * element_size
        buffers_with_gather = b * h_local * (l_local * l + l * q) * element_size # attention and values (this is allgathered)
        mem_bwd = activation_grad_mem + activation_grad_mem_att  + buffers_with_gather

        comm_bwd = [m2_parallel * (b * h_local * l * q) * element_size,
                    m2_parallel * (b * h_local * l * q) * element_size]
        comm_bwd_type = ["allgather", "reducescatter"] # reducescatter for dl(dKV)
        comm_bwd_size = [m2, m2]
        comm_bwd_topology = [t2, t2] 

        self.set_stats(name,
                       flops_fwd = flops_fwd,
                       use_tensor_cores = True,
                       mem_fwd = mem_fwd,
                       activation_buffer = activation_buffer,
                       weights_mem = 0,
                       weights_grad_mem = 0,
                       comm_fwd = comm_fwd, 
                       comm_fwd_type = comm_fwd_type,
                       comm_fwd_size = comm_fwd_size,
                       comm_fwd_topology = comm_fwd_topology,
                       flops_bwd = flops_bwd,
                       mem_bwd = mem_bwd,
                       comm_bwd = comm_bwd, 
                       comm_bwd_type = comm_bwd_type,
                       comm_bwd_size = comm_bwd_size,
                       comm_bwd_topology = comm_bwd_topology)


    def compute_times(self, flops, mem, comms, comm_sizes, comm_types, comm_tops):
        t_comp = self.get_time_flops(flops)
        t_mem  = self.get_time_mem(mem)
        t_compute = max(t_comp, t_mem)
        intensity = t_comp / t_mem

        t_comm = 0
        if isinstance(comms, list):
            for c, comm in enumerate(comms):
                comm_size = comm_sizes[c]
                comm_type = comm_types[c]
                comm_top = comm_tops[c]
                t_comm += self.get_time_comm(comm, comm_size, comm_type, comm_top)  
        else:
            t_comm = self.get_time_comm(comms, comm_sizes, comm_types, comm_tops)  

        t = t_compute + t_comm
        return t, t_comm, intensity

    def compute_time(self): # overwrite due to diff comm patterns
        # forward time
        self.stats['t_fwd'], self.stats['t_fwd_comm'], self.stats['intensity_fwd'] = self.compute_times(self.flops_fwd,
                                                                                                             self.mem_fwd,
                                                                                                             self.comm_fwd,
                                                                                                             self.comm_fwd_size,
                                                                                                             self.comm_fwd_type,
                                                                                                             self.comm_fwd_topology)
        self.stats['t_bwd'], self.stats['t_bwd_comm'], self.stats['intensity_bwd'] = self.compute_times(self.flops_bwd,
                                                                                                             self.mem_bwd,
                                                                                                             self.comm_bwd,
                                                                                                             self.comm_bwd_size,
                                                                                                             self.comm_bwd_type,
                                                                                                             self.comm_bwd_topology)
        self.stats['t'] = self.stats['t_fwd'] + self.stats['t_bwd']


    def get_stats(self):
        self.compute_time()
        return self.stats

class FusedLA(Estimates):
    def __init__(self, name, b, l, q, h, 
                 parallelism={'dim1': 1}, 
                 topology={'dim1': 'none'}, system=None):
        """
        Fused LA layer estimates
        parameters: b: batch size
                    l: seq length
                    h: number of attention heads
                    q: embedding dim/h
                    element_size: in MB

        layer arithmetic:
            forward pass:
                A = Q * K^T
                (b,h,l,l) = (b,h,l,q) * (b,h,q,l)
                A = softmax(A)
                (b,h,l,l) = (b,h,l,l)
                A = random_mask(A)
                (b,hl,l) = (b,hl,l)
                Y = AV
                (b,h,l,q) = (b,h,l,l) * (b,h,l,q)
            backward pass:
                dL/dK = dL/dA^T * Q
                (b,h,l,q) = (b,h,l,l) * (b,h,l,q)
                dL/dQ = dL/dA * K
                (b,h,l,q) = (b,h,l,l) * (b,h,l,q)

                dL/dX = Y . dL/dY - Y . sum(Y . dL/dY, axis=-1)
                (b,h,l,l) = (b,h,l,l) . (b,h,l,l) - (b,h,l,l) . (b,h,l,1)

                dl/dX = dl/dY * random_mask

                dL/dA = dL/dY * V^T
                (b,h,l,l) = (b,h,l,q) * (b,h,q,l)
                dL/dV = A^T * dL/dY
                (b,h,l,q) = (b,h,l,l) * (b,h,l,q) 
        """

        super().__init__(system=system)

        # ugly fix: some ops here wont use tensor cores so correct them
        tensor_core_factor = self.system['matrix_flops_fp16'] / self.system['vector_flops_fp16']
        
        flops_per_mult = 1 * self.flops_units
        flops_per_add = 1 * self.flops_units
        flops_per_exp = 1 * self.flops_units
        element_size = self.element_size

        m = parallelism['dim1']
        t = topology['t1']

        h_local = h // m

        # total flops
        # logits
        flops_fwd  = b * h_local * l * l * (q * flops_per_mult + (q - 1) * flops_per_add)
        # softmax
        flops_softmax = b * h_local * l * l * (flops_per_exp + flops_per_mult) + (b * h_local * l * (l - 1)) * flops_per_add
        flops_softmax *= tensor_core_factor # wont use tensor cores
        flops_fwd += flops_softmax
        # dropout
        flops_dpr = (b * h_local * l * l) * flops_per_mult
        flops_dpr *= tensor_core_factor # wont use tensor cores
        flops_fwd += flops_dpr
        # attend
        flops_fwd += b * h_local * l * q * (l * flops_per_mult + (l - 1) * flops_per_add)

        #total mem
        activation_in_mem = (b * h_local * l * q) * element_size # Q
        activation_in_mem += 2 * (b * h_local * l * q) * element_size # K and V
        activation_in_mem += (b * h_local * l) * element_size # stats for softmax
        activation_out_mem = (b * h_local * l * q) * element_size # result

        activation_buffer = 3 * (b * h_local * l * q) * element_size # q, k, v 
        activation_buffer += (b * h_local * l) * element_size # random number generator states (droppout mask is not stored); dont know if this is float
        activation_buffer += (b * h_local * l) * element_size # stats for softmax
        
        # TODO: in software this might be stored (even though the next layer will have it: might need to revisit if not
        activation_buffer += (b * h_local * l * q) * element_size # result for flashattn bwd
        weights_mem = 0
        mem_fwd = activation_in_mem + activation_out_mem + weights_mem

        ####### backward pass #######
        # logits
        flops_bwd = 2 * b * h_local * l * q * (l * flops_per_mult + (l - 1) * flops_per_add)
        # softmax
        flops_softmax = (2 * b * h_local * l * l) * flops_per_mult +  (b * h_local * l * (l - 1)) * flops_per_add + (b * h_local * l * l) * flops_per_add
        flops_softmax *= tensor_core_factor # wont use tensor cores
        flops_bwd += flops_softmax
        # dropout
        flops_dpr = b * h_local * l * l  * flops_per_mult
        flops_dpr *= tensor_core_factor # wont use tensor cores
        flops_bwd += flops_dpr
        # attend
        flops_bwd += b * h_local * l * l * (q * flops_per_mult + (q - 1) * flops_per_add)
        flops_bwd += b * h_local * l * q * (l * flops_per_mult + (l - 1) * flops_per_add)

        # extra fwd flops since attn is remat
        # logit
        flops_bwd +=  b * h_local * l * l * (q * flops_per_mult + (q - 1) * flops_per_add)
        # softmax
        flops_softmax = b * h_local * l * l * (flops_per_exp + flops_per_mult) + (b * h_local * l * (l - 1)) * flops_per_add
        flops_softmax *= tensor_core_factor # wont use tensor cores
        flops_bwd += flops_softmax
        # dropout
        flops_dpr = (b * h_local * l * l) * flops_per_mult
        flops_dpr *= tensor_core_factor # wont use tensor cores
        flops_bwd += flops_dpr

        # mem
        activation_grad_mem = 4 * (b * h_local * l * q) * element_size # dq, dk, dv, dresult
        mem_bwd = activation_grad_mem + activation_buffer

        self.set_stats(name,
                       flops_fwd = flops_fwd,
                       use_tensor_cores = True,
                       mem_fwd = mem_fwd,
                       activation_buffer = activation_buffer,
                       weights_mem = 0,
                       weights_grad_mem = 0,
                       flops_bwd = flops_bwd,
                       mem_bwd = mem_bwd)

    def get_stats(self):
        self.compute_time()
        return self.stats

class FusedLASeqp(Estimates):
    def __init__(self, name, b, l, q, h, 
                 parallelism={'dim1': 1, 'dim2': 1}, 
                 topology={'dim1': 'none', 'dim2': 'none'}, system=None):
        """
        Fused LA layer estimates
        parameters: b: batch size
                    l: seq length
                    h: number of attention heads
                    q: embedding dim/h
                    element_size: in MB

        layer arithmetic:
            forward pass:
                A = Q * K^T
                (b,h/m1,l/m2,l) = (b,h/m1,l/m2,q) * (b,h/m1,q,l/m2)
                A = softmax(A)
                (b,h/m1,l/m2,l) = (b,h/m1,l/m2,l)
                A = random_mask(A)
                Y = AV
                (b,h,l/m2,q) = (b,h,l/m2,l) * (b,h,l/m2,q)
            backward pass:
                dL/dK = dL/dA^T * Q
                (b,h/m1,l/m2,q) = (b,h/m1,l,l/m2) * (b,h/m1,l/m2,q)
                dL/dQ = dL/dA * K
                (b,h/m1,l/m2,q) = (b,h/m1,l/m2,l) * (b,h/m1,l/m2,q)

                dL/dX = Y . dL/dY - Y . sum(Y . dL/dY, axis=-1)
                (b,h/m1,l/m2,l) = ... 

                dl/dX = dl/dY * random_mask

                dL/dA = dL/dY * V^T
                (b,h,l/m2,l) = (b,h,l/m2,q) * (b,h,q,l/m2)
                dL/dV = A^T * dL/dY
                (b,h,l/m2,q) = (b,h,l,l/m2) * (b,h,l/m2,q) 
        """

        super().__init__(system=system)

        # ugly fix: some ops here wont use tensor cores so correct them
        tensor_core_factor = self.system['matrix_flops_fp16'] / self.system['vector_flops_fp16']
        
        flops_per_mult = 1 * self.flops_units
        flops_per_add = 1 * self.flops_units
        flops_per_exp = 1 * self.flops_units
        element_size = self.element_size

        m2 = parallelism['dim1']
        m1 = parallelism['dim2']
        t2 = topology['t1']
        t1 = topology['t2']
        m1_parallel = (m1 > 1)
        m2_parallel = (m2 > 1)

        l_local = l // m2
        h_local = h // m1

        # total flops
        # logits
        flops_fwd  = b * h_local * l_local * l * (q * flops_per_mult + (q - 1) * flops_per_add)
        # softmax
        flops_softmax = b * h_local * l_local * l * (flops_per_exp + flops_per_mult) + (b * h_local * l_local * (l - 1)) * flops_per_add
        flops_softmax *= tensor_core_factor # wont use tensor cores
        flops_fwd += flops_softmax
        # dropout
        flops_dpr = (b * h_local * l_local * l) * flops_per_mult
        flops_dpr *= tensor_core_factor # wont use tensor cores
        flops_fwd += flops_dpr
        # attend
        flops_fwd += b * h_local * l_local * q * (l * flops_per_mult + (l - 1) * flops_per_add)

        #total mem
        activation_in_mem = (b * h_local * l_local * q) * element_size # Q
        activation_in_mem += 2 * (b * h_local * l * q) * element_size # K and V, have to allgather KV in fwd pass
        activation_in_mem += (b * h_local * l_local) * element_size # stats for softmax
        activation_out_mem = (b * h_local * l_local * q) * element_size # result

        activation_buffer = 3 * (b * h_local * l_local * q) * element_size # q, k, v 
        activation_buffer += (b * h_local * l_local) * element_size # random number generator states (droppout mask is not stored); dont know if this is float
        activation_buffer += (b * h_local * l_local) * element_size # stats for softmax
        
        # TODO: in software this might be stored (even though the next layer will have it: might need to revisit if not
        activation_buffer += (b * h_local * l_local * q) * element_size # result for flashattn bwd
        weights_mem = 0
        mem_fwd = activation_in_mem + activation_out_mem + weights_mem

        # sync/comm layers
        comm_fwd = m2_parallel * (2 * b * h_local * l * q) * element_size # allgather KV
        comm_fwd_type = "allgather"
        comm_fwd_size = m2
        comm_fwd_topology = t2

        ####### backward pass #######
        ####### backward pass #######
        # logits
        flops_bwd = b * h_local * l_local * q * (l * flops_per_mult + (l - 1) * flops_per_add)
        flops_bwd += b * h_local * l * q * (l_local * flops_per_mult + (l_local - 1) * flops_per_add)
        # softmax
        flops_softmax = (2 * b * h_local * l_local * l) * flops_per_mult +  (b * h_local * l_local * (l - 1)) * flops_per_add + (b * h_local * l_local * l) * flops_per_add
        flops_softmax *= tensor_core_factor # wont use tensor cores
        flops_bwd += flops_softmax
        # dropout
        flops_dpr = b * h_local * l_local * l  * flops_per_mult
        flops_dpr *= tensor_core_factor # wont use tensor cores
        flops_bwd += flops_dpr
        # attend
        flops_bwd += b * h_local * l_local * l * (q * flops_per_mult + (q - 1) * flops_per_add)
        flops_bwd += b * h_local * l * q * (l_local * flops_per_mult + (l_local - 1) * flops_per_add)

        # extra fwd flops since attn is remat
        # logits
        flops_bwd += b * h_local * l_local * l * (q * flops_per_mult + (q - 1) * flops_per_add)
        # softmax
        flops_softmax = b * h_local * l_local * l * (flops_per_exp + flops_per_mult) + (b * h_local * l_local * (l - 1)) * flops_per_add
        flops_softmax *= tensor_core_factor # wont use tensor cores
        flops_bwd += flops_softmax
        # dropout
        flops_dpr = (b * h_local * l_local * l) * flops_per_mult
        flops_dpr *= tensor_core_factor # wont use tensor cores
        flops_bwd += flops_dpr

        # mem
        activation_grad_mem = 2 * (b * h_local * l * q) * element_size # dk, dv (will be reducescattered)
        activation_grad_mem += 2 * (b * h_local * l_local * q) * element_size # dq, dresult
        mem_bwd = activation_grad_mem + activation_buffer

        comm_bwd = [m2_parallel * (2 * b * h_local * l * q) * element_size,
                    m2_parallel * (2 * b * h_local * l * q) * element_size]
        comm_bwd_type = ["allgather", "reducescatter"] # reducescatter for dl(dKV)
        comm_bwd_size = [m2, m2]
        comm_bwd_topology = [t2, t2] 

        self.set_stats(name,
                       flops_fwd = flops_fwd,
                       use_tensor_cores = True,
                       mem_fwd = mem_fwd,
                       activation_buffer = activation_buffer,
                       weights_mem = 0,
                       weights_grad_mem = 0,
                       comm_fwd = comm_fwd, 
                       comm_fwd_type = comm_fwd_type,
                       comm_fwd_size = comm_fwd_size,
                       comm_fwd_topology = comm_fwd_topology,
                       flops_bwd = flops_bwd,
                       mem_bwd = mem_bwd,
                       comm_bwd = comm_bwd, 
                       comm_bwd_type = comm_bwd_type,
                       comm_bwd_size = comm_bwd_size,
                       comm_bwd_topology = comm_bwd_topology)

    def compute_times(self, flops, mem, comms, comm_sizes, comm_types, comm_tops):
        t_comp = self.get_time_flops(flops)
        t_mem  = self.get_time_mem(mem)
        t_compute = max(t_comp, t_mem)
        intensity = t_comp / t_mem

        t_comm = 0
        if isinstance(comms, list):
            for c, comm in enumerate(comms):
                comm_size = comm_sizes[c]
                comm_type = comm_types[c]
                comm_top = comm_tops[c]
                t_comm += self.get_time_comm(comm, comm_size, comm_type, comm_top)  
        else:
            t_comm = self.get_time_comm(comms, comm_sizes, comm_types, comm_tops)  

        t = t_compute + t_comm
        return t, t_comm, intensity

    def compute_time(self): # overwrite due to diff comm patterns
        # forward time
        self.stats['t_fwd'], self.stats['t_fwd_comm'], self.stats['intensity_fwd'] = self.compute_times(self.flops_fwd,
                                                                                                             self.mem_fwd,
                                                                                                             self.comm_fwd,
                                                                                                             self.comm_fwd_size,
                                                                                                             self.comm_fwd_type,
                                                                                                             self.comm_fwd_topology)
        self.stats['t_bwd'], self.stats['t_bwd_comm'], self.stats['intensity_bwd'] = self.compute_times(self.flops_bwd,
                                                                                                             self.mem_bwd,
                                                                                                             self.comm_bwd,
                                                                                                             self.comm_bwd_size,
                                                                                                             self.comm_bwd_type,
                                                                                                             self.comm_bwd_topology)
        self.stats['t'] = self.stats['t_fwd'] + self.stats['t_bwd']


    def get_stats(self):
        self.compute_time()
        return self.stats
