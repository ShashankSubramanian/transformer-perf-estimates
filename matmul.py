import numpy as np

''' 
    Perf estimates for matrix-matrix multiply
'''

def linear_estimates(b, l, e, f, element_size=4E-6, has_bias=False, flops_units=1E-12):
    """
    nn.Linear layer estimates
    parameters: b: batch size
                l: seq length
                e: embedding dim/1st dim
                f: hidden dim/2nd dim
                element_size: in MB
                has_bias: whether bias term is added
    
    tensor shapes: input tensor: (b,l,e)
                   weight tensor: (e,f)
                   bias tensor: (1,f)
                   output tensor: (b,l,f)
                   
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
    """
    
    #############################
    ####### forward pass ########
    #############################
    # can be different if complex numbers (drop for now)
    flops_per_mult = 1 * flops_units
    flops_per_add = 1 * flops_units

    # total flops
    total_flops_fwd = b * l * f * (e * flops_per_mult + (e - 1) * flops_per_add)
    if has_bias:
        total_flops_fwd += b * l * (f * flops_per_add)
        
    # total mem
    activation_in_mem = (b * l * e) * element_size
    activation_out_mem = (b * l * f) * element_size
    activation_buffer = (b * l * e) * element_size # store for bwd pass
    weights_mem = (e * f) * element_size
    if has_bias:
        weights_mem += (1 * f) * element_size
    total_mem_fwd = activation_in_mem + activation_out_mem + weights_mem

    
    # stats
    stats_fwd = {"flops_fwd": total_flops_fwd, 
                 "activation_buffer": activation_buffer,
                 'weights_mem': weights_mem, 
                 'total_mem_fwd': total_mem_fwd}

    
    #############################
    ####### backward pass #######
    #############################
    xgrad_flops = b * l * e * (f * flops_per_mult + (f - 1) * flops_per_add)
    wgrad_flops = e * f * (b * l * flops_per_mult + (b * l - 1) * flops_per_add)
    bgrad_flops = b * l * f * flops_per_add if has_bias else 0
    total_flops_bwd = xgrad_flops + wgrad_flops + bgrad_flops
    
    wgrad_mem = (e * f) * element_size
    bgrad_mem = (1 * f) * element_size if has_bias else 0
    weights_grad_mem = wgrad_mem + bgrad_mem
    xgrad_mem = (b * l * e) * element_size 
    ygrad_mem = (b * l * f) * element_size  

    num_bwd_ops = 3 if has_bias else 2 # need to bring ygrad_mem multiple times
    total_mem_bwd = weights_grad_mem + xgrad_mem + num_bwd_ops * ygrad_mem + weights_mem + activation_buffer

    
    stats_bwd = {"flops_bwd": total_flops_bwd, 
                 "weights_grad_mem": weights_grad_mem, 
                 'total_mem_bwd': total_mem_bwd}
    
    stats = {**stats_fwd, **stats_bwd}
    
    return stats

def linear_estimates_summa(b, l, e, f, element_size=4E-6, has_bias=False, flops_units=1E-12, 
                           parallelism={'dim1': 1, 'dim2': 1}, topology={'dim1': 'nvlink', 'dim2': 'ib'}):
    """
    nn.Linear layer estimates for local operations
    parameters: b: batch size
                l: seq length
                e: embedding dim/1st dim
                f: hidden dim/2nd dim
                element_size: in MB
                has_bias: whether bias term is added
                parallelism: dim1, dim2
    
    tensor shapes: input tensor: (b,l,e)
                   weight tensor: (e,f)
                   bias tensor: (1,f)
                   output tensor: (b,l,f)
                   
    layer arithmetic: 
        forward pass :  
            Y = X * W + B
            (b,l/m2,f/m1) = (b,l/m2,e/m1) * (e/m2,f/m1) + (1,f/m1)
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
            --------------------------------------------------------------        
            dL/dB = dL/dY * 1
            (1,f/m1) = \sum_{b,l/m2} (b,l/m2,f/m1)
            allreduce on m2 dim
 
    comments: 
        nb = m if square grid of procs; else it's some small number (?)
    """
    
    #############################
    ####### forward pass ########
    #############################
    # can be different if complex numbers (drop for now)
    flops_per_mult = 1 * flops_units
    flops_per_add = 1 * flops_units

    m1 = parallelism['dim1']
    m2 = parallelism['dim2']
    
    l_local = l // m2
    f_local = f // m1
    e_local_1 = e // m1
    e_local_2 = e // m2


    # total flops
    total_flops_fwd = b * l_local * f_local * (e * flops_per_mult + e * flops_per_add) # e outer products and e addns in total
    if has_bias:
        total_flops_fwd += b * l_local * (f_local * flops_per_add)
     
    # total mem
    activation_buffer = (b * l_local * e_local_1) * element_size # store for bwd pass
    weights_mem = (e_local_2 * f_local) * element_size
    if has_bias:
        weights_mem += (1 * f_local) * element_size
#    total_mem_fwd = activation_in_mem + activation_out_mem + weights_mem
     
    # assuming a loop of nb = 512 times
    # input1 is b * l_local * e/n_b
    # input2 is e/n_b * f_local
    # output is b * l_local * f_local
    # these ops are done n_b number of times
    # panel size e/n_b is some value << e/max(m1,m2)
     
    # careful, nb is arbitrarily chosen here
    n_b = e // 512 if m1 != m2 else m1
    total_mem_fwd  = (b * l_local * e + e * f_local + b * l_local * f_local * n_b) * element_size
    
    # stats
    stats_fwd = {"flops_fwd": total_flops_fwd, 
                 "activation_buffer": activation_buffer,
                 'weights_mem': weights_mem, 
                 'total_mem_fwd': total_mem_fwd}
    
    #############################
    ####### backward pass #######
    #############################
    
    xgrad_flops = b * l_local * e * (f_local * flops_per_mult + (f_local - 1) * flops_per_add) 
    wgrad_flops = e * f_local * (b * l_local * flops_per_mult + (b * l_local - 1) * flops_per_add)
    bgrad_flops = b * l_local * f_local * flops_per_add if has_bias else 0
    total_flops_bwd = xgrad_flops + wgrad_flops + bgrad_flops
    
    wgrad_mem = (e_local_2 * f_local) * element_size
    bgrad_mem = (1 * f_local) * element_size if has_bias else 0
    weights_grad_mem = wgrad_mem + bgrad_mem
    total_mem_bwd =  (f_local * e + b * l_local * f_local * n_b + b * l_local * e) * element_size
    total_mem_bwd +=  (e * b * l_local + b * l_local * f_local * n_b +  e * f_local) * element_size
    total_mem_bwd +=  (b * l_local * f_local + 1 * f_local) * element_size if has_bias else 0

    
    stats_bwd = {"flops_bwd": total_flops_bwd, 
                 "weights_grad_mem": weights_grad_mem, 
                 'total_mem_bwd': total_mem_bwd}
    
    stats = {**stats_fwd, **stats_bwd}

    # sync/comm layers
    m1_parallel = (m1 > 1)
    m2_parallel = (m2 > 1)
    t1 = topology['dim1']
    t2 = topology['dim2']
    # sync/comm layers
    stats["comm_fwd"] = [m1_parallel * (b * l_local * e) * element_size,
                         m2_parallel * (e * f_local) * element_size] # both inputs are broadcasted in diff dims
    stats["comm_fwd_type"] = ["broadcast", "broadcast"]
    stats["comm_fwd_size"] = [m1, m2]
    stats["comm_fwd_topology"] = [t1, t2]

    stats["comm_bwd"] = [m2_parallel * (e * f_local) * element_size, # dl/dx = dl/dy * wt
                         m1_parallel * (b * l_local * e) * element_size, # reduce result
                         m1_parallel * (b * l_local * e) * element_size, # dl/dw = xt * dl/dy
                         m2_parallel * (e * f_local) * element_size] # reduce temp result
    stats["comm_bwd_type"] = ["broadcast", "reduce", "broadcast", "reduce"]
    stats["comm_bwd_size"] = [m2, m1, m1, m2]
    stats["comm_bwd_topology"] = [t2, t1, t1, t2]
    if has_bias:
        stats["comm_bwd"] += [m2_parallel * (f_local) * element_size] # allreduce bias
        stats["comm_bwd_type"] += ["allreduce"]
        stats["comm_bwd_size"] += [m2]
        stats["comm_bwd_topology"] += [t2]
    
    return stats

def logit_estimates(b, l, q, h, element_size=4E-6, flops_units=1E-12):
    """
    logit layer estimates
    parameters: b: batch size
                l: seq length
                h: number of attention heads
                q: embedding dim/h
                element_size: in MB

    tensor shapes: input tensor: (b,h,l,q)
                   output tensor: (b,h,l,q)

    layer arithmetic:
        define: q = e/h
        forward pass:
             A = Q * K^T
             (b,h,l,l) = (b,h,l,q) * (b,h,q,l)
        backward pass:
             dL/dK = dL/dA * Q
             (b,h,l,q) = (b,h,l,l) * (b,h,l,q)
             dL/dQ = dL/dA * K
             (b,h,l,q) = (b,h,l,l) * (b,h,l,q)


    comments:
    """
    #############################
    ####### forward pass ########
    #############################
    # can be different if complex numbers (drop for now)
    flops_per_mult = 1 * flops_units
    flops_per_add = 1 * flops_units

    # total flops
    total_flops_fwd = b * h * l * l * (q * flops_per_mult + (q - 1) * flops_per_add)

    #total mem
    activation_in_mem = (b * h * l * q) * element_size
    activation_in_mem += (b * h * l * q) * element_size
    activation_out_mem = (b * h * l * l) * element_size
    activation_buffer = 2 * (b * h * l * q) * element_size # Q and K
    total_mem_fwd = activation_in_mem + activation_out_mem

    # stats_fwd
    stats_fwd = {"flops_fwd": total_flops_fwd,
                 "activation_buffer": activation_buffer,
                 'weights_mem': 0,
                 'total_mem_fwd': total_mem_fwd}
    #############################
    ####### backward pass #######
    #############################
    total_flops_bwd = 2 * b * h * l * q * (l * flops_per_mult + (l - 1) * flops_per_add)
    activation_grad_mem = 2 * (b * h * l * q) * element_size
    activation_grad_mem_att = 2 * (b * h * l * l) * element_size
    total_mem_bwd = activation_grad_mem + activation_grad_mem_att + activation_buffer

    stats_bwd = {"flops_bwd": total_flops_bwd,
                 "weights_grad_mem": 0,
                 'total_mem_bwd': total_mem_bwd}

    stats = {**stats_fwd, **stats_bwd}
    return stats

def logit_estimates_summa(b, l, q, h, element_size=4E-6, flops_units=1E-12,
                           parallelism={'dim1': 1, 'dim2': 1}, topology={'dim1': 'nvlink', 'dim2': 'ib'}):
    """
    logit layer estimates
    parameters: b: batch size
                l: seq length
                h: number of attention heads
                q: embedding dim/h
                element_size: in MB

    tensor shapes: input tensor: (b,h,l,q)
                   output tensor: (b,h,l,q)

    layer arithmetic:
        define: q = e/h
        forward pass:
             A = Q * K^T
             (b,h,l/m2,l/m1) = (b,h,l/m2,q/m1) * (b,h,l/m2,q/m1)^T
        backward pass:
             dL/dK = dL/dA * Q
             (b,h,l/m2,q/m1) = (b,h,l/m2,l/m1) * (b,h,l/m2,q/m1)
             dL/dQ = dL/dA * K
             (b,h,l/m2,q/m1) = (b,h,l/m2,l/m1) * (b,h,l/m2,q/m1)
    comments:
    """
    #############################
    ####### forward pass ########
    #############################
    # can be different if complex numbers (drop for now)
    flops_per_mult = 1 * flops_units
    flops_per_add = 1 * flops_units

    m1 = parallelism['dim1']
    m2 = parallelism['dim2']

    l_local_1 = l // m1
    l_local_2 = l // m2
    q_local = q // m1 

    # total flops
    # do nb times
    # broad cast bh,l/nb,q/m1 col-wise
    # temp matmul bh,l/m2,q/m1 * bh,q/m1,l/nb
    # reduce bh,l/m2,l/nb row-wise
    # careful, nb is arbitrarily chosen here
    nb = l // 512 if m1 != m2 else m1
    total_flops_fwd = b * h * l_local_2 * l * (q_local * flops_per_mult + (q_local - 1) * flops_per_add)

    #total mem
    activation_buffer = 2 * (b * h * l_local_2 * q_local) * element_size # Q and K
    weights_mem = 0
    total_mem_fwd = (b * h * l * q_local + b * h * l_local_2 * q_local * nb + b * h * l_local_2 * l) * element_size

    # stats_fwd
    stats_fwd = {"flops_fwd": total_flops_fwd,
                 "activation_buffer": activation_buffer,
                 'weights_mem': weights_mem,
                 'total_mem_fwd': total_mem_fwd}
    #############################
    ####### backward pass #######
    #############################
    # broadcast bh,l/m2,l/nb rowwise and bh,l/nb,q/m1 colwise, admm locally
    total_flops_bwd = 2 * b * h * l_local_2 * q_local * (l * flops_per_mult + (l - 1) * flops_per_add)
    total_mem_bwd = (2 * b * h * l_local_2 * l + 2 * b * h * l * q_local + 2 * b * h * l_local_2 * q_local * nb) * element_size

    stats_bwd = {"flops_bwd": total_flops_bwd,
                 "weights_grad_mem": 0,
                 'total_mem_bwd': total_mem_bwd}

    stats = {**stats_fwd, **stats_bwd}

    # sync/comm layers
    m1_parallel = (m1 > 1)
    m2_parallel = (m2 > 1)
    t1 = topology['dim1']
    t2 = topology['dim2']
    # sync/comm layers
    stats["comm_fwd"] = [m2_parallel * (b * h * l * q_local) * element_size,
                         m1_parallel * (b * h * l_local_2 * l) * element_size] 
    stats["comm_fwd_type"] = ["broadcast", "reduce"]
    stats["comm_fwd_size"] = [m2, m1]
    stats["comm_fwd_topology"] = [t2, t1]

    # broadcast bh,l/m2,l/nb rowwise and bh,l/nb,q/m1 colwise, admm locally
    stats["comm_bwd"] = [m1_parallel * (2 * b * h * l_local_2 * l) * element_size,
                         m2_parallel * (2 * b * h * l * q_local) * element_size]
    stats["comm_bwd_type"] = ["broadcast", "broadcast"]
    stats["comm_bwd_size"] = [m1, m2]
    stats["comm_bwd_topology"] = [t1, t2]

    return stats

def attend_estimates(b, l, q, h, element_size=4E-6, flops_units=1E-12):
    """
    attend layer estimates: for now, diff from linear/matmul 
    because both inputs are activations. TODO: combine both 
    into a single matmul primitive
    parameters: b: batch size
                l: seq length
                h: number of attention heads
                q: embedding dim/h
                element_size: in MB

    
    tensor shapes: input tensor: (b, h, l, l)
                   input tensor: (b, h, l, q)
                   output tensor: (b, h, l, q)
                   
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
    
    #############################
    ####### forward pass ########
    #############################
    # can be different if complex numbers (drop for now)
    flops_per_mult = 1 * flops_units
    flops_per_add = 1 * flops_units

    # total flops
    total_flops_fwd = b * h * l * q * (l * flops_per_mult + (l - 1) * flops_per_add)
        
    # total mem
    activation_in_mem = (b * h * l * l) * element_size
    activation_in_mem += (b * h * l * q) * element_size
    activation_out_mem = (b * h * l * q) * element_size
    activation_buffer = (b * h * l * l) * element_size # store for bwd pass
    activation_buffer += (b * h * l * q) * element_size # store for bwd pass
    total_mem_fwd = activation_in_mem + activation_out_mem
    
    # stats
    stats_fwd = {"flops_fwd": total_flops_fwd, 
                 "activation_buffer": activation_buffer,
                 'weights_mem': 0, 
                 'total_mem_fwd': total_mem_fwd}

    #############################
    ####### backward pass #######
    #############################
    total_flops_bwd = b * h * l * l * (q * flops_per_mult + (q - 1) * flops_per_add)
    total_flops_bwd += b * h * l * q * (l * flops_per_mult + (l - 1) * flops_per_add)
    activation_grad_mem = 3 * (b * h * l * q) * element_size
    activation_grad_mem_att = (b * h * l * l) * element_size
    total_mem_bwd = activation_grad_mem + activation_grad_mem_att + activation_buffer

    stats_bwd = {"flops_bwd": total_flops_bwd,
                 "activation_grad_mem": activation_grad_mem + activation_grad_mem_att,
                 "weights_grad_mem": 0,
                 'total_mem_bwd': total_mem_bwd}
    
    stats = {**stats_fwd, **stats_bwd}
    
    return stats

def attend_estimates_summa(b, l, q, h, element_size=4E-6, flops_units=1E-12, 
                           parallelism={'dim1': 1, 'dim2': 1}, topology={'dim1': 'nvlink', 'dim2': 'ib'}):
    """
    attend layer estimates
    parameters: b: batch size
                l: seq length
                h: number of attention heads
                q: embedding dim/h
                element_size: in MB

    
    tensor shapes: input tensor: (b, h, l, l)
                   input tensor: (b, h, l, q)
                   output tensor: (b, h, l, q)
                   
    layer arithmetic: 
        forward pass :  
            Y = AV
            (b,h,l/m2,q/m1) = (b,h,l/m2,l/m1) * (b,h,l/m2,q/m1)
        backward pass: (L = loss)
            dL/dA = dL/dY * V^T
            (b,h,l/m2,l/m1) = (b,h,l/m2,q/m1) * (b,h,q/m1,l/m2)
            dL/dV = A^T * dL/dY
            (b,h,l/m2,q/m1) = (b,h,l/m2,l/m1) * (b,h,l/m2,q/m1) 
 
    comments: 
    """
    
    #############################
    ####### forward pass ########
    #############################
    # can be different if complex numbers (drop for now)
    flops_per_mult = 1 * flops_units
    flops_per_add = 1 * flops_units

    m1 = parallelism['dim1']
    m2 = parallelism['dim2']

    l_local_1 = l // m1
    l_local_2 = l // m2
    q_local = q // m1 

    # total flops
    total_flops_fwd = b * h * l_local_2 * q_local * (l * flops_per_mult + l * flops_per_add) # l outer products and l addns in total
     
    # total mem
    activation_buffer = (b * h * l_local_2 * l_local_1 + b * h * l_local_2 * q_local) * element_size # store for bwd pass
    weights_mem = 0 
    # careful, nb is arbitrarily chosen here
    nb = l // 512 if m1 != m2 else m1
    total_mem_fwd  = (b * h * l_local_2 * l + b * h * l * q_local + b * h * l_local_2 * q_local * nb) * element_size
    
    # stats
    stats_fwd = {"flops_fwd": total_flops_fwd, 
                 "activation_buffer": activation_buffer,
                 'weights_mem': weights_mem, 
                 'total_mem_fwd': total_mem_fwd}
    
    #############################
    ####### backward pass #######
    #############################
    
    # broadcast  bh,l/nb,q/m1 colwise, bh,l/m2,q/m1 * bh,q/m1,l/nb
    agrad_flops = b * h * l_local_2 * l * (q_local * flops_per_mult + (q_local - 1) * flops_per_add) 
    # broadcast  bh,l/m2,l/nb rowsie, bh,l/nb,l/m2 * bh,l/m2,q/m1
    vgrad_flops = b * h * l * q_local * (l_local_2 * flops_per_mult + (l_local_2 - 1) * flops_per_add)
    total_flops_bwd = agrad_flops + vgrad_flops
    
    total_mem_bwd = b * h * (l_local_2 * q_local * nb + q_local * l + l_local_2 * l) * element_size
    total_mem_bwd += b * h * (l_local_2 * q_local * nb + l * l_local_2 + l * q_local) * element_size

    stats_bwd = {"flops_bwd": total_flops_bwd, 
                 "weights_grad_mem": 0, 
                 'total_mem_bwd': total_mem_bwd}
    
    stats = {**stats_fwd, **stats_bwd}

    # sync/comm layers
    m1_parallel = (m1 > 1)
    m2_parallel = (m2 > 1)
    t1 = topology['dim1']
    t2 = topology['dim2']
    # sync/comm layers
    stats["comm_fwd"] = [m1_parallel * (b * h * l_local_2 * l) * element_size,
                         m2_parallel * (b * h * l * q_local) * element_size] # both inputs are broadcasted in diff dims
    stats["comm_fwd_type"] = ["broadcast", "broadcast"]
    stats["comm_fwd_size"] = [m1, m2]
    stats["comm_fwd_topology"] = [t1, t2]

    # broadcast  bh,l/nb,q/m1 colwise, bh,l/m2,q/m1 * bh,q/m1,l/nb
    # broadcast  bh,l/m2,l/nb rowsie, bh,l/nb,l/m2 * bh,l/m2,q/m1
    stats["comm_bwd"] = [m2_parallel * (b * h * l * q_local) * element_size, # broadcast
                         m1_parallel * (b *  h * l_local_2 * l) * element_size, # reduce result
                         m1_parallel * (b * h * l_local_2 * l) * element_size, # broadcast
                         m2_parallel * (b * h * l * q_local) * element_size] # reduce result
    stats["comm_bwd_type"] = ["broadcast", "reduce", "broadcast", "reduce"]
    stats["comm_bwd_size"] = [m2, m1, m1, m2]
    stats["comm_bwd_topology"] = [t2, t1, t1, t2]
    
    return stats

def fused_logit_softmax_dpr_attend_estimates(b, l, q, h, element_size=4E-6, flops_units=1E-12):
    """
    fused LA based on flashattention layer estimates
    parameters: b: batch size
                l: seq length
                h: number of attention heads
                e: embedding size
                q: embedding dim/h
                element_size: in MB

    tensor shapes: input tensor: (b,h,l,q)
                   output tensor: (b,h,l,q)

    layer arithmetic:
        define: q = e/h
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
             dL/dK = dL/dA * Q
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
        backward pass: 

    comments:
       assume it's a single operation/kernel
    """
    #############################
    ####### forward pass ########
    #############################
    # can be different if complex numbers (drop for now)
    flops_per_mult = 1 * flops_units
    flops_per_add = 1 * flops_units
    flops_per_exp = 1 * flops_units

    # total flops
    # logits
    total_flops_fwd = b * h * l * l * (q * flops_per_mult + (q - 1) * flops_per_add)
    # softmax
    total_flops_fwd += b * h * l * l * (flops_per_exp + flops_per_mult) + (b * h * l * (l - 1)) * flops_per_add
    # dropout
    total_flops_fwd += (b * h * l * l) * flops_per_mult
    # attend
    total_flops_fwd += b * h * l * q * (l * flops_per_mult + (l - 1) * flops_per_add)

    #total mem
    activation_in_mem = (b * h * l * q) * element_size # Q
    activation_in_mem += 2 * (b * h * l * q) * element_size # K and V
    activation_in_mem += (b * h * l) * element_size # stats for softmax
    activation_out_mem = (b * h * l * q) * element_size # result

    activation_buffer = 3 * (b * h * l * q) * element_size # q, k, v 
    activation_buffer += (b * h * l) * element_size # random number generator states (droppout mask is not stored); dont know if this is float
    activation_buffer += (b * h * l) * element_size # stats for softmax
    
    # TODO: in software this might be stored (even though the next layer will have it: might need to revisit if not
    activation_buffer += (b * h * l * q) * element_size # result for flashattn bwd
    weights_mem = 0
    total_mem_fwd = activation_in_mem + activation_out_mem + weights_mem

    # stats_fwd
    stats_fwd = {"flops_fwd": total_flops_fwd,
                 "activation_buffer": activation_buffer,
                 'weights_mem': weights_mem,
                 'total_mem_fwd': total_mem_fwd}
    #############################
    ####### backward pass #######
    #############################
    # logits
    total_flops_bwd = 2 * b * h * l * q * (l * flops_per_mult + (l - 1) * flops_per_add)
    # softmax
    total_flops_bwd += (2 * b * h * l * l) * flops_per_mult +  (b * h * l * (l - 1)) * flops_per_add + (b * h * l * l) * flops_per_add
    # dropout
    total_flops_bwd += b * h * l * l  * flops_per_mult
    # attend
    total_flops_bwd += b * h * l * l * (q * flops_per_mult + (q - 1) * flops_per_add)
    total_flops_bwd += b * h * l * q * (l * flops_per_mult + (l - 1) * flops_per_add)

    # extra fwd flops since attn is remat
    # logit
    total_flops_bwd +=  b * h * l * l * (q * flops_per_mult + (q - 1) * flops_per_add)
    # softmax
    total_flops_bwd += b * h * l * l * (flops_per_exp + flops_per_mult) + (b * h * l * (l - 1)) * flops_per_add
    # dropout
    total_flops_bwd += (b * h * l * l) * flops_per_mult

    # mem
    activation_grad_mem = 4 * (b * h * l * q) * element_size # dq, dk, dv, dresult
    total_mem_bwd = activation_grad_mem + activation_buffer

    stats_bwd = {"flops_bwd": total_flops_bwd,
                 "weights_grad_mem": 0,
                 'total_mem_bwd': total_mem_bwd}

    stats = {**stats_fwd, **stats_bwd}
    return stats

