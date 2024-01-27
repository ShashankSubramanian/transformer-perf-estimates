import numpy as np

''' 
    Perf estimates for matrix-matrix multiply
'''

def linear_estimates(b, l, e, f, element_size=4E-6, has_bias=False):
    """
    nn.Linear layer estimates
    parameters: b: batch size
                l: seq length
                e: embedding dim/1st dim
                f: hidden dim/2nd dim
                element_size: in MB
                has_bias: whether bias term is added
                parallelism: number of GPUs used to shard e or f dims
    
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
    flops_units = 1E-9 # teraflops
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
                 "activation_in_mem": activation_in_mem, 
                 "activation_in_other_mem": 0, 
                 "activation_out_mem": activation_out_mem,
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
    bgrad_mem = (1 * f) * element_size
    weights_grad_mem = wgrad_mem + bgrad_mem
    xgrad_mem = (b * l * e) * element_size 
    ygrad_mem = (b * l * f) * element_size 
    total_mem_bwd = weights_grad_mem + xgrad_mem + ygrad_mem

    
    stats_bwd = {"flops_bwd": total_flops_bwd, 
                 "activation_grad_mem": xgrad_mem + ygrad_mem, 
                 "weights_grad_mem": weights_grad_mem, 
                 'total_mem_bwd': total_mem_bwd}
    
    stats = {**stats_fwd, **stats_bwd}
    
    return stats
