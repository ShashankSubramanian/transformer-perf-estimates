import numpy as np

''' 
    Perf estimates for attention kernels
'''

def logit_estimates(b, l, e, h, element_size=4E-6):
    """
    logit layer estimates
    parameters: b: batch size
                l: seq length
                e: embedding dim/hidden dim
                h: number of attention heads
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
    flops_units = 1E-9 # teraflops
    flops_per_mult = 1 * flops_units
    flops_per_add = 1 * flops_units

    q = e // h

    # total flops
    total_flops_fwd = b * h * l * l * (q * flops_per_mult + (q - 1) * flops_per_add)

    #total mem
    activation_in_mem = (b * h * l * q) * element_size
    activation_in_other_mem = (b * h * l * q) * element_size
    activation_out_mem = (b * h * l * l) * element_size
    activation_buffer = 2 * (b * h * l * q) * element_size # Q and K
    weights_mem = 0
    total_mem_fwd = activation_in_mem + activation_out_mem + activation_in_other_mem + weights_mem

    # stats_fwd
    stats_fwd = {"flops_fwd": total_flops_fwd,
                 "activation_in_mem": activation_in_mem,
                 "activation_in_other_mem": activation_in_other_mem,
                 "activation_out_mem": activation_out_mem,
                 "activation_buffer": activation_buffer,
                 'weights_mem': weights_mem,
                 'total_mem_fwd': total_mem_fwd}
    #############################
    ####### backward pass #######
    #############################
    total_flops_bwd = 2 * b * h * l * q * (l * flops_per_mult + (l - 1) * flops_per_add)
    activation_grad_mem = 2 * (b * h * l * q) * element_size
    activation_grad_mem_att = (b * h * l * l) * element_size
    total_mem_bwd = activation_grad_mem + activation_grad_mem_att

    stats_bwd = {"flops_bwd": total_flops_bwd,
                 "activation_grad_mem": activation_grad_mem + activation_grad_mem_att,
                 "weights_grad_mem": 0,
                 'total_mem_bwd': total_mem_bwd}

    stats = {**stats_fwd, **stats_bwd}
    return stats
