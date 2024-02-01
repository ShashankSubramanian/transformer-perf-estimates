import numpy as np

'''
    Perf estimates for normalization layers
'''

def layer_norm_estimates(b, l, e, element_size=4E-6, flops_units=1E-12):
    """
    TODO: parallelism incomplete!
    my notes: all_gather after in fwd pass always (but is that true?) careful
              reduce-scatter in bwd for first two certainly, the activation grad? careful

    layernorm layer estimates
    parameters: b: batch size
                l: seq length
                e: embedding dim/hidden dim
                element_size: in MB

    tensor shapes: input tensor: (b,h,l,q)
                   output tensor: (b,h,l,q)

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
    #############################
    ####### forward pass ########
    #############################
    # can be different if complex numbers (drop for now)
    flops_per_mult = 1 * flops_units
    flops_per_add = 1 * flops_units

    # total flops
    total_flops_fwd = 2 * b * l * e * flops_per_add # mean
    total_flops_fwd += 2 * b * l * e * flops_per_add + b * l * e * flops_per_mult # var
    total_flops_fwd += b * l * e * flops_per_add + b * l * e * flops_per_mult # scale
    total_flops_fwd += b * l * e * flops_per_mult + b * l * e * flops_per_add

    #total mem
    activation_in_mem = (b * l * e) * element_size
    activation_in_other_mem = 0
    activation_out_mem = (b * l * e) * element_size
    activation_buffer = (b * l * e) * element_size
    weights_mem = 2 * e * element_size
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
    # little rough calcs, pretty sure some constant factor is off..
    total_flops_bwd = 2 * (b * l * e * flops_per_mult + b * l * e * flops_per_add) # g,b
    total_flops_bwd += (5 * b * l * e * flops_per_mult) + (4 * b * l * (e - 1) * flops_per_add)
    activation_grad_mem = 2 * (b * l * e) * element_size #dldy, dldx
    weights_grad_mem = 2 * e * element_size
    total_mem_bwd = activation_grad_mem + weights_grad_mem + activation_buffer

    stats_bwd = {"flops_bwd": total_flops_bwd,
                 "activation_grad_mem": activation_grad_mem,
                 "weights_grad_mem": weights_grad_mem,
                 'total_mem_bwd': total_mem_bwd}

    stats = {**stats_fwd, **stats_bwd}
    return stats
