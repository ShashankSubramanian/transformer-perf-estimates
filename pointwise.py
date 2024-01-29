import numpy as np

'''
    Perf estimates for pointwise ops
'''

def nonlinear_act_estimates(b, l, e, element_size=4E-6):
    """
    activation function estimates
    parameters: b: batch size
                l: seq length
                e: embedding dim/hidden dim
                element_size: in MB
    
    tensor shapes: input tensor: (b,l,e)
                   output tensor: (b,l,e)
                   
    layer arithmetic: 
        forward pass: 
             Y = GELU(X)
            (b,l,e) = (b,l,e)
        backward pass:
            dL/dX = dL/dY * f(X) [?]
            (b,l,e) = (b,l,e) 

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
    total_flops_fwd = (b * l * e) * flops_per_mult
    
    #total mem
    activation_in_mem = (b * l * e) * element_size
    activation_out_mem = (b * l * e) * element_size
    activation_buffer = (b * l * e) * element_size
    weights_mem = 0
    total_mem_fwd = activation_in_mem + activation_out_mem + weights_mem
    
    # stats_fwd
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
    total_flops_bwd =  b * l * e * flops_per_mult
    activation_grad_mem = (b * l * e) * element_size
    total_mem_bwd = activation_grad_mem + activation_buffer
    
    stats_bwd = {"flops_bwd": total_flops_bwd, 
                 "activation_grad_mem": activation_grad_mem, 
                 "weights_grad_mem": 0, 
                 'total_mem_bwd': total_mem_bwd}
    
    stats = {**stats_fwd, **stats_bwd}
    return stats

def dropout_estimates(b, l, e, element_size=4E-6, mask_element_size=1E-6):
    """
    dropout layer estimates
    parameters: b: batch size
                l: seq length
                e: embedding dim/hidden dim
                element_size: in MB
                mask_element_size: in MB (random mask)
    
    tensor shapes: input tensor: (b,l,e)
                   output tensor: (b,l,e)
                   
    layer arithmetic: 
        forward pass: 
             Y = random_mask(X)
            (b,l,e) = (b,l,e) * (b,l,e)
        backward pass:
            
    
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
    total_flops_fwd = (b * l * e) * flops_per_mult
    
    #total mem
    activation_in_mem = (b * l * e) * element_size
    activation_in_other_mem = (b * l * e) * mask_element_size
    activation_out_mem = (b * l * e) * element_size
    activation_buffer = (b * l * e) * mask_element_size
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
    total_flops_bwd =  b * l * e  * flops_per_mult
    activation_grad_mem = (b * l * e) * element_size
    total_mem_bwd = activation_grad_mem + activation_buffer
    
    stats_bwd = {"flops_bwd": total_flops_bwd, 
                 "activation_grad_mem": activation_grad_mem, 
                 "weights_grad_mem": 0, 
                 'total_mem_bwd': total_mem_bwd}
    
    stats = {**stats_fwd, **stats_bwd}
    return stats

def softmax_estimates(b, l, h, element_size=4E-6):
    """
    dropout layer estimates
    parameters: b: batch size
                l: seq length
                h: number of attention heads
                element_size: in MB
    
    tensor shapes: input tensor: (b,h,l,q)
                   output tensor: (b,h,l,q)
                   
    layer arithmetic: 
        define: q = e/h
        forward pass: 
             Y = softmax(X)
             (b,h,l,l) = (b,h,l,l)
        backward pass: 
             dL/dX = Y . dL/dY - Y . sum(Y . dL/dY, axis=-1)
             (b,h,l,l) = (b,h,l,l) . (b,h,l,l) - (b,h,l,l) . (b,h,l,1)
    
    comments: . is pointwise mult
    """
    #############################
    ####### forward pass ########
    #############################
    # can be different if complex numbers (drop for now)
    flops_units = 1E-9 # teraflops
    flops_per_mult = 1 * flops_units
    flops_per_add = 1 * flops_units
    flops_per_exp = 1 * flops_units

    # total flops
    total_flops_fwd = b * h * l * l * (flops_per_exp + flops_per_mult) + (b * h * l * (l - 1)) * flops_per_add
    
    #total mem
    activation_in_mem = (b * h * l * l) * element_size
    activation_in_other_mem = 0
    activation_out_mem = (b * h * l * l) * element_size
    activation_buffer = (b * h * l * l) * element_size
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
    total_flops_bwd =  (2 * b * h * l * l) * flops_per_mult +  (b * h * l * (l - 1)) * flops_per_add + (b * h * l * l) * flops_per_add
    activation_grad_mem = 2 * (b * h * l * l) * element_size
    total_mem_bwd = activation_grad_mem + activation_buffer
    
    stats_bwd = {"flops_bwd": total_flops_bwd, 
                 "activation_grad_mem": activation_grad_mem, 
                 "weights_grad_mem": 0, 
                 'total_mem_bwd': total_mem_bwd}
    
    stats = {**stats_fwd, **stats_bwd}
    return stats
