import pandas as pd
from matmul import linear_estimates, logit_estimates, attend_estimates
from norm import layer_norm_estimates
from pointwise import softmax_estimates, dropout_estimates, nonlinear_act_estimates
from time_projections import get_time_flops, get_time_mem, get_time_comm, get_topology, get_total_time


def compute_timings_and_stats(summary, system):
    ''' timings, any other df stats '''

    # which layers use tensor cores
    tensor_core_layers = ['fc1', 'fc2', 'qkv_proj', 'v_proj', 'logits', 'attend']
   
    # time for forward
    summary['t_comp_fwd'] = summary.apply(lambda x: get_time_flops(x["flops_fwd"], 
                                                                   use_tensor=(x["layer"] in tensor_core_layers),
                                                                   system=system), axis=1)
    summary['t_mem_fwd'] = summary.apply(lambda x: get_time_mem(x["total_mem_fwd"], system=system), axis=1)
    # time for backward
    summary['t_comp_bwd'] = summary.apply(lambda x: get_time_flops(x["flops_bwd"], 
                                                                   use_tensor=(x["layer"] in tensor_core_layers),
                                                                   system=system), axis=1)
    summary['t_mem_bwd'] = summary.apply(lambda x: get_time_mem(x["total_mem_bwd"], system=system), axis=1)
    
    # times
    summary['intensity'] = summary['t_comp_fwd'] / summary['t_mem_fwd']
    # roofline
    summary['t_fwd'] = summary.apply(lambda x: max(x['t_comp_fwd'], x['t_mem_fwd']), axis=1)
    summary['t_bwd'] = summary.apply(lambda x: max(x['t_comp_bwd'], x['t_mem_bwd']), axis=1)
    
    # time for communication
    use_empirical = False
    summary['comm_topology'] = summary.apply(lambda x: get_topology(x["comm_size"], system=system), axis=1)
    summary['t_comm_fwd'] = summary.apply(lambda x: get_time_comm(x["comm_fwd"],
                                                                  n_gpus=x["comm_size"],
                                                                  comm_type=x["comm_fwd_type"], 
                                                                  topology=x["comm_topology"],
                                                                  empirical=use_empirical,
                                                                  system=system), axis=1)
    summary['t_comm_bwd'] = summary.apply(lambda x: get_time_comm(x["comm_bwd"],
                                                                  n_gpus=x["comm_size"],
                                                                  comm_type=x["comm_bwd_type"], 
                                                                  topology=x["comm_topology"],
                                                                  empirical=use_empirical,
                                                                  system=system), axis=1)
    
    # total time
    summary['t_total_fwd'] = summary.apply(lambda x: get_total_time(x['t_fwd'], x['t_comm_fwd'], use_max=False), axis=1)
    summary['t_total_bwd'] = summary.apply(lambda x: get_total_time(x['t_bwd'], x['t_comm_bwd'], use_max=False), axis=1)
    
    # fraction
    summary['frac_t_comm_fwd'] = summary['t_comm_fwd'] / summary['t_total_fwd']
    summary['frac_t_comm_bwd'] = summary['t_comm_bwd'] / summary['t_total_bwd']
    
    # memory per computing unit
    summary['total_mem_fwd'] = summary.apply(lambda x: x["total_mem_fwd"], axis=1)
    summary['total_mem_bwd'] = summary.apply(lambda x: x["total_mem_bwd"], axis=1)

    return summary


################### 1D #######################
def MLP_estimates_1D(b, l, e, f, depth, element_size=4E-6, mask_element_size=1E-6, 
                     flops_units=1E-12, parallelism={'m1': 1, 'm2': 1}, system={}):
    """
    MLP layer estimates
    parameters: b: batch size
                l: seq length
                e: embedding dim
                f: hidden dim
                element_size: in MB
                mask_element_size: in MB (for dropout)
    
    tensor shapes: input tensor: (b,l,e)
                   output tensor: (b,l,e)
                   
    layer arithmetic: 
        forward pass: 
             X = XW + b
             (b,l,f) = (b,l,e) * (e,f) + (1,f)
             X = nonlinear(X)
             (b,l,f) = (b,l,f)
             X = dropout(X)
             (b,l,f) = (b,l,f) * (b,l,f) [random mask]
             X = linear(X)
             (b,l,e) = (b,l,f) * (f,e) + (1,e)
             X = dropout(X)
             (b,l,e) = (b,l,e) * (b,l,e) [random mask]
            
        backward pass:
             chain rule
             
    parallelism:
            X = XW + b
            (b,l,f/m) = (b,l,e) * (e,f/m) + (1,f/m)
            X = nonlinear(X)
            (b,l,f/m) = (b,l,f/m)
            X = dropout(X)
            (b,l,f/m) = (b,l,f/m) * (b,l,f/m) [random mask]
            X = linear(X)
            (b,l,e) = (b,l,f/m) * (f/m,e) + (1,e)
            X = dropout(X)
            (b,l,e) = (b,l,e) * (b,l,e) [random mask]
            
    comments: 
    """
    
    summary = []
    
    flops_per_add = 1 * flops_units
    
    m1 = parallelism['m1']
    m2 = 1 # parallelism['m2'] # not used in 1D parallelism (set to 1)
    
    m1_parallel = (m1 > 1)
    
    total_time = 0
    
     ########################################################################################################
        
    stats = linear_estimates(b, l, e, f // m1, element_size=element_size, has_bias=True, flops_units=flops_units)
    stats["layer"] = "fc1"   
    # sync/comm layers
    # no fwd comms
    stats["comm_bwd"] = (m1-1)/m1 * (b * l * e) * element_size # bwd comms for partial sums of b,l,e
    stats["comm_bwd_type"] = "reducescatter" 
    stats["comm_size"] = m1
    #TODO add comps for reduce-scatter
    stats["flops_bwd"] += (m1-1)/m1 * (b * l * e) * flops_per_add 
    summary.append(stats)
    
     ########################################################################################################
    
    stats = nonlinear_act_estimates(b, l, f // m1, element_size=element_size, flops_units=flops_units)
    stats["layer"] = "act"
    summary.append(stats)
    
    ##########################################################################################################
    stats = dropout_estimates(b, l, f // m1, element_size=element_size, mask_element_size=mask_element_size, flops_units=flops_units)
    stats["layer"] = "dpr1"
    summary.append(stats)
    

    ##########################################################################################################
    
    stats = linear_estimates(b, l, f // m1, e, element_size=element_size, has_bias=True, flops_units=flops_units)
    stats["layer"] = "fc2"
    # sync/comm layers
    # no bwd comms
    stats["comm_fwd"] =  (m1-1)/m1 * (b * l * e) * element_size # fwd comms for partial sums of b,l,e
    stats["comm_fwd_type"] = "reducescatter"
    stats["comm_size"] = m1
    #TODO add comps for reduce-scatter
    stats["flops_fwd"] += (m1-1)/m1 * (b * l * e) * flops_per_add 
    summary.append(stats)
    
  #############################################################################################################
    
    stats = dropout_estimates(b, l // m1, e, element_size=element_size, mask_element_size=mask_element_size, flops_units=flops_units)
    stats["layer"] = "dpr2"
    # sync/comm layers
    stats["comm_bwd"] = (m1-1)/m1 *  (b * l * e) * element_size
    stats["comm_bwd_type"] = "allgather"
    stats["comm_size"] = m1
    summary.append(stats)
    
    ############################################################################################################
    
    summary = pd.DataFrame(summary)
    summary = compute_timings_and_stats(summary, system)
    
    return summary


def self_attention_estimates_1D(b, l, e, h, element_size=4E-6, mask_element_size=1E-6, 
                             flops_units=1E-12, parallelism={'m1': 1, 'm2': 1}, system={}):
    """
    dropout layer estimates
    parameters: b: batch size
                l: seq length
                e: embedding dim/hidden dim
                h: number of attention heads
                element_size: in MB
    
    tensor shapes: input tensor: (b,l,e)
                   output tensor: (b,l,e)
                   
    layer arithmetic: 
        define: q = e/h
        forward pass: 
             X = norm(X)
             Q = XW, K = XW, V = XW
             (b,l,h,q,3) = (b,l,e) * (e,3hq)
             A = QK'/sqrt(q)
             (b,h,l,l) = (b,h,l,q) * (b,h,q,l)
             A = softmax(A)
             (b,h,l,l) = (b,h,l,l)
             A = dpr(A)
             Y = AV
             (b,h,l,q) = (b,h,l,l) * (b,h,l,q)
             Y = VW
             (b,l,e) = (b,l,hq) * (hq,e)
             Y = dpr(Y)
             (b,l,e) = (b,l,e)
             Y = norm(Y)
             (b,l,e) = (b,l,e)
             
        backward pass:
             chain rule
             
        parallelism:
             X = norm(X)
             Q = XW, K = XW, V = XW
             (b,l,h/m,q,3) = (b,l,e) * (e,3hq/m)
             A = QK'/sqrt(q)
             (b,h/m,l,l) = (b,h/m,l,q) * (b,h/m,q,l)
             A = softmax(A)
             (b,h/m,l,l) = (b,h/m,l,l)
             A = dpr(A)
             (b,h/m,l,l) = (b,h/m,l,l)
             Y = AV
             (b,h/m,l,q) = (b,h/m,l,l) * (b,h/m,l,q)
             Y = VW
             (b,l,e) = (b,l,hq/m) * (hq/m,e)
             Y = dpr(Y)
             (b,l,e) = (b,l,e)
             Y = norm(Y)
             (b,l,e) = (b,l,e)
            
    
    comments: 
    """
    summary = []
    
    flops_per_add = 1 * flops_units
    
    q = e // h
    
    m1 = parallelism['m1']
    m2 = 1 #parallelism['m2'] # 1D parallelism for now
    
    m1_parallel = (m1 > 1)
    
     ####################################################################################################
    
    stats = layer_norm_estimates(b, l // m1, e, element_size=element_size, flops_units=flops_units)
    stats["layer"] = "layer_norm_1"
    # sync/comm layers
    stats["comm_fwd"] = (m1-1)/m1 * (b * l * e) * element_size # all gather for the next op
    stats["comm_fwd_type"] = "allgather"
    stats["comm_size"] = m1
    summary.append(stats)
    
    #####################################################################################################
    
    stats = linear_estimates(b, l, e, (3*e) // m1, element_size=element_size, has_bias=False, flops_units=flops_units)
    stats["layer"] = "qkv_proj"
    # sync/comm layers: no fwd coms here
    stats["comm_bwd"] = (m1-1)/m1 * (b * l * e) * element_size # reduce scatter before going to ln: TODO check?
    stats["comm_bwd_type"] = "reducescatter"
    stats["comm_size"] = m1
    #TODO add comps for reduce-scatter
    stats["flops_bwd"] += (m1-1)/m1 * (b * l * e) * flops_per_add
    summary.append(stats)
    
  #######################################################################################################
    
    stats = logit_estimates(b, l, q, h // m1, element_size=element_size, flops_units=flops_units)
    stats["layer"] = "logits"
    summary.append(stats)
    #####################################################################################################
    
    stats = softmax_estimates(b, l, h // m1, element_size=element_size, flops_units=flops_units)
    stats["layer"] = "softmax"
    summary.append(stats)
    
  ########################################################################################################
    
    stats = dropout_estimates(b, l, (l*h) // m1, element_size=element_size, mask_element_size=mask_element_size, flops_units=flops_units)
    stats["layer"] = "dropout_softmax"
    summary.append(stats)
    
    #########################################################################################################
    
    stats = attend_estimates(b, l, q, h // m1, element_size=element_size, flops_units=flops_units)
    stats["layer"] = "attend"
    summary.append(stats)
    
     #########################################################################################################
    
    stats = linear_estimates(b, l, (h*q) // m1, e, element_size=element_size, has_bias=True, flops_units=flops_units)
    stats["layer"] = "v_proj"
    # sync/comm layers
    stats["comm_fwd"] = (m1-1)/m1 * (b * l * e) * element_size # fwd comms for partial sums of b,l,e
    stats["comm_fwd_type"] = "reducescatter"
    stats["comm_size"] = m1
    #TODO add comps for reduce-scatter
    stats["flops_fwd"] += (m1-1)/m1 * (b * l * e) * flops_per_add 
    summary.append(stats)
    
    #######################################################################################################
    
    stats = dropout_estimates(b, l // m1, e, element_size=element_size, mask_element_size=mask_element_size, flops_units=flops_units)
    stats["layer"] = "dropout"
    # sync/comm layers
    stats["comm_bwd"] = (m1-1)/m1 * (b * l * e) * element_size
    stats["comm_bwd_type"] = "allgather"
    stats["comm_size"] = m1
    summary.append(stats)
    
    ########################################################################################################
    
    stats = layer_norm_estimates(b, l // m1, e, element_size=element_size, flops_units=flops_units)
    stats["layer"] = "layer_norm_2"
    # sync/comm layers
    stats["comm_fwd"] = (m1-1)/m1 * (b * l * e) * element_size # all gather for the next op
    stats["comm_fwd_type"] = "allgather"
    stats["comm_size"] = m1
    summary.append(stats)
    
 ############################################################################################################
    
    summary = pd.DataFrame(summary)
    summary = compute_timings_and_stats(summary, system)

    
    return summary


################### 2D ####################

def MLP_estimates_2D(b, l, e, f, depth, element_size=4E-6, mask_element_size=1E-6, flops_units=1E-12, 
                  parallelism={'m1': 1, 'm2': 1}, system={}):
    """
    MLP layer estimates
    parameters: b: batch size
                l: seq length
                e: embedding dim
                f: hidden dim
                element_size: in MB
                mask_element_size: in MB (for dropout)
    
    tensor shapes: input tensor: (b,l,e)
                   output tensor: (b,l,e)
                   
    layer arithmetic: 
        forward pass: 
             X = XW + b
             (b,l,f) = (b,l,e) * (e,f) + (1,f)
             X = nonlinear(X)
             (b,l,f) = (b,l,f)
             X = dropout(X)
             (b,l,f) = (b,l,f) * (b,l,f) [random mask]
             X = linear(X)
             (b,l,e) = (b,l,f) * (f,e) + (1,e)
             X = dropout(X)
             (b,l,e) = (b,l,e) * (b,l,e) [random mask]
            
        backward pass:
             chain rule
             
    parallelism:
            X = XW + b
            (b,l/m2,f/m1) = (b,l/m2,e/m1) * (e/m1,f/m1) + (1,f/m1)
            X = nonlinear(X)
            (b,l/m2,f/m1) = (b,l/m2,f/m1)
            X = dropout(X)
            (b,l/m2,f/m1) = (b,l/m2,f/m1) * (b,l/m2,f/m1) [random mask]
            X = linear(X)
            (b,l/m2,e/m1) = (b,l/m2,f/m1) * (f/m1,e/m1) + (1,e/m1)
            X = dropout(X)
            (b,l/m2,e/m1) = (b,l/m2,e/m1) * (b,l/m2,e/m1) [random mask]
            
    comments: 
    """
    
    summary = []
    
    flops_per_add = 1 * flops_units
    
    m1 = parallelism['m1']
    m2 = parallelism['m2'] 
    
    m1_parallel = (m1 > 1)
    m2_parallel = (m2 > 1)
    
    total_time = 0
    
   ###############################################################################################
        
    stats = linear_estimates(b, l // m2, e // m1, f // m1, element_size=element_size, 
                             has_bias=True, flops_units=flops_units)
    stats["layer"] = "fc1"   
    # sync/comm layers
    stats["comm_bwd"] = (m1+m2)/(m1*m2) * (b * l * f) * element_size # bwd comms for partial sums of b,l,e
    stats["comm_bwd_type"] = "broadcast+reduce" 
    stats["comm_size"] = m1*m2
    #TODO add extra comps for reduce
    stats["flops_bwd"] += m1/(m1*m2) * (b * l * f) * flops_per_add
    
    stats["comm_fwd"] = (m1+m2)/(m1*m2) * (b * l * e) * element_size # bwd comms for partial sums of b,l,e
    stats["comm_fwd_type"] = "broadcast" 
    stats["comm_size"] = m1*m2
    summary.append(stats)
    
    #########################################################################################################
    
    stats = nonlinear_act_estimates(b, l // m2, f // m1, element_size=element_size, flops_units=flops_units)
    stats["layer"] = "act"
    summary.append(stats)
    
    ##########################################################################################################
    
    stats = dropout_estimates(b, l // m2, f // m1, element_size=element_size, mask_element_size=mask_element_size, flops_units=flops_units)
    stats["layer"] = "dpr1"
    summary.append(stats)
    
    ##########################################################################################################
    
    stats = linear_estimates(b, l // m2, f // m1, e // m1, element_size=element_size, has_bias=True, flops_units=flops_units)
    stats["layer"] = "fc2"
    # sync/comm layers
    stats["comm_fwd"] = (m1+m2)/(m1*m2) * (b * l * f) * element_size # fwd comms for partial sums of b,l,e
    stats["comm_fwd_type"] = "broadcast"
    stats["comm_size"] = m1*m2
    
    stats["comm_bwd"] = (m1+m2)/(m1*m2) * (b * l * e) * element_size
    stats["comm_bwd_type"] = "broadcast+reduce"
    stats["comm_size"] = m1
    #TODO add extra compute for reduce
    stats["flops_bwd"] += m1/(m1*m2) * (b * l * e) * flops_per_add
    summary.append(stats)
    
    ##############################################################################################################
    
    stats = dropout_estimates(b, l // m2, e // m1, element_size=element_size, mask_element_size=mask_element_size, 
                              flops_units=flops_units)
    stats["layer"] = "dpr2"
    # sync/comm layers
    
    summary.append(stats)
    
   ################################################################################################################
    
    summary = pd.DataFrame(summary)
    summary = compute_timings_and_stats(summary, system)
    
    return summary
        
    
def self_attention_estimates_2D(b, l, e, h, element_size=4E-6, mask_element_size=1E-6, flops_units=1E-12, 
                             parallelism={'m1': 1, 'm2': 1}, system={}):
    """
    dropout layer estimates
    parameters: b: batch size
                l: seq length
                e: embedding dim/hidden dim
                h: number of attention heads
                element_size: in MB
    
    tensor shapes: input tensor: (b,l,e)
                   output tensor: (b,l,e)
                   
    layer arithmetic: 
        define: q = e/h
        forward pass: 
             X = norm(X)
             Q = XW, K = XW, V = XW
             (b,l,h,q,3) = (b,l,e) * (e,3hq)
             A = QK'/sqrt(q)
             (b,h,l,l) = (b,h,l,q) * (b,h,q,l)
             A = softmax(A)
             (b,h,l,l) = (b,h,l,l)
             A = dpr(A)
             Y = AV
             (b,h,l,q) = (b,h,l,l) * (b,h,l,q)
             Y = VW
             (b,l,e) = (b,l,hq) * (hq,e)
             Y = dpr(Y)
             (b,l,e) = (b,l,e)
             Y = norm(Y)
             (b,l,e) = (b,l,e)
             
        backward pass:
             chain rule
             
        parallelism:
             X = norm(X)
             (b,l/m2,e/m1) = (b,l/m2,e/m1)
             Q = XW, K = XW, V = XW
             (b,l/m2,h,q/m1,3) = (b,l/m2,e/m1) * (e,3hq/m1)
             A = QK'/sqrt(q)
             (b,h/m1,l/m2,l/m2) = (b,h/m1,l/m2,q) * (b,h/m1,q,l/m2)
             A = softmax(A)
             (b,h/m1,l/m2,l/m2) = (b,h/m1,l/m2,l/m2)
             A = dpr(A)
             (b,h/m1,l/m2,l/m2) = (b,h/m1,l/m2,l/m2)
             Y = AV
             (b,h/m1,l/m2,q) = (b,h/m1,l/m2,l/m2) * (b,h/m1,l/m2,q)
             Y = VW
             (b,l/m2,e/m1) = (b,l/m2,hq/m1) * (hq/m1,e/m1)
             Y = dpr(Y)
             (b,l/m2,e/m1) = (b,l/m2,e/m1)
             Y = norm(Y)
             (b,l/m2,e/m1) = (b,l/m2,e/m1)
            
    
    comments: 
    """
    summary = []
    
    flops_per_add = 1 * flops_units
    
    q = e // h
    
    m1 = parallelism['m1']
    m2 = parallelism['m2'] 
    
    m1_parallel = (m1 > 1)
    m2_parallel = (m2 > 1)
    
   ###################################################################################################
    
    stats = layer_norm_estimates(b, l // m2, e // m1, element_size=element_size, flops_units=flops_units)
    stats["layer"] = "layer_norm_1"
    # sync/comm layers
    #TODO add layer norm comms and comps
    summary.append(stats)
    

    ###################################################################################################
    
    #SUMMA
    stats = linear_estimates(b, l // m2, e // m1, (3*e) // m1, element_size=element_size, has_bias=False, flops_units=flops_units)
    stats["layer"] = "qkv_proj"
    # sync/comm layers
    stats["comm_bwd"] = (m1+m2)/(m1*m2) * (b * l * 3*e) * element_size # reduce scatter before going to ln: TODO check?
    stats["comm_bwd_type"] = "broadcast+reduce"
    stats["comm_size"] = m1*m2
    #TODO add extra comps for reduce
    stats["flops_bwd"] += m1/(m1*m2) * (b * l * 3*e) * flops_per_add
    
    stats["comm_fwd"] = (m1+m2)/(m1*m2) * (b * l * e) * element_size # all gather for the next op
    stats["comm_fwd_type"] = "broadcast"
    stats["comm_size"] = m1*m2
    summary.append(stats)
    
    ######################################################################################################
    
    stats = logit_estimates(b, l // m2, q, h // m1, element_size=element_size, flops_units=flops_units)
    stats["layer"] = "logits"
    summary.append(stats)
  
    ######################################################################################################
    
    stats = softmax_estimates(b, l // m2, h // m1, element_size=element_size, flops_units=flops_units)
    stats["layer"] = "softmax"
    #coms for distributed softmax
    stats["comm_fwd"] = (m1*m2) * element_size # all gather for the next op
    stats["comm_fwd_type"] = "broadcast"
    stats["comm_size"] = m1*m2
    
    stats["comm_bwd"] = (m1*m2) * element_size # all gather for the next op
    stats["comm_bwd_type"] = "broadcast"
    stats["comm_size"] = m1*m2
    summary.append(stats)
    
    #######################################################################################################
    
    stats = dropout_estimates(b, l // m2, (h*l) // (m1*m2), element_size=element_size, mask_element_size=mask_element_size, flops_units=flops_units)
    stats["layer"] = "dropout_softmax"
    summary.append(stats)
    
   #########################################################################################################
    # SUMMA
    stats = attend_estimates(b, l // m2, q, h // m1, element_size=element_size, flops_units=flops_units)
    stats["layer"] = "attend"
    # sync/comm layers
    stats["comm_bwd"] = (m1+m2)/(m1*m2) * (b * l * h) * element_size # reduce scatter before going to ln: TODO check?
    stats["comm_bwd_type"] = "broadcast+reduce"
    stats["comm_size"] = m1*m2
    #TODO add extra comps for reduce
    stats["flops_bwd"] += m1/(m1*m2) * (b * l * h) * flops_per_add
    
    stats["comm_fwd"] = (m1+m2)/(m1*m2) * (b * l * l) * element_size # all gather for the next op
    stats["comm_fwd_type"] = "broadcast"
    stats["comm_size"] = m1*m2
    summary.append(stats)
    
    ###########################################################################################################
    
    #SUMMA
    stats = linear_estimates(b, l // m2, (h*q) // m1, e //m1, element_size=element_size, has_bias=True, flops_units=flops_units)
    stats["layer"] = "v_proj"
    # sync/comm layers
    # sync/comm layers
    stats["comm_bwd"] = (m1+m2)/(m1*m2) * (b * l * e) * element_size # reduce scatter before going to ln: TODO check?
    stats["comm_bwd_type"] = "broadcast+reduce"
    stats["comm_size"] = m1*m2
    #TODO add extra comps for reduce
    stats["flops_bwd"] += m1/(m1*m2) * (b * l * e) * flops_per_add
    
    stats["comm_fwd"] = (m1+m2)/(m1*m2) * (b * l *  h * q) * element_size # all gather for the next op
    stats["comm_fwd_type"] = "broadcast"
    stats["comm_size"] = m1*m2
    summary.append(stats)
    
    ###########################################################################################################
    
    stats = dropout_estimates(b, l // m2, e // m1, element_size=element_size, mask_element_size=mask_element_size, flops_units=flops_units)
    stats["layer"] = "dropout"
    # sync/comm layers
    summary.append(stats)
    
    ############################################################################################################
    
    stats = layer_norm_estimates(b, l // m2, e // m1, element_size=element_size, flops_units=flops_units)
    stats["layer"] = "layer_norm_2"
    # comm for layer norm
    stats["comm_fwd"] = (m1*m2) * element_size 
    stats["comm_fwd_type"] = "broadcast"
    stats["comm_size"] = m1*m2
    summary.append(stats)
    
    #############################################################################################################
    
    summary = pd.DataFrame(summary)
    summary = compute_timings_and_stats(summary, system)

    
    return summary
        