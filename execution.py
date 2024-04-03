import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from modules import *

def factors(n):
    for c in range(1, n+1):
        if n % c == 0:
            yield c

def tp1d_candidates(n, heads, hidden):
    # candidates for tensor parallelism in 1D
    for c in factors(n):
        if c <= heads and c <= hidden:
            yield c

def tp2d_candidates(n, tp1d, heads, hidden, sequence):
    # candidates for tensor parallelism in 2D (this includes the sequence dim)
    # tp1d 
    for c in factors(n):
        if c <= heads and c <= hidden:
            yield c

def pp_candidates(n, tp, depth):
    max_pp = min(n // tp, depth)
    for c in factors(max_pp):
        if n % (tp * c) == 0 and depth % c == 0: # assume equal pp
            yield c

def get_dp(n, tp, pp):
    assert n % (tp * pp) == 0
    return n // (tp * pp)

def micro_batch_size_candidates(global_batch_size, tp, pp, dp):
    assert global_batch_size % dp == 0 # dont think this matters too much
    local_batch_size = global_batch_size // dp
    if pp == 1:
        yield local_batch_size
    else:
        for c in factors(local_batch_size):
            yield c

#def candidates(n, tp, depth):
#    p = n // tp # pipeline
#    if p > depth: # but cant pipeline more than depth
#        tp_cand = n // depth # pipeline as much as you can
#        p = n // tp_cand
#    return (p, n // p)

def totals(df_mlp, df_sa, depth, pp=1, dp=1, number_micro_batches=1, verbose=False):
    # time
    t = (df_mlp['t_fwd'].sum() + df_mlp['t_bwd'].sum() + df_sa['t_fwd'].sum() + df_sa['t_bwd'].sum()) # local_batch_size images
    t *= depth // pp # tf + tb
    bubble_time = (pp - 1) * (t / number_micro_batches)  # but not ideal
    t += bubble_time # add this
    if verbose:
        print('time for 1 itr = {}'.format(t))
    # mem
    wts = (df_mlp['weights_mem'].sum() + df_sa['weights_mem'].sum()) * (depth // pp) # not a function of batch size
#    if pp == 96:
#        print("wts = {}".format(wts))
    wts_grad = wts // dp
    wts_optimizer_states = 6 * (wts // dp) # 2wts for fp32 copy of weights, mom, variance
    acts = (df_mlp['activation_buffer'].sum() + df_sa['activation_buffer'].sum()) * (depth // pp) # store localbatch of acts
    # assume 1F1B
    acts = (acts / number_micro_batches) # for one microbatch
    acts *= pp # at most pp stages of acts need to be maintained for this
    mem = wts + wts_grad + wts_optimizer_states + acts
#    if pp == 96:
#        print("mem = {}, wts_grad = {}, wts_opt = {}, acts = {}".format(mem, wts_grad, wts_optimizer_states, acts_2))
    if verbose:
        print('mem consumed = {}'.format(mem))
    return (t, mem)

def execute_1d(model, n_gpus, global_batch_size=2048, system={}, verbose=False):
    configs = []

    l = model['l']
    e = model['e']
    f = model['f']
    h = model['h']
    depth = model['depth']
    capacity = system['hbm_capacity']
    nvs = system['nvlink_size']

    for n in n_gpus:
        cands = []
        for tp in tp1d_candidates(n, h, f):
            for pp in pp_candidates(n, tp, depth):
                dp = get_dp(n, tp, pp)
                if dp > global_batch_size:
                    continue
                for micro_batch_size in micro_batch_size_candidates(global_batch_size, tp, pp, dp):
#                    print("mbs = {}, dp = {}, tp = {}, pp = {}".format(micro_batch_size, dp, tp, pp))
                    c = (dp, tp, pp, micro_batch_size)
                    if c not in cands: # some duplicate configs due to max pipelining
                        cands.append(c)

#        best_time = np.inf
        best_throughput = 0        
        best_mem = np.inf
        best_config = []
        for (dp, tp, pp, mbs) in cands: # each candidate is (pp, tp)
            m1 = tp
            t1 = m1 if tp <= nvs else nvs # topology: num gpus in nvdomain is all if nvdomain is bigger, else use complete nvdomain
            local_batch_size = global_batch_size // dp
            b = local_batch_size
            df_mlp = mlp_1d(b, l, e, f, parallelism={'m': m1}, topology={'t': t1},  system=system)
            df_sa = sa_1d(b, l, e, h, parallelism={'m': m1}, topology={'t': t1}, flash_attention=True, system=system)
            (t, mem) = totals(df_mlp, df_sa, depth, pp=pp, dp=dp, number_micro_batches=local_batch_size//mbs)
            throughput = global_batch_size / t
            if mem > capacity:
                continue # not feasible
            if verbose:
                print("mbs = {}, dp = {}, tp = {}, pp = {}, t = {}, tput = {}, mem = {}".format(mbs, dp, tp, pp, t, throughput, mem))
#            if t < best_time:
            if throughput > best_throughput:
                best_throughput = throughput
                best_mem = mem
                best_config = {'dp': dp, 'tp': tp, 'pp': pp, 'mbs': mbs}
        configs.append((best_config, best_throughput, best_mem))
    return configs

