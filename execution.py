import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from modules import *

def factors(n):
    for c in range(1, n+1):
        if n % c == 0:
            yield c

def tp_1d(n, heads, hidden):
    # candidates for tensor parallelism in 1D
    for c in factors(n):
        if c <= heads and c <= hidden:
            yield c

def candidates(n, tp, depth):
    p = n // tp # pipeline
    if p > depth: # but cant pipeline more than depth
        tp_cand = n // depth # pipeline as much as you can
        p = n // tp_cand
    return (p, n // p)

def totals(df_mlp, df_sa, depth, pp=1, dp=1, verbose=False):
    # time
    t = (df_mlp['t_fwd'].sum() + df_mlp['t_bwd'].sum() + df_sa['t_fwd'].sum() + df_sa['t_bwd'].sum())
    t *= depth // pp
    bubble_time = (pp - 1) * t
    t += bubble_time
    if verbose:
        print('time for 1 itr = {}'.format(t))
    # mem
    wts = (df_mlp['weights_mem'].sum() + df_sa['weights_mem'].sum()) * (depth // pp)
    wts_grad = wts // dp
    wts_optimizer_states = 6 * (wts // dp) # 2wts for fp32 copy of weights, mom, variance
    acts = (df_mlp['activation_buffer'].sum() + df_sa['activation_buffer'].sum()) * (depth // pp)
    mem = wts + wts_grad + wts_optimizer_states + acts
    if verbose:
        print('mem consumed = {}'.format(mem))
    return (t, mem)

def execute_1d(model, n_gpus, dp=256, microbatch=1, system={}, best_time_or_mem="time"):
    configs = []
    b = microbatch
    for n in n_gpus:
        cands = []
        for t in tp_1d(n, model['h'], model['f']):
            c = candidates(n, t, model['depth'])
            if c not in cands: # some duplicate configs due to max pipelining
                cands.append(c)
        best_time = 0
        best_mem = np.inf
        best_config = []
        for (pp, tp) in cands: # each candidate is (pp, tp)
            m1 = tp
            t1 = 'nvlink' if tp <= system['nvlink_size'] else 'ib'
            df_mlp = mlp_1d(b, l, e, f, depth, parallelism={'m': m1}, topology={'t': t1})
            df_sa = sa_1d(b, l, e, h, depth, parallelism={'m': m1}, topology={'t': t1}, flash_attention=True)
            (t, mem) = totals(df_mlp, df_sa, model['depth'], pp=pp, dp=dp)
            if best_time_or_mem == "time":
                if t < best_time:
                    best_time = t
                    best_mem = mem
                    best_config = (pp, tp)
            else:
                if mem < best_mem:
                    best_time = t
                    best_mem = mem
                    best_config = (pp, tp)
        configs.append((best_config, best_time, best_mem))
    return configs


def plot(n_gpus, system, axs, lgnd=['MLP', 'SA'], lgnd_tot=['nvlink1'], lfmt="-"):
    t_mlp = []
    t_sa = []
    t_itr = []

    for n in n_gpus:
        m1 = n
        t1 = 'nvlink' if n <= system['nvlink_size'] else 'ib'
        
        df_mlp = mlp_1d(b, l, e, f, depth, parallelism={'m': m1}, topology={'t': t1})
        df_sa = sa_1d(b, l, e, h, depth, parallelism={'m': m1}, topology={'t': t1}, flash_attention=True)

        t_mlp_ = (df_mlp['t_fwd'].sum() + df_mlp['t_bwd'].sum()) * depth
        t_sa_ = (df_sa['t_fwd'].sum() + df_sa['t_bwd'].sum()) * depth
        t_itr.append(t_mlp_ + t_sa_)
        t_mlp.append(t_mlp_)
        t_sa.append(t_sa_)

    
    ax = axs[0]
    ax.plot(n_gpus, t_mlp, lfmt, linewidth=2, c=c1)
    ax.plot(n_gpus, t_sa, lfmt, linewidth=2, c=c2)
    ax.set_yscale('log')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Number of GPUs', fontsize=fsz)
    ax.set_xticks(n_gpus)
    ax.set_xticklabels(n_gpus, fontsize=fsz-4)
    ax.set_ylabel('Time', fontsize=fsz)    
    ax.legend(lgnd, fontsize=fsz-4)
    
    ax = axs[1]
    ax.plot(n_gpus, t_itr, lfmt, linewidth=2)
    ax.set_yscale('log')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Number of GPUs', fontsize=fsz)
    ax.set_xticks(n_gpus)
    ax.set_xticklabels(n_gpus, fontsize=fsz-4)
    ax.set_ylabel('Total time', fontsize=fsz)
    ax.legend(lgnd_tot, fontsize=fsz-4)
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%d"))
