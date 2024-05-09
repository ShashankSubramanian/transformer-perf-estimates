import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from modules import *
import heapq

def factors(n):
    for c in range(1, n+1):
        if n % c == 0:
            yield c

def tp1d_candidates(n, heads, hidden):
    # candidates for tensor parallelism in 1D
    for c in factors(n):
        if c <= heads and c <= hidden:
            yield c

def tp2d_candidates_dim1(n, heads, embed):
    # candidates for tensor parallelism in 2D in 1st dim
    for c in factors(n):
        if c <= heads and c <= embed: # for m1: use heads or e/f
            yield c

def tp2d_candidates_dim2(n, tp1, sequence, embed):
    # candidates for tensor parallelism in 2D in 2nd dim
    tp2 = n // tp1 # use remaining for other dim (seq)
    for c in factors(tp2):
        if c <= sequence and c <= embed: # for m2: use sequence or e/f
            yield c

def tpseqp_candidates_dim1(n, heads, embed):
    # candidates for tensor parallelism in seqp in 1st dim
    for c in factors(n):
        if c <= heads and c <= embed: # for m1: use heads or e/f
            yield c

def tpseqp_candidates_dim2(n, tp1, sequence):
    # candidates for tensor parallelism in seqp in 2nd dim
    tp2 = n // tp1 # use remaining for other dim (seq)
    for c in factors(tp2):
        if c <= sequence and c * tp1 <= sequence: # tp1 and tp2 can be used for seq
            yield c

def nv_candidates(tp1, tp2, nvs):
    # use partial nv domains for tp1 and tp2
    tp = tp1 * tp2
    if tp <= nvs:
        yield (tp1, tp2) # all in nv domain
    else:
        for n1 in factors(nvs):
            n2 = nvs // n1
            if tp1 % n1 == 0 and tp2 % n2 == 0:
                yield (n1, n2)

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

def summa_nb_candidates(tp1, tp2, embed):
    # nb starts from max(m1, m2) until embed. could have more for hidden/seq
    # but restrict it to this for now to keep it simple
    start = max(tp1, tp2)
    end = np.log2(12288 // start)
    nb_range = (start * 2**np.arange(0, end)).astype(int)
    
    for c in nb_range:
            yield c

def totals(df_mlp, df_sa, depth, pp=1, dp=1, number_micro_batches=1, verbose=False):
    # time
    t = (df_mlp['t_fwd'].sum() + df_mlp['t_bwd'].sum() + df_sa['t_fwd'].sum() + df_sa['t_bwd'].sum()) * number_micro_batches # local_batch_size images
    t *= depth // pp # (tf + tb) * m
    bubble_time = (pp - 1) * (t / number_micro_batches)  # but not ideal
    t += bubble_time # add this
    if verbose:
        print('time for 1 itr = {}'.format(t))
    # mem
    wts = (df_mlp['weights_mem'].sum() + df_sa['weights_mem'].sum()) * (depth // pp) # not a function of batch size
    wts_grad = wts // dp
    wts_optimizer_states = 6 * (wts // dp) # 2wts for fp32 copy of weights, mom, variance
    acts = (df_mlp['activation_buffer'].sum() + df_sa['activation_buffer'].sum()) * (depth // pp) # store microbatch of acts
    # assume 1F1B
    acts *= pp # at most pp stages of acts need to be maintained for this so not multiplied by number of micro batches
    mem = wts + wts_grad + wts_optimizer_states + acts
    if verbose:
        print('mem consumed = {}'.format(mem))
    return (t, mem)

def execute_1d(model, n_gpus, global_batch_size=2048, system={}, verbose=False, nlargest=10):
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
        configs_per_n = []
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

        for (dp, tp, pp, mbs) in cands:
            m1 = tp
            t1 = m1 if tp <= nvs else nvs # topology: num gpus in nvdomain is all if nvdomain is bigger, else use complete nvdomain
            local_batch_size = global_batch_size // dp
#            b = local_batch_size
            b = mbs # time one microbatch: careful
            df_mlp = mlp_1d(b, l, e, f, parallelism={'m': m1}, topology={'t': t1},  system=system)
            df_sa = sa_1d(b, l, e, h, parallelism={'m': m1}, topology={'t': t1}, flash_attention=True, system=system)
            (t, mem) = totals(df_mlp, df_sa, depth, pp=pp, dp=dp, number_micro_batches=local_batch_size//mbs)
            throughput = global_batch_size / t
            if mem > capacity:
                continue # not feasible
            if verbose:
                print("mbs = {}, dp = {}, tp = {}, pp = {}, t = {}, tput = {}, mem = {}".format(mbs, dp, tp, pp, t, throughput, mem))
            c = {'dp': dp, 'tp': tp, 'pp': pp, 'mbs': mbs}
            configs_per_n.append((throughput, mem, c))
        configs.append(heapq.nlargest(nlargest, configs_per_n, key=lambda ky:ky[0]))

    return configs

def execute_2d(model, n_gpus, global_batch_size=2048, system={}, verbose=False, nlargest=10):
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
        configs_per_n = []
        for tp1 in tp2d_candidates_dim1(n, h, e):
            for tp2 in tp2d_candidates_dim2(n, tp1, l, e):
                tp = tp1 * tp2
                for n1, n2 in nv_candidates(tp1, tp2, nvs):
                    for pp in pp_candidates(n, tp, depth):
                        dp = get_dp(n, tp, pp)
                        if dp > global_batch_size:
                            continue
                        for micro_batch_size in micro_batch_size_candidates(global_batch_size, tp, pp, dp):
                            for nb in summa_nb_candidates(tp1, tp2, e):
                                # print("mbs = {}, dp = {}, tp1 = {}, tp2 = {}, pp = {}, prod = {}, nb = {}".format(micro_batch_size, dp, tp1, tp2, pp, pp*tp1*tp2*dp, nb))
                                c = (dp, tp1, tp2, pp, micro_batch_size, nb, n1, n2)
                                if c not in cands: # some duplicate configs due to max pipelining
                                    cands.append(c)

        for (dp, tp1, tp2, pp, mbs, nb, n1, n2) in cands:
            m1 = tp1
            m2 = tp2

            tp = tp1 * tp2
            
            # how many gpus in nvdomain
            t1 = n1
            t2 = n2
            assert t1 * t2 <= nvs, "assigned too many gpus for nv domain"

            system['summa_nb'] = nb
            local_batch_size = global_batch_size // dp
            b = mbs #local_batch_size
            df_mlp = mlp_2d(b, l, e, f, parallelism={'m1': m1, 'm2': m2}, topology={'t1': t1, 't2': t2}, system=system)
            df_sa = sa_2d_seqp(b, l, e, h, parallelism={'m1': m1, 'm2': m2}, topology={'t1': t1, 't2': t2}, flash_attention=True, system=system)
            (t, mem) = totals(df_mlp, df_sa, depth, pp=pp, dp=dp, number_micro_batches=local_batch_size//mbs)
            throughput = global_batch_size / t
            if mem > capacity:
                continue # not feasible
            if verbose:
                print("mbs = {}, dp = {}, tp1 = {}, tp2 = {}, nv1 = {}, nv2= {}, nb = {},  pp = {}, t = {}, tput = {}, mem = {}".format(mbs, dp, tp1, tp2, n1, n2, nb, pp, t, throughput, mem))
            c = {'dp': dp, 'tp': tp, 'tp1': tp1, 'tp2': tp2, 'pp': pp, 'mbs': mbs, 'nb': nb}
            configs_per_n.append((throughput, mem, c))
        configs.append(heapq.nlargest(nlargest, configs_per_n, key=lambda ky:ky[0]))
    return configs

def execute_seqp(model, n_gpus, global_batch_size=2048, system={}, verbose=False, nlargest=10):
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
        configs_per_n = []
        for tp1 in tpseqp_candidates_dim1(n, h, e):
            for tp2 in tpseqp_candidates_dim2(n, tp1, l):
                tp = tp1 * tp2
                for n1, n2 in nv_candidates(tp1, tp2, nvs):
                    for pp in pp_candidates(n, tp, depth):
                        dp = get_dp(n, tp, pp)
                        if dp > global_batch_size:
                            continue
                        for micro_batch_size in micro_batch_size_candidates(global_batch_size, tp, pp, dp):
                            c = (dp, tp1, tp2, pp, micro_batch_size, n1, n2)
                            if c not in cands: # some duplicate configs due to max pipelining
                                cands.append(c)

        for (dp, tp1, tp2, pp, mbs, n1, n2) in cands:
            m1 = tp1
            m2 = tp2

            tp = tp1 * tp2
            
            # how many gpus in nvdomain
            t1 = n1
            t2 = n2
            assert t1 * t2 <= nvs, "assigned too many gpus for nv domain"

            local_batch_size = global_batch_size // dp
            b = mbs
            df_mlp = mlp_seqp(b, l, e, f, parallelism={'m1': m1, 'm2': m2}, topology={'t1': t1, 't2': t2}, system=system)
            df_sa = sa_seqp(b, l, e, h, parallelism={'m1': m1, 'm2': m2}, topology={'t1': t1, 't2': t2}, flash_attention=True, system=system)
            (t, mem) = totals(df_mlp, df_sa, depth, pp=pp, dp=dp, number_micro_batches=local_batch_size//mbs)
            throughput = global_batch_size / t
            if mem > capacity:
                continue # not feasible
            if verbose:
                print("mbs = {}, dp = {}, tp1 = {}, tp2 = {}, nv1 = {}, nv2= {}  pp = {}, t = {}, tput = {}, mem = {}".format(mbs, dp, tp1, tp2, n1, n2, pp, t, throughput, mem))
            c = {'dp': dp, 'tp': tp, 'tp1': tp1, 'tp2': tp2, 'pp': pp, 'mbs': mbs}
            configs_per_n.append((throughput, mem, c))
        configs.append(heapq.nlargest(nlargest, configs_per_n, key=lambda ky:ky[0]))
    return configs

