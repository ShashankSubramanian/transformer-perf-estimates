import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from modules import *
import heapq
import itertools
from multiprocessing import Pool
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


''' functions to get the candidate number of GPUs assigned to each parallelism
    as well as number of GPUs assigned to the fast bandwidth (NVS) domain '''

def factors(n):
    for c in range(1, n+1):
        if n % c == 0:
            yield c

def tp1d_candidates(n, heads, embed):
    # candidates for tensor parallelism in 1D
    for c in factors(n):
        if heads % c == 0 and embed % c == 0:
            yield c

def tp2d_candidates_dim1(n, heads, embed):
    # candidates for tensor parallelism in 2D in 1st dim
    for c in factors(n):
        if heads % c == 0 and embed % c == 0:
            yield c

def tp2d_candidates_dim2(n, tp1, embed):
    # candidates for tensor parallelism in 2D in 2nd dim
    tp2 = n // tp1 # use remaining for other dim (seq)
    for c in factors(tp2):
        if  embed % c  == 0: # for m2: use sequence or e/f; sequence is checked in microbatch candidates
            yield c

def tpseqp_candidates_dim1(n, heads, embed):
    # candidates for tensor parallelism in seqp in 1st dim
    for c in factors(n):
        if heads % c == 0 and embed % c == 0:
            yield c

def tpseqp_candidates_dim2(n, tp1):
    # candidates for tensor parallelism in seqp in 2nd dim
    tp2 = n // tp1 # use remaining for other dim (seq)
    for c in factors(tp2):
#        if sequence % c == 0 and sequence % (c * tp1) == 0: # tp1 and tp2 are used for seq
        yield c # tp1 and tp2 are used for b*l, so checked in microbatch

def nv_candidates_1d(tp, dp, pp, nvs):
    ''' candidates for fast domain '''
    total = tp * dp * pp
    if total <= nvs:
        yield (tp, dp, pp) # all in nv domain
    else:
        for n1 in factors(nvs):
            rem = nvs // n1
            for n2 in factors(rem):
                n3 = rem // n2
                assert n1 * n2 * n3 == nvs, "nv size assert in 1d"
                if tp % n1 == 0 and dp % n2 == 0 and pp % n3 == 0:
                    yield (n1, n2, n3)

def nv_candidates_2d(tp1, tp2, dp, pp, nvs):
    ''' candidates for fast domain  in 2D versions of TP'''
    total = tp1 * tp2 * dp * pp
    if total <= nvs:
        yield (tp1, tp2, dp, pp) # all in nv domain
    else:
        for n1 in factors(nvs):
            rem = nvs // n1
            for n2 in factors(rem):
                rem2 = rem // n2
                for n3 in factors(rem2):
                    n4 = rem2 // n3
                    assert n1 * n2 * n3 * n4 == nvs, "nv size assert in 2d"
                    if tp1 % n1 == 0 and tp2 % n2 == 0 and dp % n3 == 0 and pp % n4 == 0:
                        yield (n1, n2, n3, n4)


def pp_candidates(n, tp, depth):
    # pipeline candidates
    max_pp = min(n // tp, depth)
    for c in factors(max_pp):
        if n % (tp * c) == 0 and depth % c == 0: # assume equal pp
            yield c

def get_dp(n, tp, pp):
    assert n % (tp * pp) == 0
    return n // (tp * pp)

def micro_batch_size_candidates(global_batch_size, tp, pp, dp, l):
    assert global_batch_size % dp == 0 # dont think this matters too much
    local_batch_size = global_batch_size // dp
    for c in factors(local_batch_size):
        bl = c * l
        if bl % tp == 0:  # bl should divide tp or tp2
            yield c

def summa_nb_candidates(tp1, tp2, embed):
    # candidates for SUMMA nb
    # nb starts from max(m1, m2) until embed. could have more for hidden/seq
    # but restrict it to this for now to keep it simple
    start = max(tp1, tp2)
    end = np.log2(embed // start)
    nb_range = (start * 2**np.arange(0, end)).astype(int)
    # print(nb_range)
    max_dim = 512
    if embed / nb_range[0] < max_dim:
        yield nb_range[0]
    else:
        for c in nb_range:
            if embed % c == 0 and embed // c >= max_dim:
                yield c

def totals(df_mlp, df_sa, df_dp, df_pp, depth, pp=1, dp=1, number_micro_batches=1, verbose=False):
    ''' total time to do forward and backward pass given all the parallelism '''
    # flops
    flops_per_gpu = (df_mlp['flops_fwd'].sum() + df_mlp['flops_bwd'].sum() + df_sa['flops_fwd'].sum() + df_sa['flops_bwd'].sum()) * (depth // pp) * number_micro_batches
    # time
    t_fwd = (df_mlp['t_fwd'].sum() + df_sa['t_fwd'].sum()) # 1 micro batch, 1 layer
    t_bwd = (df_mlp['t_bwd'].sum() + df_sa['t_bwd'].sum()) # 1 micro batch, 1 layer
    t = (t_fwd + t_bwd) * number_micro_batches  # 1 local batch, 1 layer
    t *= depth // pp # (tf + tb) * m for all local layers
    # pp comms
    t_pp_comm = df_pp['t'].sum()
    t += t_pp_comm
    # bubble time
    bubble_time = (pp - 1) * (t / number_micro_batches)  # but not ideal
#    bubble_time /= (depth // pp)
    t += bubble_time # add this
    # dp comms
    t_dp_comm = df_dp['t'].sum()
    t += t_dp_comm

    # mem
    wts_one_layer = (df_mlp['weights_mem'].sum() + df_sa['weights_mem'].sum())
    wts = wts_one_layer * (depth // pp) # not a function of batch size
    wts_grad = wts / dp
    wts_optimizer_states = 6 * (wts / dp) # 2wts for fp32 copy of weights, mom, variance
    acts = (df_mlp['activation_buffer'].sum() + df_sa['activation_buffer'].sum()) * (depth // pp) # store microbatch of acts
    # assume 1F1B
    mem_factor = pp if number_micro_batches >= pp else number_micro_batches
    acts *= mem_factor # at most pp stages of acts need to be maintained for this so not multiplied by number of micro batches
    mem = wts + wts_grad + wts_optimizer_states + acts

    # track other times
    # time comm
    t_comm = (df_mlp['t_fwd_comm'].sum() + df_mlp['t_bwd_comm'].sum() + df_sa['t_fwd_comm'].sum() + df_sa['t_bwd_comm'].sum()) * number_micro_batches # local_batch_size images
    t_comm *= depth // pp # (tf + tb) * m
    t_comp = (df_mlp['t_fwd_comp'].sum() + df_mlp['t_bwd_comp'].sum() + df_sa['t_fwd_comp'].sum() + df_sa['t_bwd_comp'].sum()) * number_micro_batches # local_batch_size images
    t_comp *= (depth // pp) # (tf + tb) * m
    t_mem = (df_mlp['t_fwd_mem'].sum() + df_mlp['t_bwd_mem'].sum() + df_sa['t_fwd_mem'].sum() + df_sa['t_bwd_mem'].sum()) * number_micro_batches # local_batch_size images
    t_mem *= (depth // pp) # (tf + tb) * m

    if verbose:
        print('time for 1 itr = {}'.format(t))

    if verbose:
        print('mem consumed = {}'.format(mem))
    stats = {'t': t,
             'flops_per_gpu': flops_per_gpu,
             'mem': mem,
             't_comp': t_comp,
             't_mem': t_mem,
             't_tp_comm': t_comm,
             't_dp_comm': t_dp_comm,
             't_pp_comm': t_pp_comm,
             't_bubble': bubble_time,
             'comp_frac': t_comp / t,
             'tp_comm_frac': t_comm / t,
             'dp_comm_frac': t_dp_comm / t,
             'pp_comm_frac': t_pp_comm / t,
             'bubble_frac': bubble_time / t,
             'mem_frac': t_mem / t,
             'wts_mem': wts,
             'wts_grad_mem': wts_grad,
             'wts_optimizer_states_mem': wts_optimizer_states,
             'acts_mem': acts}
    return (t, mem), stats

def execute_1d(model, n_gpus, global_batch_size=2048, system={}, verbose=False, nlargest=10):
    ''' run solver to do the 1D TP search '''
    configs = []

    l = model['l']
    e = model['e']
    f = model['f']
    h = model['h']
    depth = model['depth']
    capacity = system['hbm_capacity']
    nvs = system['nvlink_size']

    for n in n_gpus:
        cands = set()
        configs_per_n = []

        for tp in tp1d_candidates(n, h, e):
            if (3*e) // tp < 128:
                continue

            for pp in pp_candidates(n, tp, depth):
                dp = get_dp(n, tp, pp)
                if dp > global_batch_size or global_batch_size % dp != 0:
                    continue

                for micro_batch_size in micro_batch_size_candidates(global_batch_size, tp, pp, dp, l):
                    for nv1, nv2, nv3 in nv_candidates_1d(tp, dp, pp, nvs):
                        c = (dp, tp, pp, micro_batch_size, nv1, nv2, nv3)
                        if c not in cands:
                            cands.add(c)

        
        print('num gpus = {}, nvs domain size = {}, #possible candidates = {}'.format(n, nvs, len(cands)))
        for (dp, tp, pp, mbs, nv1, nv2, nv3) in cands:
            m1 = tp
            t1 = nv1
            t_dp = nv2
            t_pp = nv3 
            local_batch_size = global_batch_size // dp
#            b = local_batch_size
            b = mbs # time one microbatch: careful
            df_mlp = mlp_1d(b, l, e, f, parallelism={'m': m1}, topology={'t': t1},  system=system)
            df_sa = sa_1d(b, l, e, h, parallelism={'m': m1}, topology={'t': t1}, flash_attention=True, system=system)
             
            # dp comms
            # since dp is the only other comms, try to use nv domain for this too
            df_dp = dataparallel(modules=[df_mlp, df_sa], depth=(depth//pp), dp=dp, t_dp=t_dp, overlap=True, system=system)

            # pp comms
            number_micro_batches = local_batch_size//mbs
            # only communicate the last layer activations = first layer's (ln1) input buffer
            p2p_comm_vol = float(df_mlp.loc[df_mlp['name'] == 'ln1']['activation_buffer'])
            df_pp = pipelineparallel(modules=[df_mlp, df_sa], number_micro_batches=number_micro_batches, comm_vol=p2p_comm_vol, pp=pp, t_pp=t_pp, overlap=False, system=system)

            # total time
            (t, mem), stats = totals(df_mlp, df_sa, df_dp, df_pp, depth, pp=pp, dp=dp, number_micro_batches=number_micro_batches)
            stats['nv_tp'] = t1
            stats['nv_dp'] = t_dp
            stats['nv_pp'] = t_pp

            throughput = global_batch_size / t
            if mem > capacity:
                continue # not feasible
            if verbose:
                print("mbs = {}, dp = {}, tp = {}, pp = {}, t = {}, tput = {}, mem = {}".format(mbs, dp, tp, pp, t, throughput, mem))
            c = {'dp': dp, 'tp': tp, 'pp': pp, 'mbs': mbs}
            configs_per_n.append((throughput, mem, c, stats))
        configs.append(heapq.nlargest(nlargest, configs_per_n, key=lambda ky:ky[0]))

    return configs

def candidate_filter_2d(args):
    tp1, tp2, tp, global_batch_size, depth, n, h, e, l, nvs = args
    filtered_cands = []
    
    if e // tp1 < 128 or l // tp2 < 128:
        return filtered_cands
    
    for pp in pp_candidates(n, tp, depth):
        dp = get_dp(n, tp, pp)
        if dp > global_batch_size or global_batch_size % dp != 0:
            continue
        for micro_batch_size in micro_batch_size_candidates(global_batch_size, tp2, pp, dp, l): # tp2 is used to shard bl
            for nb in summa_nb_candidates(tp1, tp2, e):
                for nv1, nv2, nv3, nv4 in nv_candidates_2d(tp1, tp2, dp, pp, nvs):
                    c = (dp, tp1, tp2, pp, micro_batch_size, nb, nv1, nv2, nv3, nv4)
                    filtered_cands.append(c)
    return filtered_cands

def execute_2d(model, n_gpus, global_batch_size=2048, system={}, verbose=False, nlargest=10):
    ''' run solver to do the 2D TP (SUMMA) search '''
    configs = []

    l = model['l']
    e = model['e']
    f = model['f']
    h = model['h']
    depth = model['depth']
    capacity = system['hbm_capacity']
    nvs = system['nvlink_size']

    for n in n_gpus:
        cands = set()
        configs_per_n = []
        
        tp1_candidates = list(tp2d_candidates_dim1(n, h, e))
        tp2_candidates = [(tp1, tp2) for tp1 in tp1_candidates for tp2 in tp2d_candidates_dim2(n, tp1, e)]
        
        args = [(tp1, tp2, tp1 * tp2, global_batch_size, depth, n, h, e, l, nvs) for tp1, tp2 in tp2_candidates]
        
        with Pool() as pool:
            results = pool.map(candidate_filter_2d, args)
        
        for res in results:
            for c in res:
                cands.add(c)
    
        cands = list(cands)
        print('num gpus = {}, nvs domain size = {}, #possible candidates = {}'.format(n, nvs, len(cands)))

        for (dp, tp1, tp2, pp, mbs, nb, nv1, nv2, nv3, nv4) in cands:
            m1 = tp1
            m2 = tp2

            tp = tp1 * tp2
            
            # how many gpus in nvdomain
            t1 = nv1
            t2 = nv2
            t_dp = nv3
            t_pp = nv4

            system['summa_nb'] = nb
            local_batch_size = global_batch_size // dp
            b = mbs #local_batch_size
            df_mlp = mlp_2d(b, l, e, f, parallelism={'m1': m1, 'm2': m2}, topology={'t1': t1, 't2': t2}, system=system)
            df_sa = sa_2d_seqp(b, l, e, h, parallelism={'m1': m1, 'm2': m2}, topology={'t1': t1, 't2': t2}, flash_attention=True, system=system)
            # dp comms
            # since dp is the only other comms, try to use nv domain for this too
            df_dp = dataparallel(modules=[df_mlp, df_sa], depth=(depth//pp), dp=dp, t_dp=t_dp, overlap=True, system=system)

            # pp comms
            number_micro_batches = local_batch_size//mbs
            # only communicate the last layer activations = first layer's (ln1) input buffer
            p2p_comm_vol = float(df_mlp.loc[df_mlp['name'] == 'ln1']['activation_buffer'])
            df_pp = pipelineparallel(modules=[df_mlp, df_sa], number_micro_batches=number_micro_batches, comm_vol=p2p_comm_vol, pp=pp, t_pp=t_pp, overlap=False, system=system)

            # total time
            (t, mem), stats = totals(df_mlp, df_sa, df_dp, df_pp, depth, pp=pp, dp=dp, number_micro_batches=number_micro_batches)
            stats['nv_tp1'] = t1
            stats['nv_tp2'] = t2
            stats['nv_dp'] = t_dp
            stats['nv_pp'] = t_pp
            throughput = global_batch_size / t
            if mem > capacity:
                continue # not feasible
            c = {'dp': dp, 'tp': tp, 'tp1': tp1, 'tp2': tp2, 'pp': pp, 'mbs': mbs, 'nb': nb}
            configs_per_n.append((throughput, mem, c, stats))
        configs.append(heapq.nlargest(nlargest, configs_per_n, key=lambda ky:ky[0]))
    return configs

def candidate_filter_seqp(args):
    tp1, tp2, tp, global_batch_size, depth, n, h, e, l, nvs = args
    filtered_cands = []
    
    if e // tp1 < 128 or l // tp2 < 128:
        return filtered_cands
    
    for pp in pp_candidates(n, tp, depth):
        dp = get_dp(n, tp, pp)
        if dp > global_batch_size or global_batch_size % dp != 0:
            continue
        for micro_batch_size in micro_batch_size_candidates(global_batch_size, tp, pp, dp, l): # tp1*tp2 is used to shard bl
            for nv1, nv2, nv3, nv4 in nv_candidates_2d(tp1, tp2, dp, pp, nvs):
                c = (dp, tp1, tp2, pp, micro_batch_size, nv1, nv2, nv3, nv4)
                filtered_cands.append(c)
    return filtered_cands

def execute_seqp(model, n_gpus, global_batch_size=2048, system={}, verbose=False, nlargest=10):
    ''' run solver to do the 2D TP (context parallel or sequence parallel) search '''
    configs = []

    l = model['l']
    e = model['e']
    f = model['f']
    h = model['h']
    depth = model['depth']
    capacity = system['hbm_capacity']
    nvs = system['nvlink_size']

    for n in n_gpus:
        cands = set()
        configs_per_n = []
        
        tp1_candidates = list(tpseqp_candidates_dim1(n, h, e))
        tp2_candidates = [(tp1, tp2) for tp1 in tp1_candidates for tp2 in tpseqp_candidates_dim2(n, tp1)]
        
        args = [(tp1, tp2, tp1 * tp2, global_batch_size, depth, n, h, e, l, nvs) for tp1, tp2 in tp2_candidates]
        
        with Pool() as pool:
            results = pool.map(candidate_filter_seqp, args)
        
        for res in results:
            for c in res:
                cands.add(c)
    
        cands = list(cands)
        print('num gpus = {}, nvs domain size = {}, #possible candidates = {}'.format(n, nvs, len(cands)))

        for (dp, tp1, tp2, pp, mbs, nv1, nv2, nv3, nv4) in cands:
            m1 = tp1
            m2 = tp2

            tp = tp1 * tp2
            
            # how many gpus in nvdomain
            t1 = nv1
            t2 = nv2
            t_dp = nv3
            t_pp = nv4

            local_batch_size = global_batch_size // dp
            b = mbs #local_batch_size
            df_mlp = mlp_seqp(b, l, e, f, parallelism={'m1': m1, 'm2': m2}, topology={'t1': t1, 't2': t2}, system=system)
            df_sa = sa_seqp(b, l, e, h, parallelism={'m1': m1, 'm2': m2}, topology={'t1': t1, 't2': t2}, flash_attention=True, system=system)
            # dp comms
            # since dp is the only other comms, try to use nv domain for this too
            # additional seqp data parallel comms here
            df_dp = dataparallel(modules=[df_mlp, df_sa], depth=(depth//pp), dp=dp*tp2, t_dp=t_dp*t2, overlap=True, system=system)

            # pp comms
            number_micro_batches = local_batch_size//mbs
            # only communicate the last layer activations = first layer's (ln1) input buffer
            p2p_comm_vol = float(df_mlp.loc[df_mlp['name'] == 'ln1']['activation_buffer'])
            df_pp = pipelineparallel(modules=[df_mlp, df_sa], number_micro_batches=number_micro_batches, comm_vol=p2p_comm_vol, pp=pp, t_pp=t_pp, overlap=False, system=system)

            # total time
            (t, mem), stats = totals(df_mlp, df_sa, df_dp, df_pp, depth, pp=pp, dp=dp, number_micro_batches=number_micro_batches)
            stats['nv_tp1'] = t1
            stats['nv_tp2'] = t2
            stats['nv_dp'] = t_dp
            stats['nv_pp'] = t_pp
            throughput = global_batch_size / t
            if mem > capacity:
                continue # not feasible
            c = {'dp': dp, 'tp': tp, 'tp1': tp1, 'tp2': tp2, 'pp': pp, 'mbs': mbs}
            configs_per_n.append((throughput, mem, c, stats))
        configs.append(heapq.nlargest(nlargest, configs_per_n, key=lambda ky:ky[0]))
    return configs

