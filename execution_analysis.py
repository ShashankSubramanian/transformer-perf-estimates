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

def module_stats(df,precision=16):
    coms={}
    for index ,row in df[['comm_fwd','comm_fwd_type']].iterrows():
        com_size, com_type = row['comm_fwd'], row['comm_fwd_type']
        if com_size>0 and com_type != 'none':
            if com_type not in coms:
                coms[com_type]=[np.round(com_size,precision)]
            else:
                coms[com_type].append(np.round(com_size,precision))
                
    for index ,row in df[['comm_bwd','comm_bwd_type']].iterrows():
        com_size, com_type = row['comm_bwd'], row['comm_bwd_type']
        if com_size>0 and com_type != 'none':
            if com_type not in coms:
                coms[com_type]=[np.round(com_size,precision)]
            else:
                coms[com_type].append(np.round(com_size,precision))
    com_stats=[]
    for key in coms:
        # com_stats.append([key,np.quantile(coms[key],0.25),np.quantile(coms[key],0.5),np.quantile(coms[key],0.75)])
        com_stats.append([key,coms[key]])
    return com_stats



def totals(df_mlp, df_sa, depth, pp=1, dp=1, number_micro_batches=1, verbose=False):
    # time
    # total time
    t = (df_mlp['t_fwd'].sum() + df_mlp['t_bwd'].sum() + df_sa['t_fwd'].sum() + df_sa['t_bwd'].sum()) * number_micro_batches # local_batch_size images
    t *= (depth // pp) # (tf + tb) * m
    bubble_time = (pp - 1) * (t / number_micro_batches)  # but not ideal
    t += bubble_time # add this
    if verbose:
        print('time for 1 itr = {}'.format(t))
    # time comm
    t_comm = (df_mlp['t_fwd_comm'].sum() + df_mlp['t_bwd_comm'].sum() + df_sa['t_fwd_comm'].sum() + df_sa['t_bwd_comm'].sum()) * number_micro_batches # local_batch_size images
    t_comm *= depth // pp # (tf + tb) * m
    bubble_time_comm = (pp - 1) * (t_comm / number_micro_batches)  # but not ideal
    t_comm += bubble_time_comm # add this
    
    # time comp
    t_comp = (df_mlp['t_fwd_comp'].sum() + df_mlp['t_bwd_comp'].sum() + df_sa['t_fwd_comp'].sum() + df_sa['t_bwd_comp'].sum()) * number_micro_batches # local_batch_size images
    t_comp *= (depth // pp) # (tf + tb) * m
    bubble_time_comp = (pp - 1) * (t_comp / number_micro_batches)  # but not ideal
    t_comp += bubble_time_comp # add this

    # time mem
    t_mem = (df_mlp['t_fwd_mem'].sum() + df_mlp['t_bwd_mem'].sum() + df_sa['t_fwd_mem'].sum() + df_sa['t_bwd_mem'].sum()) * number_micro_batches # local_batch_size images
    t_mem *= (depth // pp) # (tf + tb) * m
    bubble_time_mem = (pp - 1) * (t_mem / number_micro_batches)  # but not ideal
    t_mem += bubble_time_mem # add this
    
    # mem
    wts = (df_mlp['weights_mem'].sum() + df_sa['weights_mem'].sum()) * (depth // pp) # not a function of batch size
    wts_grad = (wts // dp)
    wts_optimizer_states = 6 * (wts // dp) # 2wts for fp32 copy of weights, mom, variance
    acts = (df_mlp['activation_buffer'].sum() + df_sa['activation_buffer'].sum()) * (depth // pp) # store microbatch of acts
    # assume 1F1B
    acts *= pp # at most pp stages of acts need to be maintained for this so not multiplied by number of micro batches
    mem = wts + wts_grad + wts_optimizer_states + acts

    # comms
    # fwd
    df_mlp['comm_fwd_per_gpu']=df_mlp.apply(lambda x: np.array(x['comm_fwd'])/(np.array(x['comm_fwd_size']) \
                                                                +1e-16*np.ones_like(x['comm_fwd_size'])),axis=1)
    df_mlp['comm_fwd_per_gpu']=df_mlp.apply(lambda x: np.sum(x['comm_fwd_per_gpu']), axis=1)
    comm_fwd = df_mlp['comm_fwd_per_gpu'].sum()

    df_sa['comm_fwd_per_gpu']=df_sa.apply(lambda x: np.array(x['comm_fwd'])/(np.array(x['comm_fwd_size']) \
                                                                +1e-16*np.ones_like(x['comm_fwd_size'])),axis=1)
    df_sa['comm_fwd_per_gpu']=df_sa.apply(lambda x: np.sum(x['comm_fwd_per_gpu']), axis=1)
    comm_fwd += df_sa['comm_fwd_per_gpu'].sum()

    comm_fwd *= (depth // pp) # communication volume per gpu fwd

    # bwd
    df_mlp['comm_bwd_per_gpu']=df_mlp.apply(lambda x: np.array(x['comm_bwd'])/(np.array(x['comm_bwd_size']) \
                                                                +1e-16*np.ones_like(x['comm_bwd_size'])),axis=1)
    df_mlp['comm_bwd_per_gpu']=df_mlp.apply(lambda x: np.sum(x['comm_bwd_per_gpu']), axis=1)
    comm_bwd = df_mlp['comm_bwd_per_gpu'].sum()

    df_sa['comm_bwd_per_gpu']=df_sa.apply(lambda x: np.array(x['comm_bwd'])/(np.array(x['comm_bwd_size']) \
                                                                +1e-16*np.ones_like(x['comm_bwd_size'])),axis=1)
    df_sa['comm_bwd_per_gpu']=df_sa.apply(lambda x: np.sum(x['comm_bwd_per_gpu']), axis=1)
    comm_bwd += df_sa['comm_bwd_per_gpu'].sum()

    comm_bwd *= (depth // pp) # communication volume per gpu bwd

    # comps
    flops_fwd = (df_mlp['flops_fwd'].sum() + df_sa['flops_fwd'].sum()) * (depth // pp)
    flops_bwd = (df_mlp['flops_bwd'].sum() + df_sa['flops_bwd'].sum()) * (depth // pp)
    
    if verbose:
        print('mem consumed = {}'.format(mem))
    return (t, t_comm, t_mem, t_comp, mem,  wts, wts_grad, wts_optimizer_states, acts, comm_fwd, comm_bwd, flops_fwd, flops_bwd)

def execute_1d(model, n_gpus, global_batch_size=2048, system={}, verbose=True, nlargest=10):
    configs = {}

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
        if verbose:
            print('n = '+str(n)+', cands = '+str(len(cands)))
        for (dp, tp, pp, mbs) in cands:
            m1 = tp
            t1 = m1 if tp <= nvs else nvs # topology: num gpus in nvdomain is all if nvdomain is bigger, else use complete nvdomain
            local_batch_size = global_batch_size // dp
#            b = local_batch_size
            b = mbs # time one microbatch: careful
            df_mlp = mlp_1d(b, l, e, f, parallelism={'m': m1}, topology={'t': t1},  system=system)
            df_sa = sa_1d(b, l, e, h, parallelism={'m': m1}, topology={'t': t1}, flash_attention=True, system=system)
            (tot_time, t_comm, t_mem, t_comp, mem,  wts, wts_grad, wts_optimizer_states, acts, comm_fwd, comm_bwd, flops_fwd, flops_bwd) = totals(df_mlp, df_sa, depth, pp=pp, dp=dp, number_micro_batches=local_batch_size//mbs)
            throughput = global_batch_size / tot_time
            if mem > capacity:
                continue # not feasible
            if verbose:
                print("mbs = {}, dp = {}, tp = {}, pp = {}, t = {}, tput = {}, mem = {}".format(mbs, dp, tp, pp, tot_time, throughput, mem))
            c = {'dp': dp, 'tp': tp, 'pp': pp, 'mbs': mbs}
            stats = {'t': tot_time, 't_com': t_comm, 't_mem': t_mem, 't_comp': t_comp, 
                     'mem': mem, 'wts': wts, 'wts_grad': wts_grad, 'wts_optimizer_states': wts_optimizer_states, 
                     'acts': acts, 'comm_fwd': comm_fwd, 'comm_bwd': comm_bwd, 'flops_fwd': flops_fwd, 'flops_bwd': flops_bwd}
            modules = {'mpl': df_mlp, 'sa': df_sa}
            configs_per_n.append((throughput, stats, modules, c))
        tmp_config=heapq.nlargest(nlargest, configs_per_n, key=lambda ky:ky[0])
        
        if len(tmp_config):
            configs[n]=tmp_config

    return configs 

def execute_2d(model, n_gpus, global_batch_size=2048, system={}, verbose=False, nlargest=10):
    configs = {}

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
            (tot_time, t_comm, t_mem, t_comp, mem,  wts, wts_grad, wts_optimizer_states, acts, comm_fwd, comm_bwd, flops_fwd, flops_bwd) = totals(df_mlp, df_sa, depth, pp=pp, dp=dp, number_micro_batches=local_batch_size//mbs)
            throughput = global_batch_size / tot_time
            if mem > capacity:
                continue # not feasible
            if verbose:
                print("mbs = {}, dp = {}, tp1 = {}, tp2 = {}, nv1 = {}, nv2= {}, nb = {},  pp = {}, t = {}, tput = {}, mem = {}".format(mbs, dp, tp1, tp2, n1, n2, nb, pp, ttot_time, throughput, mem))
            c = {'dp': dp, 'tp': tp, 'tp1': tp1, 'tp2': tp2, 'n1': n1, 'n2': n2,'pp': pp, 'mbs': mbs, 'nb': nb}
            stats = {'t': tot_time, 't_com': t_comm, 't_mem': t_mem, 't_comp': t_comp, 
                     'mem': mem, 'wts': wts, 'wts_grad': wts_grad, 'wts_optimizer_states': wts_optimizer_states, 
                     'acts': acts, 'comm_fwd': comm_fwd, 'comm_bwd': comm_bwd, 'flops_fwd': flops_fwd, 'flops_bwd': flops_bwd}
            modules = {'mpl': df_mlp, 'sa': df_sa}
            configs_per_n.append((throughput, stats, modules, c))
        tmp_config=heapq.nlargest(nlargest, configs_per_n, key=lambda ky:ky[0])
        
        if len(tmp_config):
            configs[n]=tmp_config

    return configs


