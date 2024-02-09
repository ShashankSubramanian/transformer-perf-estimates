import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from parallelism import MLP_estimates_1D, self_attention_estimates_1D, MLP_estimates_2D, self_attention_estimates_2D
from time_projections import get_time_flops, get_time_mem, get_time_comm, get_topology, get_total_time



def compute_stats(n_gpus, parallelism, model, system):
    
    b = model['b']
    l = model['l']
    e = model['e']
    f = model['f']
    depth=model['depth']
    h = model['h']
    element_size = model['element_size'] #fp16
    mask_element = model['mask_element'] # int
    flops_units = model['flops_units']
    fp16_sz = model['fp16_sz']
    int_sz = model['int_sz']
    

    time_mlp_total = []
    time_sa_total = []

    time_mlp_comm = []
    time_sa_comm = []
    
    total_memory_fwd = []
    total_memory_bwd = []
    
    parallel={'m1': 1, 'm2': 1}
    
    if parallelism == "1D":
        MLP_estimates = MLP_estimates_1D
        self_attention_estimates = self_attention_estimates_1D
    elif parallelism == "2D":
        MLP_estimates = MLP_estimates_2D
        self_attention_estimates = self_attention_estimates_2D
        

    for n in n_gpus:
        if parallelism == "1D":
            parallel['m1'] = n
            parallel['m2'] = 1
        elif parallelism == "2D":
            parallel['m1'] = int(np.sqrt(n))
            parallel['m2'] = int(np.sqrt(n))

        df_mlp = MLP_estimates(b, l, e, f, depth, element_size=element_size, mask_element_size=mask_element, 
                               flops_units=flops_units, 
                               parallelism=parallel, system=system)
        df_sa = self_attention_estimates(b, l, e, h, element_size=element_size, mask_element_size=mask_element, 
                                         flops_units=flops_units, 
                                     parallelism=parallel, system=system)
        t_f = df_mlp['t_total_fwd'].sum()
        t_c = df_mlp['t_comm_fwd'].sum()
        time_mlp_total.append(t_f)
        time_mlp_comm.append(t_c)


        t_f = df_sa['t_total_fwd'].sum()
        t_c = df_sa['t_comm_fwd'].sum()
        time_sa_total.append(t_f)
        time_sa_comm.append(t_c)

        
        total_memory_fwd.append(df_mlp["total_mem_fwd"].sum()+df_sa["total_mem_fwd"].sum())
        total_memory_bwd.append(df_mlp["total_mem_bwd"].sum()+df_sa["total_mem_bwd"].sum())
        
    
    return np.array(time_mlp_total), np.array(time_sa_total), np.array(time_mlp_comm), np.array(time_sa_comm), np.array(total_memory_fwd), np.array(total_memory_bwd)
        
        


    

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seq_len_min", default=2048, type=int, help="min sequence length")
    parser.add_argument("--seq_len_max", default=64800, type=int, help="max sequence length")
    parser.add_argument("--ngpus_min", default=4, type=int, help="min num gpus")
    parser.add_argument("--ngpus_max", default=2500, type=int, help="max num gpus")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--depth", default=96, type=int, help="number of layers")
    parser.add_argument("--nattn", default=96, type=int, help="number of attention heads")
    parser.add_argument("--embed_dim", default=12288, type=int, help="embedding dimension")
    parser.add_argument("--hidden_dim_mlp", default=4*12288, type=int, help="hidden dimension for mlp")
    parser.add_argument("--element_size", default=2E-9, type=float, help="size of weight element")
    parser.add_argument("--mask_element", default=1E-9, type=float, help="size of mask element")
    
    parser.add_argument("--fp32_size", default=4E-9, type=float, help="size of fp32")
    parser.add_argument("--fp16_size", default=2E-9, type=float, help="size of fp16")
    parser.add_argument("--int_size", default=1E-9, type=float, help="size of int")
    parser.add_argument("--flops_units", default=1E-12, type=float, help="size of flop units")
    
    
    parser.add_argument("--matrix_flops_fp16", default=312, type=float, help="tbd")
    parser.add_argument("--vector_flops_fp32", default=19.5, type=int, help="tbd")
    parser.add_argument("--vector_flops_fp16", default=78, type=int, help="tbd")
    parser.add_argument("--hbm_bandwidth", default=1555, type=int, help="max num gpus")
    parser.add_argument("--nvlink_bandwidth", default=600, type=int, help="Batch size")
    parser.add_argument("--ib_bandwidth", default=100, type=int, help="Batch size")
    parser.add_argument("--nvlink_size", default=64, type=int, help="number of layers")
    parser.add_argument("--nvlink_latency", default=0, type=int, help="number of attention heads")
    parser.add_argument("--ib_latency", default=0, type=int, help="number of attention heads")
    
    args = parser.parse_args()

    return args

def main():
    args=parse_args()
    
    system = {'matrix_flops_fp16': args.matrix_flops_fp16,
              'vector_flops_fp32': args.vector_flops_fp32,
              'vector_flops_fp16': args.vector_flops_fp16,
              'hbm_bandwidth': args.hbm_bandwidth,
              'nvlink_bandwidth': args.nvlink_bandwidth,
              'ib_bandwidth': args.ib_bandwidth,
              'nvlink_size': args.nvlink_size,
              'nvlink_latency': args.nvlink_latency,
              'ib_latency': args.ib_latency}

    model = {'b': args.batch_size, 
             'l': args.seq_len_min,
             'e': args.embed_dim,
             'f': args.hidden_dim_mlp,
             'h': args.nattn,
             'depth': args.depth,
             'element_size': args.element_size,
             'mask_element': args.mask_element,
             'fp32_sz': args.fp32_size,
             'fp16_sz':  args.fp16_size,
             'int_sz': args.int_size,
             'flops_units': args.flops_units}
    
    ngpus=[x**2 for x in range(int(np.sqrt(args.ngpus_min)),int(np.sqrt(args.ngpus_max)+1))]
    
    for seq_len in [args.seq_len_min,args.seq_len_max]:
        model['l']=seq_len
        
        time_mlp_total_1D, time_sa_total_1D, time_mlp_comm_1D, time_sa_comm_1D, total_memory_fwd_1D, total_memory_bwd_1D=\
compute_stats(ngpus, '1D', model, system)
        time_mlp_total_2D, time_sa_total_2D, time_mlp_comm_2D, time_sa_comm_2D, total_memory_fwd_2D, total_memory_bwd_2D=\
compute_stats(ngpus, '2D', model, system)
    
        lfmt ='-'
        fig, axs = plt.subplots(1,2,figsize=(15,5), tight_layout=True) 
        c1 = 'steelblue'
        c2 = 'salmon'
        fsz = 14
        lgnd = ["1D-nvlink{}".format(args.nvlink_size), "2D-nvlink{}".format(args.nvlink_size)]
        ax = axs[0]
        ax.plot(ngpus, time_mlp_total_1D + time_sa_total_1D, lfmt, linewidth=2, c=c1, marker = '.')
        ax.plot(ngpus, time_mlp_total_2D + time_sa_total_2D, lfmt, linewidth=2, c=c2, marker = '.')
        ax.set_xlabel('Number of GPUs', fontsize=fsz)
#     ax.set_xticks(ngpus)
        ax.legend(lgnd, fontsize=fsz-4)
        ax.set_xscale('log')
        ax.set_yscale('log')
#     ax.set_xticklabels(ngpus, fontsize=fsz-4)
        ax.set_ylabel('Total time', fontsize=fsz)
        ax.set_xticks([10,100,1000])
        ax.set_xticklabels([10,100,1000], fontsize=fsz-4)

        ax = axs[1]
        ax.plot(ngpus, total_memory_fwd_1D + total_memory_bwd_1D, lfmt, linewidth=2, c=c1, marker = '.')
        ax.plot(ngpus, total_memory_fwd_2D + total_memory_bwd_2D, lfmt, linewidth=2, c=c2, marker = '.')
        ax.set_xlabel('Number of GPUs', fontsize=fsz)
#     ax.set_xticks(ngpus)
#     ax.set_xticklabels(ngpus, fontsize=fsz-4)
        ax.set_ylabel('Memory (FW+BW) per GPU [GB]', fontsize=fsz)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xticks([10,100,1000])
        ax.set_xticklabels([10,100,1000], fontsize=fsz-4)
        ax.legend(lgnd, fontsize=fsz-4)
        plt.savefig("Parallelism_1D_2D_seq_len_"+str(seq_len)+".pdf")
        plt.show()
    
if __name__ == "__main__":
    main()

    
    
