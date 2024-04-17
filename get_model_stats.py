import numpy as np
import json
from execution import execute_1d, execute_2d
import argparse



# models

gpt2 = {'l': 1024, 'e': 1600, 'h': 32, 'depth': 48}
gpt3 = {'l': 2048, 'e': 12288, 'h': 96, 'depth': 96}
gpt3_1T = {'l': 2048, 'e': 25600, 'h': 160, 'depth': 180}
gpt3_lowdepth = {'l': 2048, 'e': 12288, 'h': 256, 'depth': 96 // 8}
vit_era5 = {'l': 64800, 'e': 5120, 'h': 32, 'depth': 24}
vit_era5_big = {'l': 64800, 'e': 6144, 'h': 32, 'depth': 32}
vit_era5_1px = {'l': 1036800, 'e': 6144, 'h': 32, 'depth': 32}
vit_era5_1px_big = {'l': 1036800, 'e': 12288, 'h': 96, 'depth': 96}

models= {'gpt2': gpt2, 'gpt3': gpt3, 'gpt3_1T': gpt3_1T, 'gpt3_lowdepth': gpt3_lowdepth, \
         'vit_era5': vit_era5, 'vit_era5_big': vit_era5_big, 'vit_era5_1px': vit_era5_1px, 'vit_era5_1px_big': vit_era5_1px_big}



def compute_stats(global_batch_size,config_type,n_gpus,nvs_list,model, model_str, exec_model,nlargest,verbose):
    if exec_model == '1d':
        execute_function = execute_1d
    elif exec_model == '2d':
        execute_function = execute_2d
        
    with open('config-'+config_type+'.json', 'r') as file:
        system = json.load(file)
        
    throughputs = {}
    stats = {}
    confs = {}

    start = None
    
    for nvs in nvs_list:
        throughputs[nvs] = []
        stats[nvs] = []
        confs[nvs] = []
        start = None
        system['nvlink_size'] = nvs
    
        configs = execute_function(model, n_gpus, global_batch_size=global_batch_size, \
                                   system=system, verbose=verbose, nlargest=nlargest)
        for s,config in enumerate(configs):
            if len(config) > 0: # check feasibility
                if not start:
                    start = s
                throughput, stat, _, conf = config[0]

                throughputs[nvs].append(throughput)
                stats[nvs].append(stats)
                confs[nvs].append(conf)

        n_gpus = n_gpus[start:]
    np.savez('outputs/exec_'+exec_model+'_model_'+str(model_str)+'_config_'+config_type+'.npz', \
         confs=confs, stats=stats, throughputs = throughputs, ngpus=n_gpus, global_batch_size=global_batch_size)
    
def parse_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--global_batch_size", default=4096, type=int, help="gobal batch size")
    parser.add_argument("--config_type", default='A100', type=str, help="the gpu type")
    parser.add_argument("--ngpus_index", default=14, type=int, help="the max number of gpus for the system as a power of 2")
    parser.add_argument("--nv_index", default=7, type=float, help="the max number of gpus for NVLink domain as a power of 2")
    parser.add_argument("--LLM_model", default='gpt3_1T', type=str, help="LLM model")
    parser.add_argument("--f_multiple", default=4, type=int, help="f = multiple * e for the LLM")
    parser.add_argument("--parallel_option", default='1d', type=str, help="tensor parallelization strategy")
    parser.add_argument("--top_n", default=10, type=int, help="how many top results to store")
    parser.add_argument("--verbose", default=False, type=bool, help="whether to print results for intermediate steps")
    
    
    args = parser.parse_args()

    return args
    
def main():
    args = parse_args(); 
    
    global_batch_size = args.global_batch_size
    n_gpus = 2**np.array([i for i in range(2,args.ngpus_index)])
    nvs_list = list(2**np.array([i for i in range(1,args.nv_index)]))
    config_type=args.config_type
    model_str = args.LLM_model
    model = models[model_str]
    model['f'] = args.f_multiple * model['e']
    exec_model = args.parallel_option
    nlargest = args.top_n
    verbose = args.verbose
    
    compute_stats(global_batch_size,config_type,n_gpus,nvs_list,model, model_str, exec_model, nlargest, verbose)
    
if __name__ == "__main__":
    main()
    