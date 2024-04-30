import numpy as np
import json
from execution_analysis import execute_1d, execute_2d
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



def compute_stats(global_batch_size, config_type, n_gpus, nvs_list, model, model_str, exec_model, lfactor, efactor, dfactor, nlargest, verbose):
    if exec_model == '1d':
        execute_function = execute_1d
    elif exec_model == '2d':
        execute_function = execute_2d
        
    with open('config-'+config_type+'.json', 'r') as file:
        system = json.load(file)
        
    throughputs = {}
    stats = {}
    confs = {}
    ngpus={}

    start = None
    
    for nvs in nvs_list:
        throughputs[nvs] = {}
        stats[nvs] = {}
        confs[nvs] = {}
        start = None
        system['nvlink_size'] = nvs
    
        configs = execute_function(model, n_gpus, global_batch_size=global_batch_size, \
                                   system=system, verbose=verbose, nlargest=nlargest)
        ngpus[nvs]=sorted(configs.keys())
      
        
        for n in ngpus[nvs]:
            throughputs[nvs][n]=[]
            stats[nvs][n]=[]
            confs[nvs][n]=[]
            for conf in configs[n]:
                throughputs[nvs][n].append(conf[0])
                stats[nvs][n].append(conf[1])
                confs[nvs][n].append(conf[3])
      
    np.savez('outputs/exec_'+exec_model+'_model_'+str(model_str)+'_config_'+config_type+'_lfactor_'+str(lfactor)+'_efactor_'+str(efactor)+'_dfactor_'+str(dfactor)+'.npz', \
         confs=confs, stats=stats, throughputs = throughputs, ngpus=ngpus, global_batch_size=global_batch_size)
    
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
    parser.add_argument("--lfactor", default=0, type=int, help="2^i x l")
    parser.add_argument("--efactor", default=0, type=int, help="2^i x e")
    parser.add_argument("--dfactor", default=0, type=int, help="2^i x depth")
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
    if args.efactor>0:
        model['e'] = 2**args.efactor * model['e']
    model['f'] = args.f_multiple * model['e']
    if args.dfactro>0:
        model['depth'] = 2**args.dfactor * model['depth']
    if args.lfactor>0:
        model['l'] = 2**args.lfactor*model['l']
        print(model['l'])
    exec_model = args.parallel_option
    nlargest = args.top_n
    verbose = args.verbose
    
    compute_stats(global_batch_size,config_type,n_gpus,nvs_list,model, model_str, exec_model, args.lfactor, args.efactor, args.dfactor, nlargest, verbose)
    
if __name__ == "__main__":
    main()
    
