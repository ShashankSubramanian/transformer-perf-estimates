#!/bin/bash
## On Perlmutter at NERSC
N=220
cmd="python run_configs_sweep_flopscap.py --model gpt3_1T --parallel_strat 1d --global_batch_size 4096"
image=nersc/pytorch:ngc-22.05-v1 
srun -u --mpi=pmi2 -N 2 -n $N --cpus-per-task 1 shifter --image=${image} --module=none\
    bash -c "
    $cmd
    "

## On your machine (or equivalent)
# mpirun -np 220 python run_configs_sweep_flopscap.py --model gpt3_1T --parallel_strat 1d --global_batch_size 4096
