#!/bin/bash
N=9

## on Perlmutter at NERSC, you can do this:
cmd="python run_configs.py --model gpt3_1T --parallel_strat 1d --global_batch_size 4096"
image=nersc/pytorch:ngc-22.05-v1 
srun -u --mpi=pmi2 -N 1 -n $N --cpus-per-task 12 shifter --image=${image} --module=none\
    bash -c "
    $cmd
    "

## On your machine (or equivalent)
# mpirun -np 9 python run_configs.py --model gpt3_1T --parallel_strat 1d --global_batch_size 4096
