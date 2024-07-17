#!/bin/bash
N=9
cmd="python run_configs.py --model gpt3"
image=nersc/pytorch:ngc-22.05-v1 
srun -u --mpi=pmi2 -N 1 -n $N --cpus-per-task 12 shifter --image=${image} --module=none\
    bash -c "
    $cmd
    "
