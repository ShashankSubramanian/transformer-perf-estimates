#!/bin/bash
N=9
cmd="python run_configs.py --model vit_era5 --parallel_strat 2d-seqp --global_batch_size 4096"
image=nersc/pytorch:ngc-22.05-v1 
srun -u --mpi=pmi2 -N 2 -n $N --cpus-per-task 24 shifter --image=${image} --module=none\
    bash -c "
    $cmd
    "
