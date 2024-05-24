from __future__ import print_function
from collections import OrderedDict
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    data = [{'a': 7}, {'b': 3.14}]
    comm.send(data,dest=1,tag=13)
else:
    data= None
    data = comm.recv(data,source=0,tag=13)

#data = comm.scatter(data, root=0)

data_type = type(data)
print(f'Data is {data} on rank {rank} with type {data_type}')

