from mpi4py import MPI
import numpy as np 
import heapq

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
#print(size)

array=np.arange(100)
per_rank=len(array)//size
remainder = len(array)-per_rank*size
if rank==0:
    for i in range(size):
        print(array[i*per_rank:(i+1)*per_rank])
    print(array[size*per_rank:])
result={}
configs=[]
result[rank]=np.sum(array[(rank-1)*per_rank:rank*per_rank])
if remainder>0:
    if rank==0:
        for i in array[(size-1)*per_rank:]:
            configs.append((i,str(i)))
        result[size+1] = heapq.nlargest(3, configs, key=lambda ky:ky[0])


if rank>0:
    for i in array[(rank)*per_rank:(rank+1)*per_rank]:
        configs.append((i,str(i)))
        result[i] = heapq.nlargest(3, configs, key =lambda ky:[ky[0]])
    comm.send(result,dest=0,tag=13)
else:
    for i in range(1,size):
        tmp=comm.recv(source=i,tag=13)
        result.update(tmp)

comm.Barrier()
if rank == 0:
    print(result)
