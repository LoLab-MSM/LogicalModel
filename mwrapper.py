#!/usr/bin/env python
from mpi4py import MPI
from subprocess import call
import numpy as np
import sys
exctbl = sys.argv[1]
model = sys.argv[2]
nproc = sys.argv[3]
nstates = sys.argv[4]
runs = np.int(np.float(nstates)/np.float(nproc))
#print runs
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
start = rank*runs
end = (rank +1)* runs
#print start, end
cmd = 'python '+exctbl+" -m %s -s %s -e %s"%(model,start,end)
sts = call(cmd,shell=True)
comm.Barrier()
