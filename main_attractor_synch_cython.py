# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 16:55:23 2014

@author: James C Pino ,  Leonard Harris, Carlos F. Lopez

"""

#import pypar
from numpy import *
import itertools
from itertools import combinations_with_replacement
import numpy as np
import timeit
import argparse

# Since we had to compile the functions with Cython, we must choose within
# the program which to use. Options so far are
# allNodes , ironFunction, IRP2_1, IRP2_2
#from ironFunct2 import ironFunction as function
#from functions import ironFunction as function
parser = argparse.ArgumentParser()
parser.add_argument("-n","--numberstates", type=str, help="provide a number of states")
parser.add_argument("-nn","--numbernodes", type=str, help="provide a number of nodes")
parser.add_argument("-s","--start",type=str,help='starting string to convert to base Nstates')
parser.add_argument("-e","--end",type=str,help='ending string to convert to based Nstates')
parser.add_argument("-f","--function",type=str,help='function to run simulation on')
parser.add_argument("-v","--verbose",type=str,help='if you want verbose updates (use with single processor)')
args = parser.parse_args()
numStates=int(args.numberstates)
numNodes = int(args.numbernodes)
start = int(args.start)
end = int(args.end)
Function = str(args.function)
print Function
v = int(args.verbose)
samplesize = int(end) - int(start)
Function = __import__(Function)
function = Function.function
#print samplesize

totalStates = int(numStates) ** (1.*int(numNodes))


def changebase(number):
    counter = -1
    state = np.zeros(numNodes,dtype=int)
    quotient = number/numStates
    remainder = int(number) % int(numStates)
    number = quotient
    state[counter] = remainder
    counter -=1
    while quotient !=0:
        quotient = int(number)/int(numStates)
        remainder = int(number) % int(numStates)
        state[counter] = remainder
        counter -=1
        number = quotient
    return state


def getNextState(state):
    nextstate = function(state)
    return nextstate

def checkStates(x):
    if ((x[:-1,:] == x[-1,:]).sum(axis=1)==numNodes).any() ==1:
        i = np.where((x[:-1,:] != x[-1,:]).sum(axis=1) == 0)
        return i[0][0]
    else:
        return -1
blank = np.empty((numStates**9,numNodes), dtype=int)
def run(x):
    counter = 1
    blank[0:2,:] = x
    while checkStates(blank[0:counter,:])==-1:
        blank[counter+1,:]=getNextState(blank[counter,:])
        counter += 1
    xx = blank[checkStates(blank[:counter,:]):counter]
    ncols = xx.shape[1]
    dtype = xx.dtype.descr * ncols
    struct = xx.view(dtype)
    uniq = np.unique(struct)
    uniq = uniq.view(xx.dtype).reshape(-1, ncols)
    uniq[uniq[:,1].argsort()]
    uniq = uniq[0]
    data = ''
    for u in uniq:
        data+=str(u)
    return data


def main():
    import time
    print 'Started '
    start_time = time.time()
    data = dict()
    x = np.zeros((2,numNodes),dtype=int)
    for i in range(start,end):
        if v == 1:
            print str(i),'/',end
        x[0,:]=changebase(i)
        x[1,:]=getNextState(x[0,:])
        tmp=run(x)
        if tmp in data:
            data[tmp]+=1
        else:
            data[tmp]=1
    print 'time of '+str(samplesize)+' calculations '+ str((time.time() - start_time)/60)+' minutes'
    print data
main()

# For parallel usage on a cluster or multi core computer uncomment below
# and comment main() above.
# Must have pypar installed, uses a "stepping" of 100, which means splits up
# the job in batches of 100 over the processors
"""
#Initialise
t = pypar.time()
P = pypar.size()
p = pypar.rank()
processor_name = pypar.get_processor_name()
# Block stepping
stepping = 100
samplesize = int(end) - int(start)
B = samplesize/stepping +10 # Number of blocks

print 'Processor %d initialised on node %s' % (p, processor_name)
assert P > 1, 'Must have at least one slave'
assert B > P - 1, 'Must have more work packets than slaves'



if p == 0:
    print 'samplesize = ',samplesize
    print 'split up into %s segements' % str(1.*samplesize/stepping)
    #Create array for storage
    Results = dict()
    # Create work pool (B blocks)
    workpool = []
    for i in range(B):
        workpool.append(i)
    # Distribute initial work to slaves
    w = 0
    for d in range(1, P):
        pypar.send(workpool[w], destination=d)
        w += 1
    # Receive computed work and distribute more
    terminated = 0
    while(terminated < P - 1):
        data,status= pypar.receive(pypar.any_source,return_status=True)
        for tmp in data[1]:     # check to see if new states are already present
            if tmp in Results:
                Results[tmp]+=1
            else:
                Results[tmp]=1
        d = status.source  # Id of slave that just finished
        if w < len(workpool):
            # Send new work to slave d
            pypar.send(workpool[w], destination=d)
            w += 1
        else:
            # Tell slave d to terminate
            pypar.send(None, destination=d)
            terminated += 1
    print 'Computed '+str(samplesize)+' samples in %.2f seconds' % (pypar.time() - t)

else:
    while(True):
        # Receive work (or None)
        W = pypar.receive(source=0)
        if W is None:
            print 'Slave p%d finished: time = %.2f' % (p, pypar.time() - t)
            break
        # Compute allocated work
        data = []
        for i in xrange(0,stepping):
            #print W*stepping+i
            if W*stepping+i == samplesize or W*stepping+i > samplesize:
                #print 'Finished with p%d'% (p)
                break
            else:

                x=changebase((W)*stepping+i)
                x = np.vstack((x,getNextState(x)))
                tmp = run(x)
                data.append(tmp)

        # Return result
        pypar.send((W,data), destination=0)
pypar.finalize()
if p == 0:
    print Results.keys()
    print Results.values()
    print np.sum(Results.values())
    #np.savetxt('output.txt',A)
"""
