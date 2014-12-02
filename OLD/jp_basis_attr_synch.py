# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 16:55:23 2014

@author: pinojc
"""

import pypar
from numpy import *
import itertools
from functions import allNodes as function
from itertools import combinations_with_replacement
import numpy as np
import timeit
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n","--numberstates", type=str, help="provide a number of states")
parser.add_argument("-nn","--numbernodes", type=str, help="provide a number of nodes")
parser.add_argument("-s","--start",type=str,help='starting string to convert to base Nstates')
parser.add_argument("-e","--end",type=str,help='ending string to convert to based Nstates')

args = parser.parse_args()
numStates=int(args.numberstates)
numNodes = int(args.numbernodes)
start = int(args.start)
end = int(args.end)

#numStates = 3
#numNodes = 6
totalStates = int(numStates) ** (1.*int(numNodes))
#samples = np.empty((totalStates,numNodes), dtype=int)
#samples = np.zeros((numStates**9,numNodes), dtype=int)
#list(map("".join, itertools.product('012',repeat=21)))



#for i,comb in enumerate(combinations_with_replacement(range(numStates),numNodes)):
#    samples[i,:] = comb
#    samples[i,:].astype(int)
#for i,comb in enumerate(itertools.product(range(numStates),repeat=numNodes)):
#    samples[i,:] = comb

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

#samples = np.zeros((float(end) - float(start),float(numNodes)),dtype=int)
#print len(samples)
#for i in xrange(len(samples)):
#    samples[i,:] = changebase(start)
#    start+=1

#print samples

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
    #print blank[:2,:]
    while checkStates(blank[0:counter,:])==-1:
        #print counter,blank[counter]
        blank[counter+1,:]=getNextState(blank[counter,:])
        counter += 1
        #x1 = getNextState(x[-1,:])
        #x = np.vstack((x,x1))
    #print 'counter',counter
    xx = blank[checkStates(blank[:counter,:]):counter]
    #xx = x[checkStates(x):-1,:]
    #if len(xx.shape)==1:
    #    print 'here'
    #    xx.reshape(len(xx),1)
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

#import time
#start = time.time()

def main():
    data = dict()
    x = np.zeros((2,numNodes),dtype=int)
    for i in range(0,len(samples)):
        #print i+1,'/',len(samples)
        x[0,:]=samples[i,:]
        #x = samples[i,:]
        x[1,:]=getNextState(x[0,:])
        #x=np.vstack((x,getNextState(x)))
        tmp=run(x)
        if tmp in data:
            data[tmp]+=1
        else:
            data[tmp]=1
        #data.append(run(x))
    print 'time of '+str(np.shape(samples)[0])+' calculations '+ str((time.time() - start)/60)+' minutes'
    print data
#main()

#Initialise
t = pypar.time()
P = pypar.size()
p = pypar.rank()
processor_name = pypar.get_processor_name()
# Block stepping
stepping = 100
# Number of blocks
#print end ,start
samplesize = int(end) - int(start)
print 'samplesize = ',samplesize

print 1.*samplesize/stepping
B = samplesize/stepping +10 # Number of blocks

print 'Processor %d initialised on node %s' % (p, processor_name)
assert P > 1, 'Must have at least one slave'
assert B > P - 1, 'Must have more work packets than slaves'
#assert samplesize%stepping ==0, "Block size doesn't match sample size!\n Must \
#have sample size / stepping ==0."


if p == 0:
    #Create array for storage
    A = dict()
    #A = np.zeros((samplesize,numNodes), dtype='i')
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
        for tmp in data[1]:
            if tmp in A:
                A[tmp]+=1
            else:
                A[tmp]=1
        print A
        #A [data[0]*stepping:data[0]*stepping+stepping] = data[1]             # Aggregate data
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
    print A.keys()
    print A.values()
    print np.sum(A.values())
    #np.savetxt('output.txt',A)
