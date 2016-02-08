# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:48:06 2015

@author: pinojc
"""
import numpy as np

log = open('logical_model_normal_24.o2528822')
Start = []
End = []
attractors = {}
time = 0
for line in log:
    l = line[:-1]
    if l.startswith('start'):
        start,end = l.split(',')
        start = np.int(start.split('=')[1])
        end = np.int(end.split('=')[1])
        Start.append(start)
        End.append(end)
    if l.startswith('Attractors'):
        for j in ('Attractors','[',']','(',')',',',' '):
            l = l.replace(j,'')
            
        try:
            attractors[l] += 1
        except:
            attractors[l] = 1
    if l.startswith('Computed'):
        time+= np.float(l.split()[3])
for i in Start:
    if i in End:
        continue
    else:
        if np.int(i)==0:
            continue
        else:
            print 'Gap at %s'%i
print "Started at %s"%np.min(Start)
print "Ended at = %s"%np.max(End)
print "Attractors",attractors.keys()
print "Freq",attractors.values()
print "Total time = %s cpu hours"%(time/60)