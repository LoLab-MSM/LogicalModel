# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 14:36:46 2014

@author: pinojc
"""

Python ='mpirun -npernode 8 /usr/local/python2/latest/x86_64/gcc46/nonet/bin/python main_attractor_synch_cython2.py '

PLACE = 'cd /home/pinojc/LogicModel2'
N = 3
NP=4
for i in xrange(0,730):
    f=open('submit-'+str(i)+".pbs",'w')
    print>>f ,'#/bin/tcsh \n#PBS -l nodes=1:ppn=8 \n#PBS\
    -M james.c.pino@vanderbilt.edu\n#PBS -m e\n#PBS -l mem=1000mb\n#PBS \
    -l walltime=00:40:00\n#PBS -o logfile-'+str(i)+'.out\nsetpkgs  \
    -a python2 openmpi_intel  \n '
    print>>f ,PLACE
    #print 'bash mpirun -np',str(NP),
    print>>f,str(Python),' -n',str(N),'\
    -s ',str(i*14348907),' -e ',str((i+1)*14348907)
