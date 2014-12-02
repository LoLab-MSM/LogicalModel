# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 14:36:46 2014

@author: pinojc
"""

#/bin/tcsh
#PBS -l nodes=1:ppn=16
#PBS -M james.c.pino@vanderbilt.edu
#PBS -m bae
#PBS -l mem=4000mb
#PBS -l walltime=00:20:00
#PBS -o sample_0-100.out


PLACE = 'cd /home/pinojc/LogicModel'
#mpirun -np 16 /dors/meilerlab/apps/Linux2/x86_64/bin/python basis_attr_synch_cython.py -n 3 -nn 21 -s 0 -e 14348907
Python = '/dors/meilerlab/apps/Linux2/x86_64/bin/python basis_attr_synch_cython.py'
N = 3
NN = 21
NP=4
for i in xrange(33,50):
    #f=open('submit-'+str(i)+".pbs",'w')
    #print>>f ,'#/bin/tcsh \n#PBS -l nodes=1:ppn=1 \n#PBS\
    #-M james.c.pino@vanderbilt.edu\n#PBS -m e\n#PBS -l mem=1000mb\n#PBS \
    #-l walltime=02:00:00\n#PBS -o logfile-'+str(i)+'.out\nsbset \
    #meilerlab  \n '
    #print>>f ,PLACE
    #print 'bash mpirun -np',str(NP),
    print str(Python),' -n',str(N),' -nn ',str(NN),' \
    -s ',str(i*14348907),' -e ',str((i+1)*14348907)