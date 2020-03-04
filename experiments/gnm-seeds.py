# Script to find top-level seeds that produce connected G_{n,m}
# random graphs.
import copy
import random
from timeit import default_timer as timer

import networkx as nx
import csv

import sys
sys.path.append('../code')

from probesim import util as ps
from probesim import jrnx

import datetime

from os.path import isdir
from os import makedirs

import glob

control_rng = random.Random()

# Are we trying to extend a seed file?
# If so, update j0, jmax, and num_g in various places below
EXTEND = False

top_param_list = [(42,63,391),(42,63,195)] # This should be an iterable of (b,n,m) tuples

# top_param_list = [(9,25,48), (16,36,80), (16,46,120), (27,105,234), (2,4,3), (6,12,12), (42,84,105), (6,9,9), (12,16,18), (16,48,128), (42,63,84), (20,25,30), (64,190,504), (64,192,1024), (64,253,756), (64,256,768), (81,401,960), (420,525,840), (156,208,312)] # This should be an iterable of (b,n,m) tuples

# top_param_list = [(156,208,312)] # This should be an iterable of (b,n,m) tuples
# top_param_list = [(156,208,312), (420,525,840)] # This should be an iterable of (b,n,m) tuples

# Update j0 and jmax to reflect the values to search if we're extending
# a seed file.  Also uncomment the "while proc_seed != " loop below.
# We'll stop once we find num_connected graphs that are connected
if EXTEND:
    j0 = 800000000
    jmax = 3000000000
else:
    j0 = 0
    jmax = 250000000
num_connected = 10000


header_row = ['n','m','proc_seed','g_seed']

for (b,n,m) in top_param_list:

    print 'Doing (b,n,m) =', b, n, m
    
    control_rng.seed(a=1)

    if not EXTEND:
        with open('./gnm-rng/gnm-seeds-' + str(b) + '-' + str(n) + '-' + str(m) + '.csv', 'w') as ofile:
            writer = csv.DictWriter(ofile, header_row, extrasaction='ignore')
            writer.writeheader()
        
    output = {'b':b, 'n':n, 'm':m}
    
    if EXTEND:
        num_g = 0 # Number of connected G found so far
    else:
        num_g = 0
    if num_connected < num_g:    
        raise RuntimeError('Already have enough connected graphs!')
    
    j = 0 # Number of previous uses of control_rng
    
    proc_seed = None
#     while proc_seed != NNNNNNNN: # Replace this with the last good
#        # proc_seed (i.e., the last proc_seed that produced a connected
#        # graph) that we found
#         proc_seed = control_rng.randint(-sys.maxint - 1, sys.maxint)
#         j += 1
# 
#     print 'Done with old seeds at:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    while j < jmax:
    
        if j % 200000000 == 0:
            print 'At', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print 'j = ', j
            print 'Num. good seeds:', num_g
        # proc_seed would be generated in the control program
        proc_seed = control_rng.randint(-sys.maxint - 1, sys.maxint)
        j += 1
        
        # The following would be done inside the pool process
        proc_rng = random.Random()
        proc_rng.seed(a=proc_seed)
        g_seed = proc_rng.randint(-sys.maxint - 1, sys.maxint)
        G = jrnx.jr_gnm_random_graph(n,m,seed=g_seed)
        if nx.is_connected(G):
            output['proc_seed'] = proc_seed
            output['g_seed'] = g_seed
            num_g += 1
            with open('./gnm-rng/gnm-seeds-' + str(b) + '-' + str(n) + '-' + str(m) + '.csv', 'a') as ofile:
                writer = csv.DictWriter(ofile, header_row, extrasaction='ignore')
                writer.writerow(output)
            if num_g == num_connected:
                break

