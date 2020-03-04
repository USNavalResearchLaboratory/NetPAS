import sys
import csv

import networkx as nx
import random
import math


def outputline(w, v, line):
    w.writerow(line)
    if v is not None:
        v.writerow(line)

def build_params(params, seed):
    newparams = dict(params)
    newparams['seed'] = seed
    return newparams

# ----------------------------------------

def graph_from_catalog_row(row):
    f = eval(row['genfunc'])
    params = eval(row['gparams'])
    seed = int(row['gseed'])

    G = f(**build_params(params, seed))

    assert G.number_of_nodes()==int(row['n']) and G.number_of_edges()==int(row['e']), "Regeneration of " + row['GID'] + " failed"
    
    return G


def graph_from_GID(GID, catalog):

    G = None
    
    with open(catalog, 'r') as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row['GID']==GID:
                G = graph_from_catalog_row(row)
                break
                    
        stream.close()
        
    return G

# ----------------------------------------

def ERgraph1(n, p, seed=None):
    if seed is None:
        seed = random.randint(-sys.maxint-1, sys.maxint)
    G = nx.fast_gnp_random_graph(n, p, seed)
    GL = list(nx.connected_component_subgraphs(G))
    comp = max([(GL[i].number_of_nodes(), -min(GL[i].nodes()), i) for i in range(len(GL))])[2]
    return GL[comp]

# ----------------------------------------

def ERgen(w, v, prefix, ID, nstart=20, nend=101, nstep=5, pfunc=(lambda n: math.log(n)/float(n)), pmodfunc=(lambda n, p, r: p), ngraphs=10):
    f = 'ERgraph1'
    for n in range(nstart, nend, nstep):
        for i in range(ngraphs):
            params = dict()
            params['n'] = n
            p = pfunc(n)
            rand = random.Random()
            rand.seed(random.random())
            p = pmodfunc(n, p, rand)
            p = round(p, 12)
            params['p'] = p
            graphseed = random.randint(-sys.maxint-1, sys.maxint)
            G = eval(f)(**build_params(params, graphseed))
            OL = [str(prefix)+str(ID), f, params, graphseed, G.number_of_nodes(), G.number_of_edges()]
            outputline(w, v, OL)
            ID += 1
    
    return ID

# ----------------------------------------

def graphgen(filename=None, prefix="", ID=0, header=True, verbose=False, seed=None, commands=None):
    if filename is None:
        file = sys.stdout
    else:
        file = open(filename, 'w')
    w = csv.writer(file)
    
    if header:
        w.writerow(['GID', 'genfunc', 'gparams', 'gseed', 'n', 'e'])
        
    v = csv.writer(sys.stderr) if verbose else None
    
    random.seed(seed)
    
    if commands is not None:
        for func, params in commands:
            ID = func(w, v, prefix, ID, **params)
    
    if file != sys.stdout:
        file.close()


