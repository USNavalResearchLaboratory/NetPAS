import sys
import csv
import networkx as nx
import random

from .. import util as ps
from ..evaluation import probestats as st
import graphgen as gg


####
#
# module helper functions
#

def outputline(w, v, line):
    w.writerow(line)
    if v is not None:
        v.writerow(line)

def build_params(G, params, seed):
    newparams = dict(params)
    newparams['G'] = G
    newparams['seed'] = seed
    return newparams
    
####
#
# generation directly from catalog
#

def inputgen_from_catalog_row_number(rownum, icatalog, gcatalog):
    tup = None
    
    with open(icatalog, 'r') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            count += 1
            if count == rownum:
                tup = inputgen_from_catalog_row(row, gcatalog)
                break
                
        f.close()
    
    return tup


def inputgen_from_catalog_row_numbers(row_numbers, icatalog, gcatalog):
    rows = sorted(row_numbers)

    list_of_input_tuples = []
    list_of_IIDs = []
    target_position = 0

    with open(icatalog, 'r') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            count += 1
            if target_position >= len(rows) or count > rows[target_position]: break

            if count == rows[target_position]:
                tup = inputgen_from_catalog_row(row, gcatalog)
                list_of_input_tuples.append(tup)
                list_of_IIDs.append(row['IID'])
                target_position += 1

        f.close()

    return list_of_IIDs, list_of_input_tuples


def inputgen_from_catalog_range(icatalog, gcatalog, start=1, stop=None, step=1):
    list_of_input_tuples = []
    list_of_IIDs = []

    with open(icatalog, 'r') as f:
        reader = csv.DictReader(f)
        current = 0
        grabrow = start
        for row in reader:
            current += 1
            if stop is not None and current >= stop: break

            if current == grabrow:
                tup = inputgen_from_catalog_row(row, gcatalog)
                list_of_input_tuples.append(tup)
                list_of_IIDs.append(row['IID'])
                grabrow += step

        f.close()

    return list_of_IIDs, list_of_input_tuples


def inputgen_from_catalog_row(row, gcatalog, pathfunc=None, pparams=None):
    
    G = None
    R = None
    
    G = gg.graph_from_GID(row['GID'], gcatalog)
    f = eval(row['rfunc'])
    params = eval(row['rparams'])
    seed = int(row['rseed'])
    R = f(**build_params(G, params, seed))
    
    assert len(R)==int(row['rsize']) and abs(round(float(len(R))/G.number_of_nodes(), 12) - float(row['rfrac'])) < 0.00001, "Regeneration of probing-nodes set for IID " + row['IID'] + " failed"
    
        
    if pathfunc is None:
        pathfunc = eval(row['pathfunc'])
    elif not callable(pathfunc):
        pathfunc = eval(pathfunc)

    params = eval(row['pparams'])
    if type(pparams)==type(dict()):
        for k, v in pparams.iteritems():
            params[k] = v
            
    P = ps.filternodes(pathfunc(G, **params), R)
    
    return G, R, P

def inputgen_from_IID(IID, icatalog, gcatalog):

    tup = None
    
    with open(icatalog, 'r') as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row['IID']==IID:
                tup = inputgen_from_catalog_row(row, gcatalog)
                break
                    
        stream.close()
        
    return tup


####
#
# catalog of path-generating functions
#

def shortestpaths(G):
    return nx.all_pairs_shortest_path(G)

####
#
# helper functions for inputgen functions
#

def rpermute(G, rfrac, seed=None):
    if seed is None:
        seed = random.randint(-sys.maxint-1, sys.maxint)
    rand = random.Random()
    rand.seed(seed)
    R = G.nodes()
    rand.shuffle(R)
    return R[:int(round(len(R)*float(rfrac)))]
    
####
#
# functions that can be called by inputgen
#    

def rfracstep(w, v, prefix, ID, catalog, cfilter=(lambda d: True), pathfunc='shortestpaths', pathfuncparams=None, rstart=0.2, rend=0.51, rstep=0.05, numr=10):
    f = 'rpermute'
    params = dict()
    pf = pathfunc if callable(pathfunc) else eval(pathfunc)
    if pathfuncparams is None:
        pathfuncparams = dict()
    with open(catalog, 'r') as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if cfilter(row):
                G = gg.graph_from_catalog_row(row)
                rfrac = round(rstart, 12)
                while rfrac < rend:
                    for i in range(numr):
                        rseed = random.randint(-sys.maxint-1, sys.maxint)
                        params['rfrac'] = rfrac
                        R = eval(f)(**build_params(G, params, rseed))
                        pd = ps.filternodes(pf(G, **pathfuncparams), R)
                        epd = ps.edgesinpd(pd, histo=True)
                        npd = ps.nodesinpd(pd, histo=True)
                        tpd = ps.nodesinpd(pd, histo=True, transit=True)
                        OL = [
                            prefix + str(ID),           # IID
                            row['GID'],                 # GID
                            f,                          # name of helper function
                            params,                     # parameters to helper function
                            rseed,                      # seed given to helper function
                            str(pathfunc),              # function used to generate paths
                            pathfuncparams,             # parameters to path-generation function
                            float(len(R))/G.number_of_nodes(),  # actual fraction of all nodes that are probing nodes
                            len(R),                     # number of probing nodes
                            ps.countpaths(pd),          # number of distinct paths in the probing set
                            len(npd),                   # number of nodes covered by union of probing paths
                            st.avgDictValue(npd),       # average number of times a node appears in union of probing paths
                            len(tpd),                   # number of non-probing nodes covered by union of probing paths
                            st.avgDictValue(tpd),       # average number of times a transit node appears in union of probing paths
                            len(epd),                   # number of edges covered by union of probing paths
                            st.avgDictValue(epd),       # average number of times an edge appears in union of probing paths
                            st.pctDictValue(epd),       # median number of times an edge appears in union of probing paths
                        ]
                        outputline(w, v, OL)
                        ID += 1
                    
                    rfrac = round(rfrac + rstep, 12)
                    
        stream.close()
    
    return ID

####
#
# main functions for the module
#


def inputgen(filename=None, prefix="", ID=0, header=True, verbose=False, seed=None, commands=None):
    if filename is None:
        stream = sys.stdout
    else:
        stream = open(filename, 'w')
    w = csv.writer(stream)
    
    if header:
        w.writerow(['IID', 'GID', 'rfunc', 'rparams', 'rseed', 'pathfunc', 'pparams', 'rfrac', 'rsize', 'numpaths', 'nodesinrunion', 'nodeavgfreqinrunion', 'transitinrunion', 'transitavgfreqinrunion', 'edgesinrunion', 'edgeavgfreqinrunion', 'edgemedfreqinrunion'])
        
    v = csv.writer(sys.stderr) if verbose else None
    
    random.seed(seed)
    
    if commands is not None:
        for func, catalog, params in commands:
            ID = func(w, v, prefix, ID, catalog, **params)
    
    if stream != sys.stdout:
        stream.close()
