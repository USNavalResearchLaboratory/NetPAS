# Script to run experiments using setcover strategy
import copy
import random
from timeit import default_timer as timer

import networkx as nx
import csv

import sys
sys.path.append('../code')

from probesim.evaluation import seqstats2 as stats
from probesim.strategy import mwscapprox as strategy
from probesim.topology import inputgen
from probesim.topology import weighting
from probesim import util as ps
from probesim import jrnx


from probesim.topology import dcell
from probesim.topology import xgft

import multiprocessing as mp


from os.path import isdir
from os.path import isfile as isfile
from os import makedirs

import glob

import argparse

parser = argparse.ArgumentParser(description='Run simulations using basic random probing')

parser.add_argument('--simroot', metavar='DIR', type=str, default='../', help='Root directory for sim hierarchy (likely netpas/)')

parser.add_argument('--ifsuff', type=str, default='', help='Suffix on input file (excluding extension)')
parser.add_argument('--nprocs', type=int, default=None, help='Number of processes to use')

seed_group = parser.add_mutually_exclusive_group(required=False)
seed_group.add_argument('--seed', type=int, default=None, help='Seed to use for top-level RNG')
seed_group.add_argument('--seedfile', action='store_true', default=None, help='Use seed file(s)')

parser.add_argument('--k', type=int, default=None, help='Number of paths k to probe (in expectation) per timestep')
parser.add_argument('--maxs', type=int, default=None, help='Maximum number of timesteps to probe')
parser.add_argument('--seta', type=int, default=None, help='Parameter a (parameter alpha is a/100).')

# Which topology collection to use
top_group = parser.add_mutually_exclusive_group(required=True)
top_group.add_argument('--zoo', action='store_true', help='Use topologies from the topology zoo')
top_group.add_argument('--cat', action='store_true', help='Use topologies from the catalog')
top_group.add_argument('--smallcat', action='store_true', help='Use topologies from the small catalog')
top_group.add_argument('--dcell', action='store_true', help='Use DCell topologies')
top_group.add_argument('--spdcell', action='store_true', help='Use DCell topologies with shortest-paths routing')
top_group.add_argument('--xgft', action='store_true', help='Use XGFT topologies')
top_group.add_argument('--gnm', action='store_true', help='Use random G_{n,m} graphs')
top_group.add_argument('--ba', action='store_true', help='Use Barabasi--Albert graphs')
top_group.add_argument('--fracba', action='store_true', help='Use fractional version of Barabasi--Albert graphs')

# How many trials?
trials_group = parser.add_mutually_exclusive_group(required=False)
trials_group.add_argument('--T100', action='store_true', help='Run 100 trials; overridden by --test')
trials_group.add_argument('--T316', action='store_true', help='Run 316 trials; overridden by --test')
trials_group.add_argument('--T1000', action='store_true', help='Run 1000 trials; overridden by --test')
trials_group.add_argument('--T3162', action='store_true', help='Run 3162 trials; overridden by --test')
trials_group.add_argument('--T10000', action='store_true', help='Run 10000 trials; overridden by --test')


# Which data directory to use (default is local, i.e., data/)
data_group = parser.add_mutually_exclusive_group(required=False)
data_group.add_argument('--test', action='store_true', help='Use data from testdata/; always runs three trials')
data_group.add_argument('--server', action='store_true', help='Use data from server-data/')
data_group.add_argument('--local', action='store_true', help='Use data from local-data/; always runs 25 trials')

########

args = parser.parse_args()

########


def setcover_trial_from_paths(sc_args):
    """
    Run a single setcover trial when given the graph and probing paths.
    
    :param sc_args is a tuple with the following:
    
    ofname: file name for output
    header_row: header for use in opening a DictWriter
    seed: main seed used for this trial
    P: set of probing paths
    G: NetworkX graph object
    alpha: setcover parameter alpha
    k: number of paths to probe per timestep
    output: dict with partial info about this trial; this is mutated here
    statImp: importance weight for computing result statistics
    statTol: tolerance weight for computing result statistics
    
    This runs a trial, computes the statistics on the probing sequence, and writes the result to ofile (after taking a lock on it so that many copies of this can be run in parallel).
    """
    (ofname, header_row, trial_num, seed, maxs, P, G, alpha, k, output, statImp, statTol) = sc_args
    
    # Create the RNG for this trial and seed it with the seed from the
    # top-level RNG passed in the argument.  Even if we don't use all these
    # seeds, create them in the same order so that, e.g., sim_seed is always
    # the seventh output of our trial's RNG.  Save the relevant ones in 
    # the output dict to be written to file.
    output['trialseed'] = seed
    output['trialnum'] = trial_num
    trial_RNG = random.Random()
    trial_RNG.seed(a = seed)
    g_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    b_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    p_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    w_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    t_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    res_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    sim_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    # Seeds:
    # g: graph G
    # b: beacons
    # p: paths P
    # w: weighting
    # t: test set
    # res: reserved for future use
    # sim: simulation

    output['gseed'] = 'na'
    output['bseed'] = 'na'
    output['pseed'] = 'na'
    output['wseed'] = 'na'
    output['tseed'] = 'na'
    output['simseed'] = sim_seed    

    # Below, olock is a global lock that is shared by all processes
    # running this in parallel.  It is acquired before writing to the
    # output file and released after the writing is done.

    # Run a trial, with sim_seed as the randomness, to produce the
    # probing sequence seq
    try:
        t = timer()
        start_time = timer()
        seq = strategy.simulate(copy.deepcopy(G), copy.deepcopy(P), alpha=alpha, trialseed=sim_seed, k=k, maxsteps=maxs)
        output['simtime'] = timer() - start_time
    except Exception as e:
        output['simtime'] = timer() - start_time
        output['status'] = 'Exception raised during simulate: ' + repr(e)
        with olock:
            with open(ofname, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, header_row, extrasaction='ignore')
                writer.writerow(output)
        return

    # Compute the statistics on seq
    try:
        t = timer()
        results = stats.all_sequence_stats(seq, P, G, importance=statImp, tolerance=statTol)
    except Exception as e:
        output['status'] = 'Exception raised during stats computation ' + repr(e)
        with olock:
            with open(ofname, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, header_row, extrasaction='ignore')
                writer.writerow(output)
        return

    # Update output with the statistics and write it to file
    output.update(results)
    with olock:
        with open(ofname, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, header_row, extrasaction='ignore')
            writer.writerow(output)
    return


def setcover_gnm_trial(sc_args):
    """
    Run a single setcover trial for specified G_{n,m} parameters
    
    :param rb_args is a tuple with the following:
    
    ofname: file name for output
    header_row: header for use in opening a DictWriter
    b: number of beacons to use
    n: number of graph nodes
    m: number of graph edges
    trial_seed: main seed used for this trial
    alpha: setcover parameter alpha
    k: number of paths to probe in expectation
    output: dict with partial info about this trial; this is mutated here
    edge_attr: weight attribute for edges
    node_attr: weight attribute for nodes
    statImp: importance weight for computing result statistics
    statTol: tolerance weight for computing result statistics
    
    This generates a G_{n,m} graph, chooses beacons, computes probing paths, runs a trial, computes the statistics on the probing sequence, and writes the result to ofile (after taking a lock on it so that many copies of this can be run in parallel).
    """
    (ofname, header_row, trial_num, b, n, m, trial_seed, maxs, alpha, k, output, edge_attr, node_attr, statImp, statTol) = sc_args

    # Create the RNG for this trial and seed it with the seed from the
    # top-level RNG passed in the argument.  Even if we don't use all these
    # seeds, create them in the same order so that, e.g., sim_seed is always
    # the seventh output of our trial's RNG.  Save the relevant ones in 
    # the output dict to be written to file.
    output['trialseed'] = trial_seed
    output['trialnum'] = trial_num
    trial_RNG = random.Random()
    trial_RNG.seed(a = trial_seed)
    g_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    b_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    p_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    w_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    t_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    res_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    sim_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    # Seeds:
    # g: graph G
    # b: beacons
    # p: paths P
    # w: weighting
    # t: test set
    # res: reserved for future use
    # sim: simulation

    output['gseed'] = g_seed
    output['bseed'] = b_seed
    output['pseed'] = p_seed
    output['wseed'] = 'na'
    output['tseed'] = 'na'
    output['simseed'] = sim_seed    

    # Below, olock is a global lock that is shared by all processes
    # running this in parallel.  It is acquired before writing to the
    # output file and released after the writing is done.

    # Construct a G_{n,m} graph using g_seed as its randomness.  Make
    # sure this is connected.
    G = jrnx.jr_gnm_random_graph(n,m,seed=g_seed)
    if not nx.is_connected(G):
        output['status'] = 'G is unexpectedly not connected!'
        with olock:
            with open(ofname, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, header_row, extrasaction='ignore')
                writer.writerow(output)
        return

    # Choose b beacons using a RNG seeded with b_seed
    beacon_RNG = random.Random()
    beacon_RNG.seed(a=b_seed)
    beacons = beacon_RNG.sample(sorted(G.nodes()),b)

    # Construct all paths in G with randomness p_seed
    all_paths = jrnx.jr_all_pairs_shortest_path(G, seed=p_seed)
    P = dict()

    # Now select only the paths that are between (distinct) beacons
    for s in beacons:
        for d in beacons:
            if d != s:
                P.setdefault(s, dict())[d] = all_paths[s][d]
    
    # Add weights to edges and nodes if desired
    if edge_attr is not None:
        weighting.add_edge_weights_from_paths(G, P, edge_attr_name=edge_attr)
        weighting.add_reciprocal_weights(G, src_attr_name=edge_attr, dst_attr_name=statTol, components='edges', use_default=True, default=0)
        if node_attr is not None:
            weighting.add_node_weights_from_edge_mean(G, node_attr_name=node_attr, edge_attr_name=edge_attr)
            weighting.add_reciprocal_weights(G, src_attr_name=node_attr, dst_attr_name=statTol, components='nodes', use_default=True, default=0)


    # Run a trial, with sim_seed as the randomness, to produce the
    # probing sequence seq
    try:
        t = timer()
        start_time = timer()
        seq = strategy.simulate(copy.deepcopy(G), copy.deepcopy(P), alpha=alpha, trialseed=sim_seed, k=k, maxsteps=maxs)
        output['simtime'] = timer() - start_time
    except Exception as e:
        output['simtime'] = timer() - start_time
        output['status'] = 'Exception raised during simulate: ' + repr(e)
        with olock:
            with open(ofname, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, header_row, extrasaction='ignore')
                writer.writerow(output)
        return

    # Compute the statistics on seq
    try:
        t = timer()
        results = stats.all_sequence_stats(seq, P, G, importance=statImp, tolerance=statTol)
    except Exception as e:
        output['status'] = 'Exception raised during stats computation ' + repr(e)
        with olock:
            with open(ofname, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, header_row, extrasaction='ignore')
                writer.writerow(output)
        return

    # Update output with the statistics and write it to file
    output.update(results)
    with olock:
        with open(ofname, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, header_row, extrasaction='ignore')
            writer.writerow(output)
    return


def setcover_ba_trial(sc_args):
    """
    Run a single setcover trial for specified B--A graph parameters
    
    :param rb_args is a tuple with the following:
    
    fractional: whether to use fractional version of B--A algorithm
    ofname: file name for output
    header_row: header for use in opening a DictWriter
    trial_num: index of this trial
    b: number of beacons to use
    n: number of graph nodes
    bam: int number of edges for each new node in B--A model
    trial_seed: main seed used for this trial
    alpha: setcover parameter alpha
    k: number of paths to probe in expectation
    output: dict with partial info about this trial; this is mutated here
    edge_attr: weight attribute for edges
    node_attr: weight attribute for nodes
    statImp: importance weight for computing result statistics
    statTol: tolerance weight for computing result statistics
    
    This generates a B--A preferential-attachment graph, chooses beacons, computes probing paths, runs a trial, computes the statistics on the probing sequence, and writes the result to ofile (after taking a lock on it so that many copies of this can be run in parallel).
    """
    (fractional, ofname, header_row, trial_num, b, n, bam, trial_seed, maxs, alpha, k, output, edge_attr, node_attr, statImp, statTol) = sc_args

    # Create the RNG for this trial and seed it with the seed from the
    # top-level RNG passed in the argument.  Even if we don't use all these
    # seeds, create them in the same order so that, e.g., sim_seed is always
    # the seventh output of our trial's RNG.  Save the relevant ones in 
    # the output dict to be written to file.
    output['trialseed'] = trial_seed
    output['trialnum'] = trial_num
    output['fractional'] = fractional
    trial_RNG = random.Random()
    trial_RNG.seed(a = trial_seed)
    g_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    b_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    p_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    w_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    t_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    res_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    sim_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    # Seeds:
    # g: graph G
    # b: beacons
    # p: paths P
    # w: weighting
    # t: test set
    # res: reserved for future use
    # sim: simulation

    output['gseed'] = g_seed
    output['bseed'] = b_seed
    output['pseed'] = p_seed
    output['wseed'] = 'na'
    output['tseed'] = 'na'
    output['simseed'] = sim_seed    

    # Below, olock is a global lock that is shared by all processes
    # running this in parallel.  It is acquired before writing to the
    # output file and released after the writing is done.

    # Construct a B--A graph using g_seed as its randomness.  Use fractional
    # version as indicated by fractional parameter.  Make
    # sure G is connected.
    if fractional:
        G = jrnx.jr_fractional_barabasi_albert_graph(n,bam,seed=g_seed)
    else:
        G = jrnx.jr_barabasi_albert_graph(n,bam,seed=g_seed)
    if not nx.is_connected(G):
        output['status'] = 'G is unexpectedly not connected!'
        with olock:
            with open(ofname, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, header_row, extrasaction='ignore')
                writer.writerow(output)
        return

    # Choose b beacons using a RNG seeded with b_seed
    beacon_RNG = random.Random()
    beacon_RNG.seed(a=b_seed)
    beacons = beacon_RNG.sample(sorted(G.nodes()),b)

    # Construct all paths in G with randomness p_seed
    all_paths = jrnx.jr_all_pairs_shortest_path(G, seed=p_seed)
    P = dict()

    # Now select only the paths that are between (distinct) beacons
    for s in beacons:
        for d in beacons:
            if d != s:
                P.setdefault(s, dict())[d] = all_paths[s][d]
    
    # Add weights to edges and nodes if desired
    if edge_attr is not None:
        weighting.add_edge_weights_from_paths(G, P, edge_attr_name=edge_attr)
        weighting.add_reciprocal_weights(G, src_attr_name=edge_attr, dst_attr_name=statTol, components='edges', use_default=True, default=0)
        if node_attr is not None:
            weighting.add_node_weights_from_edge_mean(G, node_attr_name=node_attr, edge_attr_name=edge_attr)
            weighting.add_reciprocal_weights(G, src_attr_name=node_attr, dst_attr_name=statTol, components='nodes', use_default=True, default=0)


    # Run a trial, with sim_seed as the randomness, to produce the
    # probing sequence seq
    try:
        t = timer()
        start_time = timer()
        seq = strategy.simulate(copy.deepcopy(G), copy.deepcopy(P), alpha=alpha, trialseed=sim_seed, k=k, maxsteps=maxs)
        output['simtime'] = timer() - start_time
    except Exception as e:
        output['simtime'] = timer() - start_time
        output['status'] = 'Exception raised during simulate: ' + repr(e)
        with olock:
            with open(ofname, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, header_row, extrasaction='ignore')
                writer.writerow(output)
        return

    # Compute the statistics on seq
    try:
        t = timer()
        results = stats.all_sequence_stats(seq, P, G, importance=statImp, tolerance=statTol)
    except Exception as e:
        output['status'] = 'Exception raised during stats computation ' + repr(e)
        with olock:
            with open(ofname, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, header_row, extrasaction='ignore')
                writer.writerow(output)
        return

    # Update output with the statistics and write it to file
    output.update(results)
    with olock:
        with open(ofname, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, header_row, extrasaction='ignore')
            writer.writerow(output)
    return



def setcover_trial_from_beacons(sc_args):
    """
    Run a single setcover trial when given the graph and beacon set
    
    :param rb_args is a tuple with the following:
    
    ofname: file name for output
    header_row: header for use in opening a DictWriter
    trial_num: index of this trial
    G: NetworkX graph object
    B: set of nodes from G to use as beacons
    trial_seed: main seed used for this trial
    alpha: setcover parameter alpha
    k: number of paths to probe in expectation
    output: dict with partial info about this trial; this is mutated here
    edge_attr: weight attribute for edges
    node_attr: weight attribute for nodes
    statImp: importance weight for computing result statistics
    statTol: tolerance weight for computing result statistics
    
    This chooses probing paths between the specified set of beacons, runs a trial, computes the statistics on the probing sequence, and writes the result to ofile (after taking a lock on it so that many copies of this can be run in parallel).
    """
    (ofname, header_row, trial_num, G, B, trial_seed, maxs, alpha, k, output, edge_attr, node_attr, statImp, statTol) = sc_args

    # Create the RNG for this trial and seed it with the seed from the
    # top-level RNG passed in the argument.  Even if we don't use all these
    # seeds, create them in the same order so that, e.g., sim_seed is always
    # the seventh output of our trial's RNG.  Save the relevant ones in 
    # the output dict to be written to file.
    output['trialseed'] = trial_seed
    output['trialnum'] = trial_num
    trial_RNG = random.Random()
    trial_RNG.seed(a = trial_seed)
    g_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    b_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    p_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    w_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    t_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    res_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    sim_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    # Seeds:
    # g: graph G
    # b: beacons
    # p: paths P
    # w: weighting
    # t: test set
    # res: reserved for future use
    # sim: simulation

    output['gseed'] = g_seed
    output['bseed'] = 'na'
    output['pseed'] = p_seed
    output['wseed'] = 'na'
    output['tseed'] = 'na'
    output['simseed'] = sim_seed    

    # Below, olock is a global lock that is shared by all processes
    # running this in parallel.  It is acquired before writing to the
    # output file and released after the writing is done.

    # Construct all paths in G with randomness p_seed
    all_paths = jrnx.jr_all_pairs_shortest_path(G, seed=p_seed)
    P = dict()

    # Now select only the paths that are between (distinct) beacons
    for s in B:
        for d in B:
            if d != s:
                P.setdefault(s, dict())[d] = all_paths[s][d]
    
    # Add weights to edges and nodes if desired
    if edge_attr is not None:
        weighting.add_edge_weights_from_paths(G, P, edge_attr_name=edge_attr)
        weighting.add_reciprocal_weights(G, src_attr_name=edge_attr, dst_attr_name=statTol, components='edges', use_default=True, default=0)
        if node_attr is not None:
            weighting.add_node_weights_from_edge_mean(G, node_attr_name=node_attr, edge_attr_name=edge_attr)
            weighting.add_reciprocal_weights(G, src_attr_name=node_attr, dst_attr_name=statTol, components='nodes', use_default=True, default=0)


    # Run a trial, with sim_seed as the randomness, to produce the
    # probing sequence seq
    try:
        t = timer()
        start_time = timer()
        seq = strategy.simulate(copy.deepcopy(G), copy.deepcopy(P), alpha=alpha, trialseed=sim_seed, k=k, maxsteps=maxs)
        output['simtime'] = timer() - start_time
    except Exception as e:
        output['simtime'] = timer() - start_time
        output['status'] = 'Exception raised during simulate: ' + repr(e)
        with olock:
            with open(ofname, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, header_row, extrasaction='ignore')
                writer.writerow(output)
        return

    # Compute the statistics on seq
    try:
        t = timer()
        results = stats.all_sequence_stats(seq, P, G, importance=statImp, tolerance=statTol)
    except Exception as e:
        output['status'] = 'Exception raised during stats computation ' + repr(e)
        with olock:
            with open(ofname, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, header_row, extrasaction='ignore')
                writer.writerow(output)
        return

    # Update output with the statistics and write it to file
    output.update(results)
    with olock:
        with open(ofname, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, header_row, extrasaction='ignore')
            writer.writerow(output)
    return


def setcover_trial_from_num_beacons(sc_args):
    """
    Run a single setcover trial when given the graph and number of beacons
    
    :param rb_args is a tuple with the following:
    
    ofname: file name for output
    header_row: header for use in opening a DictWriter
    trial_num: index of this trial
    G: NetworkX graph object
    num_beacons: number of nodes from G to use as beacons
    trial_seed: main seed used for this trial
    alpha: setcover parameter alpha
    k: number of paths to probe in expectation
    output: dict with partial info about this trial; this is mutated here
    edge_attr: weight attribute for edges
    node_attr: weight attribute for nodes
    statImp: importance weight for computing result statistics
    statTol: tolerance weight for computing result statistics
    
    This chooses a set of beacons of specified size, chooses probing paths between the set of beacons, runs a trial, computes the statistics on the probing sequence, and writes the result to ofile (after taking a lock on it so that many copies of this can be run in parallel).
    """
    (ofname, header_row, trial_num, G, num_beacons, trial_seed, maxs, alpha, k, output, edge_attr, node_attr, statImp, statTol) = sc_args

    # Create the RNG for this trial and seed it with the seed from the
    # top-level RNG passed in the argument.  Even if we don't use all these
    # seeds, create them in the same order so that, e.g., sim_seed is always
    # the seventh output of our trial's RNG.  Save the relevant ones in 
    # the output dict to be written to file.
    output['trialseed'] = trial_seed
    output['trialnum'] = trial_num
    trial_RNG = random.Random()
    trial_RNG.seed(a = trial_seed)
    g_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    b_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    p_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    w_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    t_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    res_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    sim_seed = trial_RNG.randint(-sys.maxint - 1, sys.maxint)
    # Seeds:
    # g: graph G
    # b: beacons
    # p: paths P
    # w: weighting
    # t: test set
    # res: reserved for future use
    # sim: simulation

    output['gseed'] = g_seed
    output['bseed'] = b_seed
    output['pseed'] = p_seed
    output['wseed'] = 'na'
    output['tseed'] = 'na'
    output['simseed'] = sim_seed    

    # Below, olock is a global lock that is shared by all processes
    # running this in parallel.  It is acquired before writing to the
    # output file and released after the writing is done.

    # Choose num_beacons beacons using a RNG seeded with b_seed
    beacon_RNG = random.Random()
    beacon_RNG.seed(a=b_seed)
    beacons = beacon_RNG.sample(sorted(G.nodes()),num_beacons)

    # Construct all paths in G with randomness p_seed
    all_paths = jrnx.jr_all_pairs_shortest_path(G, seed=p_seed)
    P = dict()

    # Now select only the paths that are between (distinct) beacons
    for s in beacons:
        for d in beacons:
            if d != s:
                P.setdefault(s, dict())[d] = all_paths[s][d]
    
    # Add weights to edges and nodes if desired
    if edge_attr is not None:
        weighting.add_edge_weights_from_paths(G, P, edge_attr_name=edge_attr)
        weighting.add_reciprocal_weights(G, src_attr_name=edge_attr, dst_attr_name=statTol, components='edges', use_default=True, default=0)
        if node_attr is not None:
            weighting.add_node_weights_from_edge_mean(G, node_attr_name=node_attr, edge_attr_name=edge_attr)
            weighting.add_reciprocal_weights(G, src_attr_name=node_attr, dst_attr_name=statTol, components='nodes', use_default=True, default=0)

    # Run a trial, with sim_seed as the randomness, to produce the
    # probing sequence seq
    try:
        t = timer()
        start_time = timer()
        seq = strategy.simulate(copy.deepcopy(G), copy.deepcopy(P), alpha=alpha, trialseed=sim_seed, k=k, maxsteps=maxs)
        output['simtime'] = timer() - start_time
    except Exception as e:
        output['simtime'] = timer() - start_time
        output['status'] = 'Exception raised during simulate: ' + repr(e)
        with olock:
            with open(ofname, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, header_row, extrasaction='ignore')
                writer.writerow(output)
        return

    # Compute the statistics on seq
    try:
        t = timer()
        results = stats.all_sequence_stats(seq, P, G, importance=statImp, tolerance=statTol)
    except Exception as e:
        output['status'] = 'Exception raised during stats computation ' + repr(e)
        with olock:
            with open(ofname, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, header_row, extrasaction='ignore')
                writer.writerow(output)
        return

    # Update output with the statistics and write it to file
    output.update(results)
    with olock:
        with open(ofname, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, header_row, extrasaction='ignore')
            writer.writerow(output)
    return

########################

stats_columns = stats.all_stats_header()


if args.zoo:
    topology_list = glob.glob(args.simroot + 'zoo/*.graphml')
    ofname = 'zoo-setcover'
    header_row = ['topology', 'trialnum', 'status', 'alpha', 'k', 'maxS', 'edgeAttr', 'nodeAttr', 'statImp', 'statTol', 'simtime', 'trialseed', 'gseed', 'bseed', 'pseed', 'wseed', 'tseed', 'simseed'] + stats_columns
    if args.test or args.local:
        topology_list = topology_list[:3]

elif args.dcell or args.spdcell:
    top_param_list = [(2,2),(3,2)] # This should be an iterable of (dcn,dck) tuples
#     top_param_list = [(1,2),(1,3),(2,2),(3,2),(4,2)] # This should be an iterable of (dcn,k) tuples
    if args.dcell:
        ofname = 'dcell-setcover'
    else:
        ofname = 'spdcell-setcover'
    header_row = ['dcn', 'dck', 'trialnum', 'status', 'alpha', 'k', 'maxS', 'edgeAttr', 'nodeAttr', 'statImp', 'statTol', 'simtime', 'trialseed', 'gseed', 'bseed', 'pseed', 'wseed', 'tseed', 'simseed'] + stats_columns
    if args.test or args.local:
        top_param_list = top_param_list[:4]

elif args.gnm:
    top_param_list = [(42,63,84), (64,192,1024), (81,401,960), (156,208,312)]
#     top_param_list = [(2,4,3)] # This should be an iterable of (b,n,m) tuples
#     top_param_list = [(6,12,12), (42,84,105), (42,63,84), (156,208,312), (2,4,3), (9,25,48), (16,36,80), (16,46,120), (27,105,234), (6,9,9), (12,16,18), (16,48,128), (64,256,768), (64,190,504), (20,25,30)] # This should be an iterable of (b,n,m) tuples
    ofname = 'gnm-setcover'
    header_row = ['b', 'n', 'm', 'trialnum', 'status', 'alpha', 'k', 'maxS', 'edgeAttr', 'nodeAttr', 'statImp', 'statTol', 'simtime', 'trialseed', 'gseed', 'bseed', 'pseed', 'wseed', 'tseed', 'simseed'] + stats_columns
    if args.test or args.local:
        top_param_list = top_param_list[:4]

elif args.xgft:
    top_param_list = [(4,[3,3,3,3],[8,1,1,1]), (2,[8,8],[8,8])]
#     top_param_list = [(2,[3,3],[4,1]),(2,[4,4],[6,1]),(3,[3,3,3],[6,1,1]),(2,[4,4],[4,4]),(3,[4,4,4],[4,4,4]),(3,[4,4,4],[6,1,1])]
    ofname = 'xgft-setcover'
    header_row = ['h', 'mlist', 'wlist', 'trialnum', 'status', 'alpha', 'k', 'maxS', 'edgeAttr', 'nodeAttr', 'statImp', 'statTol', 'simtime', 'trialseed', 'gseed', 'bseed', 'pseed', 'wseed', 'tseed', 'simseed'] + stats_columns
    if args.test or args.local:
        top_param_list = top_param_list[:4]

elif args.ba:
#     top_param_list = [(2,4,3)] # This should be an iterable of (b,n,m) tuples
    top_param_list = [(6,12,1), (42,84,1), (42,63,1), (156,208,2), (2,4,1), (9,25,2), (16,36,2), (16,46,3), (27,105,2), (6,9,1), (12,16,1), (16,48,3), (64,256,3), (64,190,3), (20,25,1)] # This should be an iterable of (b,n,m) tuples
    ofname = 'ba-setcover'
    header_row = ['fractional', 'b', 'n', 'bam', 'trialnum', 'status', 'alpha', 'k', 'maxS', 'edgeAttr', 'nodeAttr', 'statImp', 'statTol', 'simtime', 'trialseed', 'gseed', 'bseed', 'pseed', 'wseed', 'tseed', 'simseed'] + stats_columns
    if args.test or args.local:
        top_param_list = top_param_list[:3]

elif args.fracba:
    top_param_list = [(42,63,1.36281), (81,401,2.40848), (64,192,5.49033), (156,208,1.51098)] # This should be an iterable of (b,n,bam) tuples
#     top_param_list = [(2,4,3)] # This should be an iterable of (b,n,m) tuples
#     top_param_list = [(6,12,1.10102), (42,84,1.26918), (42,63,1.36281), (156,208,1.51098), (2,4,1.0), (9,25,2.09567), (16,36,2.3795), (16,46,2.77625), (27,105,2.27799), (6,9,1.1459), (12,16,1.21767), (16,48,2.83399), (64,256,3.03601), (64,190,2.69074), (20,25,1.2639)] # This should be an iterable of (b,n,m) tuples
    ofname = 'fracba-setcover'
    header_row = ['fractional', 'b', 'n', 'bam', 'trialnum', 'status', 'alpha', 'k', 'maxS', 'edgeAttr', 'nodeAttr', 'statImp', 'statTol', 'simtime', 'trialseed', 'gseed', 'bseed', 'pseed', 'wseed', 'tseed', 'simseed'] + stats_columns
    if args.test or args.local:
        top_param_list = top_param_list[:4]
    
elif args.smallcat:
    icatalog = args.simroot + 'catalog/inputcat-small.csv'
    gcatalog = args.simroot + 'catalog/graphcat-small.csv'
    ofname = 'smallcat-setcover'
    header_row = ['IID', 'trialnum', 'status', 'alpha', 'k', 'maxS', 'edgeAttr', 'nodeAttr', 'statImp', 'statTol', 'simtime', 'trialseed', 'gseed', 'bseed', 'pseed', 'wseed', 'tseed', 'simseed'] + stats_columns
    if args.test or args.local:
        catalog_range = {'start': 1, 'stop': 4, 'step': 1}
    else:
        catalog_range = {'start': 1, 'stop': None, 'step': 1}
    
# The first few entries in the catalog only have a single path    
elif args.cat:
    icatalog = args.simroot + 'catalog/inputcat.csv'
    gcatalog = args.simroot + 'catalog/graphcat.csv'
    ofname = 'cat-setcover'
    header_row = ['IID', 'trialnum', 'status', 'alpha', 'k', 'maxS', 'edgeAttr', 'nodeAttr', 'statImp', 'statTol', 'simtime', 'trialseed', 'gseed', 'bseed', 'pseed', 'wseed', 'tseed', 'simseed'] + stats_columns
    if args.test or args.local:
        catalog_range = {'start': 4, 'stop': 8, 'step': 1}
    else:
        catalog_range = {'start': 1, 'stop': None, 'step': 1}


if args.test or args.local:
    aList = [40] # could use [0,50,100]
elif args.seta is None:
    aList = [0,25,40,50,75,100]
else:
    aList = [args.seta]


# Value(s) of k to use here
if args.k is None:
    kList = [3]
else:
    kList = [args.k]

if args.maxs is None:
    maxs = 400
else:
    maxs = args.maxs


if args.test:
    number_of_trials = 3
elif args.local:
    number_of_trials = 25
    tstr = 'T25/'
elif args.T100:
    number_of_trials = 100
    tstr = 'T100/'
elif args.T316:
    number_of_trials = 316
    tstr = 'T316/'
elif args.T1000:
    number_of_trials = 1000
    tstr = 'T1000/'
elif args.T3162:
    number_of_trials = 3162
    tstr = 'T3162/'
elif args.T10000:
    number_of_trials = 10000
    tstr = 'T10000/'


odir_test = args.simroot + 'testdata/'
odir = args.simroot + 'data/'
odir_server = args.simroot + 'server-data/'
odir_local = args.simroot + 'local-data/'


if args.test:
    ofile_base = odir_test + ofname
    if not isdir(odir_test):
        makedirs(odir_test)
elif args.server:
    ofile_base = odir_server + tstr + ofname
    if not isdir(odir_server + tstr):
        makedirs(odir_server + tstr)
elif args.local:
    ofile_base = odir_local + tstr + ofname
    if not isdir(odir_local + tstr):
        makedirs(odir_local + tstr)
else: # Default: Local, but not testing
    ofile_base = odir + tstr + ofname
    if not isdir(odir + tstr):
        makedirs(odir + tstr)

    
# Various weights
# The corresponding strings are so that we can easily
# write 'None' to file if the weight is None
statImp = 'weight'
if statImp is None:
    statImpStr = 'None'
else:
    statImpStr = statImp

statTol = 'inv_weight'
if statTol is None:
    statTolStr = 'None'
else:
    statTolStr = statTol

# Do we weight edges and nodes?
edge_attr = statImp
if edge_attr is None:
    edge_attr_str = 'None'
else:
    edge_attr_str = edge_attr

node_attr = statImp
if node_attr is None:
    node_attr_str = 'None'
else:
    node_attr_str = node_attr


if args.nprocs is None:
    nprocs = mp.cpu_count()
else:
    nprocs = args.nprocs


# Initialization for pool processes.  Make the mp.Lock() global
# and accessible to all pool processes.  This lock will be acquired
# by each process before it writes to the common output file.
def pool_init(l):
    global olock
    olock = l

writer_lock = mp.Lock()

for k in kList:
    for a in aList:
        # Identify the header file and write the header row to it
        #
        # Use file_idx to distinguish different output files for the
        # same parameters, e.g., from different data runs (which might
        # contain different specific topologies, etc.).  Duplicates
        # between files will be filtered out during analysis.
        ofile_sans_idx = ofile_base + '-k%03d-s%04d-a%03d-' % (k,maxs,a)
        file_idx = 0
        ofile = ofile_sans_idx + str(file_idx) + '.csv'
        while isfile(ofile):
            file_idx += 1
            ofile = ofile_sans_idx + str(file_idx) + '.csv'
        
        alpha = float(a)/100.

        with open(ofile, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, header_row, extrasaction='ignore')
            writer.writeheader()

        # Set up the pool of processes
        pool = mp.Pool(processes=nprocs, initializer = pool_init, initargs=(writer_lock,), maxtasksperchild=1)


        if args.zoo:

            for topofile in topology_list:

                # For each topology, read in the graph G
                # Use all nodes as beacons.
                G = ps.largest_connected_component(nx.Graph(nx.read_graphml(topofile)))
                num_beacons = len(G.nodes())

                # Set up the top-level RNG.  Outputs from this will be used to seed
                # the different trials.
                control_RNG = random.Random()
                control_RNG.seed(a=1)

                output = {'topology': topofile, 'alpha':alpha, 'k':k, 'maxS':maxs, 'edgeAttr':edge_attr_str, 'nodeAttr':node_attr_str, 'statImp':statImpStr, 'statTol':statTolStr}

                # Set up a generator that produces arguments to the trial
                # function.  Call pool.map to apply the trial function to these
                # arguments
                trial_arg_gen = ((ofile, header_row, i, G, num_beacons, control_RNG.randint(-sys.maxint - 1, sys.maxint), maxs, alpha, k, copy.deepcopy(output), edge_attr, node_attr, statImp, statTol) for i in range(number_of_trials))
            
                pool.map(setcover_trial_from_num_beacons, trial_arg_gen)



        elif args.cat or args.smallcat:

            IID_list, tuples_list = inputgen.inputgen_from_catalog_range(icatalog, gcatalog, **catalog_range)
            for j in range(len(IID_list)):
            
                # Here, read in G and P from the catalog.
                # Add weights, set up the RNG, and start the parallel
                # processes.  Here, we use the trial function for the case
                # where we already have the paths.
            
                IID = IID_list[j]

                G = nx.Graph(tuples_list[j][0])
                P = ps.removeSelfLoops(tuples_list[j][2])

                if edge_attr is not None:
                    weighting.add_edge_weights_from_paths(G, P, edge_attr_name=edge_attr)
                    weighting.add_reciprocal_weights(G, src_attr_name=edge_attr, dst_attr_name=statTol, components='edges', use_default=True, default=0)
                    if node_attr is not None:
                        weighting.add_node_weights_from_edge_mean(G, node_attr_name=node_attr, edge_attr_name=edge_attr)
                        weighting.add_reciprocal_weights(G, src_attr_name=node_attr, dst_attr_name=statTol, components='nodes', use_default=True, default=0)


                control_RNG = random.Random()
                control_RNG.seed(a=1)

                output = {'IID': IID, 'alpha':alpha, 'k':k, 'maxS':maxs, 'edgeAttr':edge_attr_str, 'nodeAttr':node_attr_str, 'statImp':statImpStr, 'statTol':statTolStr}

                trial_arg_gen = ((ofile, header_row, i, control_RNG.randint(-sys.maxint - 1, sys.maxint), maxs, P, G, alpha, k, copy.deepcopy(output), statImp, statTol) for i in range(number_of_trials))
            
                pool.map(setcover_trial_from_paths, trial_arg_gen)


        elif args.dcell:
            for (dcn,dck) in top_param_list:
            
                # Here, generate G from parameters and P from the
                # built-in DCell routing.
                # Add weights, set up the RNG, and start the parallel
                # processes.  Here, we use the trial function for the case
                # where we already have the paths.
            
                G = dcell.DCellGraph(dcn,dck)
                P = ps.removeSelfLoops(dcell.all_server_pairs_DCell_routes(G,dcn,dck))

                if edge_attr is not None:
                    weighting.add_edge_weights_from_paths(G, P, edge_attr_name=edge_attr)
                    weighting.add_reciprocal_weights(G, src_attr_name=edge_attr, dst_attr_name=statTol, components='edges', use_default=True, default=0)
                    if node_attr is not None:
                        weighting.add_node_weights_from_edge_mean(G, node_attr_name=node_attr, edge_attr_name=edge_attr)
                        weighting.add_reciprocal_weights(G, src_attr_name=node_attr, dst_attr_name=statTol, components='nodes', use_default=True, default=0)


                control_RNG = random.Random()
                control_RNG.seed(a=1)

                output = {'dcn':dcn, 'dck':dck, 'alpha':alpha, 'k':k, 'maxS':maxs, 'edgeAttr':edge_attr_str, 'nodeAttr':node_attr_str, 'statImp':statImpStr, 'statTol':statTolStr}

                trial_arg_gen = ((ofile, header_row, i, control_RNG.randint(-sys.maxint - 1, sys.maxint), maxs, P, G, alpha, k, copy.deepcopy(output), statImp, statTol) for i in range(number_of_trials))
            
                pool.map(setcover_trial_from_paths, trial_arg_gen)


        elif args.spdcell:
            for (dcn,dck) in top_param_list:
            
                # Here, generate G from parameters.  B is the collection
                # of all non-switches in the DCell.
                # Add weights, set up the RNG, and start the parallel
                # processes.  Here, we use the trial function for the case
                # where we already have the beacon set.
            
                G = dcell.DCellGraph(dcn,dck)
                B = []
                for v in sorted(G.nodes()):
                    if dcell.IsNotSwitch(v):
                        B.append(v)

                control_RNG = random.Random()
                control_RNG.seed(a=1)

                output = {'dcn':dcn, 'dck':dck, 'alpha':alpha, 'k':k, 'maxS':maxs, 'edgeAttr':edge_attr_str, 'nodeAttr':node_attr_str, 'statImp':statImpStr, 'statTol':statTolStr}

                trial_arg_gen = ((ofile, header_row, i, G, B, control_RNG.randint(-sys.maxint - 1, sys.maxint), maxs, alpha, k, copy.deepcopy(output), edge_attr, node_attr, statImp, statTol) for i in range(number_of_trials))
            
                pool.map(setcover_trial_from_beacons, trial_arg_gen)


        elif args.xgft:
            for (h, mList, wList) in top_param_list:
            
                # Here, generate G from parameters.  B is the collection
                # of all non-switches in the DCell.
                # Add weights, set up the RNG, and start the parallel
                # processes.  Here, we use the trial function for the case
                # where we already have the beacon set.
            
                G = xgft.XGFTgraph(h,mList,wList)
                B = []
                for v in sorted(G.nodes()):
                    if v[0] == 0:
                        B.append(v)

                control_RNG = random.Random()
                control_RNG.seed(a=1)

                output = {'h':h, 'mlist':mList, 'wlist':wList, 'alpha':alpha, 'k':k, 'maxS':maxs, 'edgeAttr':edge_attr_str, 'nodeAttr':node_attr_str, 'statImp':statImpStr, 'statTol':statTolStr}

                trial_arg_gen = ((ofile, header_row, i, G, B, control_RNG.randint(-sys.maxint - 1, sys.maxint), maxs, alpha, k, copy.deepcopy(output), edge_attr, node_attr, statImp, statTol) for i in range(number_of_trials))
            
                pool.map(setcover_trial_from_beacons, trial_arg_gen)


        elif args.gnm:
            for (b,n,m) in top_param_list:
            
                # Set up the top-level RNG.  A seedfile might be given
                # with precomputed top-level seeds that produce connected
                # graphs for the (n,m) parameters in question.  G is generated
                # inside the trial function.
                # Here, we use the trial function for the G_{n,m} case.
            

                control_RNG = random.Random()
                control_RNG.seed(a=1)
            
                output = {'b':b, 'n':n, 'm':m, 'alpha':alpha, 'k':k, 'maxS':maxs, 'edgeAttr':edge_attr_str, 'nodeAttr':node_attr_str, 'statImp':statImpStr, 'statTol':statTolStr}

                if args.seedfile:
                    seed_list = []
                    seed_ifile = './gnm-rng/gnm-seeds-' + str(b) + '-' + str(n) + '-' + str(m) + '.csv'
                    with open(seed_ifile) as istream:
                        reader = csv.DictReader(istream)
                        i = 0
                        for row in reader:
                            seed_list += [int(row['proc_seed'])]
                            i += 1
                            if i == number_of_trials:
                                break

                    gnm_trial_arg_gen = ((ofile, header_row, i, b, n, m, seed_list[i], maxs, alpha, k, copy.deepcopy(output), edge_attr, node_attr, statImp, statTol) for i in range(number_of_trials))            
                else:
                    gnm_trial_arg_gen = ((ofile, header_row, i, b, n, m, control_RNG.randint(-sys.maxint - 1, sys.maxint), maxs, alpha, k, copy.deepcopy(output), edge_attr, node_attr, statImp, statTol) for i in range(number_of_trials))            
            
                pool.map(setcover_gnm_trial, gnm_trial_arg_gen)


        elif args.ba or args.fracba:
            for (b,n,bam) in top_param_list:
            
                # Here, generate G randomly in the trial, so we just set up
                # the top-level RNG.  No seedfile is expected because these
                # graphs are always connected by construction.
                # Here, we use the trial function for the B--A or fractional
                # B--A case.  In the former, bam is an int; in the latter, it
                # is a float.
            
                control_RNG = random.Random()
                control_RNG.seed(a=1)
            
                output = {'b':b, 'n':n, 'bam':bam, 'alpha':alpha, 'k':k, 'maxS':maxs, 'edgeAttr':edge_attr_str, 'nodeAttr':node_attr_str, 'statImp':statImpStr, 'statTol':statTolStr}

                # Now produce argument generator for setcover_ba_trial.  The first
                # field in the tuple indicates whether to use the fractional
                # version.
                if args.ba:
                    ba_trial_arg_gen = ((False, ofile, header_row, i, b, n, bam, control_RNG.randint(-sys.maxint - 1, sys.maxint), maxs, alpha, k, copy.deepcopy(output), edge_attr, node_attr, statImp, statTol) for i in range(number_of_trials))            
                elif args.fracba:
                    ba_trial_arg_gen = ((True, ofile, header_row, i, b, n, bam, control_RNG.randint(-sys.maxint - 1, sys.maxint), maxs, alpha, k, copy.deepcopy(output), edge_attr, node_attr, statImp, statTol) for i in range(number_of_trials))            
           
                pool.map(setcover_ba_trial, ba_trial_arg_gen)


        else:
            raise RuntimeError('No supported topologies specified!')

        # We're now done with assigning tasks to the pool.
        pool.close()
        pool.join()
