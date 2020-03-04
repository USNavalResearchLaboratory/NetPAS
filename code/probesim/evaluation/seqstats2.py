import math
import networkx as nx

import pandas
import numpy as np

from .. import util as ps
from .. import pathcollection as pc
import probestats as st

# ---------- dictionary iteration ----------------
if hasattr(dict, 'itervalues'):
    def itervalues(d):
        return d.itervalues()
else:
    def itervalues(d):
        return d.values()

if hasattr(dict, 'iteritems'):
    def iteritems(d):
        return d.iteritems()
else:
    def iteritems(d):
        return d.items()

# ---------- weight lookup helper function ----------------
def get_weight(G, path_collection, edge_label, weight_attr):
    u, v = path_collection.from_edge_ids[edge_label]
    return G[u][v][weight_attr]


# ---------- aggregation labels ------------------

agglabel_bytime = lambda s: s + '_time'
agglabel_bycomp = lambda s: s + '_comp'
agglabel_bycompwt = lambda s: s + '_comp_wt'

def agglabel_simple(stat, method, prefix=None):
    label = "{method}({stat})".format(method=method, stat=stat)
    if prefix is None: return label
    else: return str(prefix) + '_' + label

def agglabel_full(stat, first, second, prefix=None):
    label = "{second}({first}({stat}))".format(second=second, first=first, stat=stat)
    if prefix is None: return label
    else: return str(prefix) + '_' + label

aggfuncs = ('min', 'max', 'mean', 'std')
aggquants = {0.5: 'median', 0.05: 'pct05', 0.95: 'pct95'}

# ---------- aggregation code --------------------

def double_aggregate(dataframes, comp_weights, weight_type, statlabel, prefix=None):
    stats_output = dict()

    for ct, df in dataframes.items():
        # aggregate over time first (keep components as columns/axis-1)
        first_aggregation = df.quantile(aggquants.keys()).rename(aggquants, inplace=False)
        first_aggregation = first_aggregation.append(df.agg(aggfuncs))
        first_aggregation.rename(mapper=agglabel_bytime, axis=0, inplace=True)

        # then aggregate those over component
        second_aggregation = first_aggregation.quantile(aggquants.keys(), axis=1).rename(aggquants, inplace=False)
        second_aggregation = second_aggregation.append(first_aggregation.T.agg(aggfuncs))
        second_aggregation.rename(mapper=agglabel_bycomp, axis=0, inplace=True)

        # add unweighted stats to output dictionary
        for first in second_aggregation:
            stats_output.update(second_aggregation[first].rename(lambda s: agglabel_full(ct+statlabel, first, s, prefix)))

        # use tolerances for second aggregation by component
        first_aggregation *= comp_weights[ct+'_'+weight_type]

        second_aggregation = first_aggregation.quantile(aggquants.keys(), axis=1).rename(aggquants, inplace=False)
        second_aggregation = second_aggregation.append(first_aggregation.T.agg(aggfuncs))
        second_aggregation.rename(mapper=agglabel_bycompwt, axis=0, inplace=True)

        # add weighted stats to output dictionary
        for first in second_aggregation:
            stats_output.update(second_aggregation[first].rename(lambda s: agglabel_full(ct+statlabel, first, s, prefix)))

        # aggregate over component first
        first_aggregation = df.quantile(aggquants.keys(), axis=1).rename(aggquants, inplace=False)
        first_aggregation = first_aggregation.append(df.T.agg(aggfuncs)).T
        first_aggregation.rename(mapper=agglabel_bycomp, axis=1, inplace=True)

        # then aggregate those over time
        second_aggregation = first_aggregation.quantile(aggquants.keys()).rename(aggquants, inplace=False)
        second_aggregation = second_aggregation.append(first_aggregation.agg(aggfuncs))
        second_aggregation.rename(mapper=agglabel_bytime, axis=0, inplace=True)

        # add unweighted stats to output dictionary
        for first in second_aggregation:
            stats_output.update(second_aggregation[first].rename(lambda s: agglabel_full(ct+statlabel, first, s, prefix)))

        # aggregate over component in a weighted fashion first
        df_weighted = df * comp_weights[ct+'_'+weight_type]
        first_aggregation = df_weighted.quantile(aggquants.keys(), axis=1).rename(aggquants, inplace=False)
        first_aggregation = first_aggregation.append(df_weighted.T.agg(aggfuncs)).T
        first_aggregation.rename(mapper=agglabel_bycompwt, axis=1, inplace=True)

        # then aggregate those over time
        second_aggregation = first_aggregation.quantile(aggquants.keys()).rename(aggquants, inplace=False)
        second_aggregation = second_aggregation.append(first_aggregation.agg(aggfuncs))
        second_aggregation.rename(mapper=agglabel_bytime, axis=0, inplace=True)

        # add weighted stats to output dictionary
        for first in second_aggregation:
            stats_output.update(second_aggregation[first].rename(lambda s: agglabel_full(ct+statlabel, first, s, prefix)))

    return stats_output

# ------------------------------------------------

def all_sequence_stats(seq, pd, G=None, importance=None, tolerance=None,
                        edge_importance_attr=None, edge_tolerance_attr=None,
                        node_importance_attr=None, node_tolerance_attr=None,
                        beacon_importance_attr=None, beacon_tolerance_attr=None):
    """

    :param seq: a list of *collections* of ordered pairs.
            - The list is over time, such that the entry for each index corresponds to the paths selected at that timestep.
            - Each ordered pair in a collection corresponds to the (src, dst) pair of a path in pd that is probed in that timestep.
    :param pd: probing-path dictionary
    :param G: the graph topology
    :param importance: the name of the attribute in G containing component importance weights (if None, use 1)
                       superseded by individual component arguments for attributes, if present
    :param tolerance: the name of the attribute in G containing component load-tolerance weights (if None, use 1)
                       superseded by individual component arguments for attributes, if present
    :param edge_importance_attr: the name of the attribute in G containing link importance weights
                                 if omitted or None, use the value of the attribute name provided for the 'importance' keyword argument,
                                   or 1 if that is None or omitted also (the "unweighted" case)
    :param edge_tolerance_attr: the name of the attribute in G containing link tolerance weights
                                if omitted or None, use the value of the attribute name provided for the 'tolerance' keyword argument,
                                  or 1 if that is None or omitted also (the "unweighted" case)
    :param node_importance_attr: the name of the attribute in G containing node importance weights
                                 if omitted or None, use the value of the attribute name provided for the 'importance' keyword argument,
                                   or 1 if that is None or omitted also (the "unweighted" case)
    :param node_tolerance_attr: the name of the attribute in G containing node tolerance weights
                                if omitted or None, use the value of the attribute name provided for the 'tolerance' keyword argument,
                                  or 1 if that is None or omitted also (the "unweighted" case)
    :param beacon_importance_attr: the name of the attribute in G containing beacon importance weights
                                   if omitted or None, use the value of the attribute name provided for the 'importance' keyword argument,
                                     or 1 if that is None or omitted also (the "unweighted" case)
    :param beacon_tolerance_attr: the name of the attribute in G containing beacon tolerance weights
                                  if omitted or None, use the value of the attribute name provided for the 'tolerance' keyword argument,
                                    or 1 if that is None or omitted also (the "unweighted" case)


    :return: a dictionary of computed stats
            - Keys are names of stats
            - Values are computed stats
            - Some stats are unweighted; for these, we treat importance and tolerance as 1 everywhere
            - Weighted stats are included if weights are available:
                - the component loads are multiplied by tolerance weights
                - the component delays are multiplied by importance weights
    """

    all_stats = dict()

    # topology properties, if G is not None
    if G is not None:
        all_stats['G_n'] = G.number_of_nodes()
        all_stats['G_e'] = G.number_of_edges()
        all_stats['G_degrees'] = st.freqtodist(G.degree())


    # probing path set properties

    ## Produce total edge, node, beacon, and distinct path sets from path dictionary
    path_collection = pc.PathCollection(pd)
    total_sets = {'e': path_collection.get_edge_id_set(),
                  'n': path_collection.get_node_id_set(),
                  'r': path_collection.get_endpoint_set()}

    for ct in ('e', 'n', 'r'):
        all_stats['P_'+ct] = len(total_sets[ct])
    all_stats['P_paths'] = len(path_collection)
    

    # weight properties
    comp_weights = dict()

    if G is None or (edge_importance_attr is None and importance is None):
            comp_weights['e_imp'] = pandas.Series(data=1, index=total_sets['e'])
    else:
        weight_attr = importance if edge_importance_attr is None else edge_importance_attr
        comp_weights['e_imp'] = pandas.Series(data={e : get_weight(G, path_collection, e, weight_attr) for e in total_sets['e']})

    if G is None or (node_importance_attr is None and importance is None):
            comp_weights['n_imp'] = pandas.Series(data=1, index=total_sets['n'])
    else:
        weight_dictionary = nx.get_node_attributes(G, importance if node_importance_attr is None else node_importance_attr)
        comp_weights['n_imp'] = pandas.Series(data={n : weight_dictionary[path_collection.from_node_ids[n]] for n in total_sets['n']})

    if G is None or (beacon_importance_attr is None and importance is None):
            comp_weights['r_imp'] = pandas.Series(data=1, index=total_sets['r'])
    else:
        weight_dictionary = nx.get_node_attributes(G, importance if beacon_importance_attr is None else beacon_importance_attr)
        comp_weights['r_imp'] = pandas.Series(data={r : weight_dictionary[path_collection.from_node_ids[r]] for r in total_sets['r']})


    if G is None or (edge_tolerance_attr is None and tolerance is None):
            comp_weights['e_tol'] = pandas.Series(data=1, index=total_sets['e'])
    else:
        weight_attr = tolerance if edge_tolerance_attr is None else edge_tolerance_attr
        comp_weights['e_tol'] = pandas.Series(data={e : get_weight(G, path_collection, e, weight_attr) for e in total_sets['e']})

    if G is None or (node_tolerance_attr is None and tolerance is None):
            comp_weights['n_tol'] = pandas.Series(data=1, index=total_sets['n'])
    else:
        weight_dictionary = nx.get_node_attributes(G, tolerance if node_tolerance_attr is None else node_tolerance_attr)
        comp_weights['n_tol'] = pandas.Series(data={n : weight_dictionary[path_collection.from_node_ids[n]] for n in total_sets['n']})

    if G is None or (beacon_tolerance_attr is None and tolerance is None):
            comp_weights['r_tol'] = pandas.Series(data=1, index=total_sets['r'])
    else:
        weight_dictionary = nx.get_node_attributes(G, tolerance if beacon_tolerance_attr is None else beacon_tolerance_attr)
        comp_weights['r_tol'] = pandas.Series(data={r : weight_dictionary[path_collection.from_node_ids[r]] for r in total_sets['r']})


    for ct in ('e', 'n', 'r'):
        for wt in ('imp', 'tol'):
            stat = ct+'_'+wt
            all_stats['W_'+stat+'_min'] = comp_weights[stat].min()
            all_stats['W_'+stat+'_max'] = comp_weights[stat].max()
            all_stats['W_'+stat+'_sum'] = comp_weights[stat].sum()


    # simulation time

    maxtime = len(seq)
    all_stats['T_maxsteps'] = maxtime

    # --- Go through probing sequence and build pandas dataframes of the probing events ---

    # create frames for beacons, nodes, edges
    probe_frames = dict()
    for ct in ('e', 'n', 'r'):
        probe_frames[ct] = pandas.DataFrame(data=0, index=pandas.RangeIndex(maxtime), columns = total_sets[ct], dtype=int)

    # data structure for tracking probed paths
    probed_paths = set()

    # iterate through each timestep of the probing sequence
    for timestep in range(maxtime):

        # go through each probed path
        for s, d in seq[timestep]:
            path_id = path_collection.get_path_id(s, d)

            # record that this path was used
            probed_paths.add(path_id)

            # add to probe count for components in current timestep
            probe_frames['r'].loc[timestep][{path_collection.get_node_id(s), path_collection.get_node_id(d)}] += 1
            probe_frames['e'].loc[timestep][path_collection.get_edge_ids_from_path_id(path_id)] += 1
            probe_frames['n'].loc[timestep][path_collection.get_node_ids_from_path_id(path_id)] += 1

    # --- Analyze dataframes for statistics ---

    # items not probed
    for ct, df in probe_frames.items():
        unprobed = df.columns[df.any()==False]
        all_stats['U_unprobed_{}_num'.format(ct)] = len(unprobed)
        all_stats['U_unprobed_{}_wt'.format(ct)] = comp_weights[ct+'_imp'][unprobed].sum()
    
    # paths not probed
    unused_paths = path_collection.get_path_id_set() - probed_paths
    all_stats['U_unused_paths_num'] = len(unused_paths)
    all_stats['U_unused_paths_wt'] = sum(comp_weights['e_imp'][path_collection.get_edge_ids_from_path_id(path_id)].sum() for path_id in unused_paths)

    # initialcovertime
    all_covered = (probe_frames['e'].cummax() > 0).all(axis=1)
    initialcovertime = all_covered.idxmax()
    all_stats['T_initialcovertime'] = initialcovertime if all_covered[initialcovertime] else np.inf

    # load statistics
    all_stats.update(double_aggregate(probe_frames, comp_weights, 'tol', 'load', prefix='L'))

    # staleness statistics
    stale_frames = dict()
    for ct, df in probe_frames.items():
        b = (df==0)
        c = b.cumsum()
        stale_frames[ct] = c.sub(c.mask(b == True).ffill(), fill_value=0)
    all_stats.update(double_aggregate(stale_frames, comp_weights, 'imp', 'stale', prefix='S'))

    # probing rate statistics
    for ct, df in probe_frames.items():
        # unweighted probing rates
        rates = (df>0).mean()
        agg = rates.quantile(aggquants.keys()).rename(aggquants, inplace=False)
        agg = agg.append(rates.agg(aggfuncs)).rename(agglabel_bycomp)
        all_stats.update(agg.rename(lambda s: agglabel_simple(ct+'rate', s, prefix='R')))

        # weighted probing rates
        rates *= comp_weights[ct+'_imp']
        agg = rates.quantile(aggquants.keys()).rename(aggquants, inplace=False)
        agg = agg.append(rates.agg(aggfuncs)).rename(agglabel_bycompwt)
        all_stats.update(agg.rename(lambda s: agglabel_simple(ct+'rate', s, prefix='R')))

    return all_stats


# -------- dummy topology and header generation ----------

def dummy_arguments():
    G = nx.Graph()
    G.add_node(0, weight=1)
    G.add_node(1, weight=1)
    G.add_edge(0, 1, weight=1)
    P = {0: {1: [0, 1]}, 1: {0: [1, 0]}}
    seq = [{(0,1)}]
    return G, P, seq

def enum(*args):
    return {a: i for i, a in enumerate(args)}

preferred_order = enum('G', 'P', 'W', 'T', 'U', 'L', 'S', 'R')
preferred_ordering = lambda col: (preferred_order.get(col[0], len(preferred_order)), col[2:])

def all_stats_header():
    G, P, seq = dummy_arguments()
    all_stats = all_sequence_stats(seq, P, G, 'weight', 'weight')

    return sorted(all_stats, key=preferred_ordering)


# -------- for compatibility with original seqstats ----------

def remove_weighted_stats(all_stats):
    return {stat: val for stat, val in all_stats.items() if not ('W_' in stat or '_wt' in stat)}


def basic_stats(seq, pd, G=None):
    '''
    Input:
        seq is a list of sets of ordered pairs.
            - The list is over time, such that the entry for each index corresponds to the paths selected at that timestep.
            - Each ordered pair in a set corresponds to the (src, dst) pair of a path in pd that is probed in that timestep.
        pd is the probing-path dictionary.
        G (optional) is the network topology
    '''
    all_stats = all_sequence_stats(seq, pd, G)
    all_stats = remove_weighted_stats(all_stats)
    return [all_stats[stat] for stat in sorted(all_stats, key=preferred_ordering)]


def basic_stats_header(topology_metrics=False):
    G, P, seq = dummy_arguments()
    all_stats = all_sequence_stats(seq, P, (G if topology_metrics else None))
    all_stats = remove_weighted_stats(all_stats)
    return sorted(all_stats, key=preferred_ordering)


def weighted_stats(seq, pd, G=None, importance=None, tolerance=None):
    """

    :param seq: a list of sets of ordered pairs.
            - The list is over time, such that the entry for each index corresponds to the paths selected at that timestep.
            - Each ordered pair in a set corresponds to the (src, dst) pair of a path in pd that is probed in that timestep.
    :param pd: probing-path dictionary
    :param G: the graph topology.  If None, then basic stats are computed.
    :param importance: the name of the attribute in G containing component importance weights (if None, use 1)
    :param tolerance: the name of the attribute in G containing component load-tolerance weights (if None, use 1)
    :return: a list of computed weighted stats, corresponding to the fields described by weighted_stats_header
            - the unweighted set of stats is the same as in basic_stats (treat importance and tolerance as 1 everywhere)
            - the component loads are multiplied by tolerance weights
            - the component delays are multiplied by importance weights
    """

    all_stats = all_sequence_stats(seq, pd, G, importance, tolerance)
    return [all_stats[stat] for stat in sorted(all_stats, key=preferred_ordering)]

def weighted_stats_header():
    return all_stats_header()
