from functools import reduce
import networkx as nx

def identity(*args):
    if len(args) == 1:
        return args[0]
    return args


def er(e):
    """
    This function returns the reverse of an edge represented as a tuple
    """
    s, d = e
    return (d, s)


def cer(e):
    """
    Canonicalize the representation of an undirected edge.

    Works by returning a sorted tuple.
    """
    return tuple(sorted(e))


def countedgesin(path, edges, directed=False):
    """
    This function takes a path and a collection of edges, and returns
    the number of edges in that path that appear in the edge list
    """

    es = set(edges)

    return reduce(lambda count, edge: count + int(edge in es or (not directed and er(edge) in es)), edgesin(path), 0)

def triple_count_from_pd(pd, directed=False, return_sets=False):
    """
    This function combines the functionality of
    edgesinpd, nodesinpd, probesinpd
    by returning frequency count dictionaries
    (or sets, depending on the parameter)
    of edges, nodes, and probing nodes
    in the given path dictionary.

    This functionality matches setting repeat=False, trim=True, transit=False
    as named parameters for the above functions.  For transit==True,
    subtract probesinpd (return[2]) from nodesinpd (return[1]).

    The directed parameter can be set to True (it is False by default)
    to treat the graph as directed and so edges and their reverses
    will be recorded separately.

    It is more efficient because it iterates through the path dictionary just once
    to compute all three sets.
    """
    edge_dict, node_dict, probe_dict = dict(), dict(), dict()

    for s in pd:
        for d in pd[s]:
            only_loops = True
            if d != s:
                only_loops = False
                probe_dict[d] = probe_dict.get(d, 0) + 1
                node_dict[d] = node_dict.get(d, 0) + 1
                path = pd[s][d]
                for i in range(len(path) - 1):
                    node = path[i]
                    node_dict[node] = node_dict.get(node, 0) + 1
                    edge = (node, path[i + 1])
                    if not directed: edge = cer(edge)
                    edge_dict[edge] = edge_dict.get(edge, 0) + 1

            if not only_loops:
                probe_dict[s] = probe_dict.get(s, 0) + 1

    if return_sets:
        return set(edge_dict), set(node_dict), set(probe_dict)

    else:
        return edge_dict, node_dict, probe_dict


def edgesin(path, directed=False, trim=True):
    """
    This function takes a path (a sequence of nodes) and returns
    a list of tuples representing the edges in that path.

    The directed parameter specifies whether edge direction matters.
    If directed == False (default), then edges are returned in their canonical representation.
    If directed == True, then edges are returned with nodes appearing in the same order as in the path.

    trim==True (default) does not include self-loop edges in the result.
    To include, set trim=False.

    The returned list is empty for single-node paths.
    """
    edges = []

    t = cer if not directed else identity

    for i in range(len(path) - 1):
        if not trim or path[i] != path[i+1]:
            edges.append(t((path[i], path[i + 1])))
    return edges


def histogram_to_list(histo, repeat=False):
    if not repeat:
        return list(histo.keys())

    return_list = []
    for i, count in histo.iteritems():
        return_list += [i] * count
    return return_list


def all_edge_set(pseq, directed=False, trim=True):
    edges = set()

    if isinstance(pseq, list):
        for p in pseq:
            edges |= set(edgesin(p, directed=directed, trim=trim))

    elif isinstance(pseq, dict):
        for s in pseq:
            for d in pseq[s]:
                if not trim or s != d:
                    edges |= set(edgesin(pseq[s][d], directed=directed, trim=trim))

    else:
        raise TypeError

    return edges


def edgesinpaths(pl, histo=False, repeat=False, directed=False, trim=True):
    """
    This function takes a list of paths (sequences of nodes) and returns
    either a list of tuples (default) representing edges in those paths
    or a histogram of edge counts (set histo=True), which is a dictionary
    where keys are edges and values are their frequencies.

    If histo==False (default), then by default at most one occurrence of an edge
    will appear in the returned list, unless the parameter repeat==True, in
    which case the number of times an edge appears in the list is
    exactly its frequency in the list of paths.

    directed==False (default) assumes that the graph is undirected, and so
    a tuple and its reverse represent the same edge.  In this case, edges are returned
    in canonical representation.  Setting directed=True treats an edge as
    a separate entity than its reverse.

    trim==True (default) does not include self-loop edges.
    To include, set trim=False.
    """
    ed = {}
    for p in pl:
        for e in edgesin(p, directed=directed, trim=trim):
            ed[e] = ed.get(e, 0) + 1

    if histo: return ed
    return histogram_to_list(ed, repeat=repeat)


def edgesinpd(pd, histo=False, repeat=False, directed=False, trim=True):
    """
    This function takes a path dictionary and returns
    either a list of tuples (default) representing edges in those paths
    or a histogram of edge counts (set histo=True), which is a dictionary
    where keys are edges and values are their frequencies.

    If histo==False (default), then by default at most one occurrence of an edge
    will appear in the returned list, unless the parameter repeat==True, in
    which case the number of times an edge appears in the list is
    exactly its frequency in the list of paths.

    directed==False (default) assumes that the graph is undirected, and so
    a tuple and its reverse represent the same edge.  In this case, edges are returned
    in canonical representation.  Setting directed==True treats an edge as
    a separate entity than its reverse.

    trim==True (default) does not include self-loop edges or loop paths.
    To include, set trim=False.
    """
    ed = {}
    for s in pd:
        for d in pd[s]:
            if not trim or s != d:
                for e in edgesin(pd[s][d], directed=directed, trim=trim):
                    ed[e] = ed.get(e, 0) + 1

    if histo: return ed
    return histogram_to_list(ed, repeat=repeat)


def probesinpd(pd, trim=True):
    """
    This function takes a path dictionary and returns a set of all
    the source and destination nodes in it.

    By default, endpoints of loops are not included.
    To include, set trim=False.
    """

    probes = set()
    for src in pd:
        dsts = set(pd[src])
        if len(dsts - {src}) > 0 or not trim:
            probes |= {src}
            probes |= set(pd[src])

    return probes


def nodesinpd(pd, histo=False, repeat=False, transit=False, trim=True):
    '''
    This function returns the set of nodes
    (and, if histo==True, their counts)
    that appear in a path dictionary.

    If returning a list, the repeat parameter can be used
    to include multiple entries corresponding to the number of times
    that a node appears.

    The transit parameter can be set to avoid
    counting source and destination nodes.

    The trim parameter ignores self loops.
    '''
    nd = {}

    for s in pd:
        for d in pd[s]:
            if not trim or s != d:
                path = pd[s][d]
                if transit:
                    path = path[1:-1]
                for n in path:
                    nd[n] = nd.get(n, 0) + 1

    if histo: return nd
    return histogram_to_list(nd, repeat=repeat)


def plfrompd(pd, trim=True):
    """
    This function takes a path dictionary and returns a simple list of paths
    in that dictionary by iterating through the keys of the dictionary.

    There is a parameter to trim self loops from the dictionary.
    """
    paths = []
    for s in pd:
        for d in pd[s]:
            if not trim or s != d:
                paths.append(pd[s][d])
    return paths


def pdfrompl(pl, trim=True):
    """
    This function takes a list of paths and returns a path dictionary
    (two-level by source and destination node).

    :param pl: List of paths
    :param trim: Eliminate full loops (True by default)
    :return: path dictionary
    """
    pd = dict()
    for path in pl:
        src = path[0]
        dst = path[-1]
        if not trim or src != dst:
            pd.setdefault(src, dict())[dst] = path
    return pd


def filternodes(pseq, R, trim=True):
    """
    This function trims a collection of paths to those that start or end with nodes in R.
    There is a parameter to prevent trimming self loops (set trim==False).

    When trimmed, nodes with only loops do not appear in the path dictionary at all.
    """

    if isinstance(pseq, dict):

        pd = pseq
        pd2 = {}

        for s in pd:
            if s in R:
                for d in pd[s]:
                    if (not trim or s != d) and d in R:
                        pd2.setdefault(s, dict())[d] = pd[s][d]

        return pd2

    elif isinstance(pseq, list):

        pl2 = []
        for path in pseq:
            s = path[0]
            d = path[-1]
            if s in R and d in R and (s != d or not trim):
                pl2.append(path)

        return pl2

    else:

        raise TypeError


def removeSelfLoops(pd):
    """
    This function trims pd of all full loops

    When trimmed, nodes with only loops do not appear in the path dictionary at all.
    """

    pd2 = {}

    for s in pd:
        for d in pd[s]:
            if s != d:
                pd2.setdefault(s, dict())[d] = pd[s][d]

    return pd2


def reversepath(path):
    """
    This function reverses a path.
    """
    return path[::-1]


def distincthelper(coll, path):
    src = path[0]
    dst = path[-1]
    if coll.get(dst, dict()).get(src, None) != reversepath(path):
        coll.setdefault(src, dict())[dst] = path

def distinctpaths(paths, nolist=False):
    """
    This function takes a collection of paths and returns a copy with
    duplicates eliminated; assuming undirected edges, each undirected path
    will appear in the return value at most once.

    Supported collections are lists and dictionaries.
    """

    if isinstance(paths, list):
        paths2 = dict()
        for path in paths:
            distincthelper(paths2, path)
        return plfrompd(paths2, trim=False) if not nolist else paths2

    elif isinstance(paths, dict):
        paths2 = dict()
        for src in sorted(paths.keys()):
            for dst in paths[src]:
                distincthelper(paths2, paths[src][dst])
        return paths2

    else:
        raise TypeError


def countpaths(pseq, distinct=True, trim=True):
    """
    This function returns the number of paths in a path dictionary or list.

    The distinct parameter excludes paths that are reverses of other paths
    in the count.

    The trim parameter excludes loops.
    """

    if isinstance(pseq, list):
        if not distinct:
            count = 0
            for path in pseq:
                if path[0] != path[-1] or not trim:
                    count += 1
            return count
        else:
            pd = distinctpaths(pseq, nolist=True)
            count = 0
            for src in pd:
                count += len(pd[src])
                if trim and src in pd[src]:
                    count -= 1
            return count

    elif isinstance(pseq, dict):
        if not distinct:
            count = 0
            for src in pseq:
                count += len(pseq[src])
                if trim and src in pseq[src]:
                    count -= 1
            return count
        else:
            count = 0
            for src in pseq:
                for dst in pseq[src]:
                    if src == dst:
                        count += int(not trim)
                    elif src < dst:
                        count += 1
                    else:  # src > dst
                        count += int(not (reversepath(pseq[src][dst]) == pseq.get(dst, dict()).get(src, None)))
            return count

    else:
        raise TypeError


def get_edge_weight(edge, G=None, weight=None, default=None, none_error=True):
    result = 1 if (G is None or weight is None) else G[edge[0]][edge[1]].get(weight, default)
    if result is not None or not none_error: return result
    raise ValueError('Edge (' + repr(edge[0]) + ', ' + repr(edge[1]) + ') has no attribute ' + weight)

def get_node_weight(node, G=None, weight=None, default=None, none_error=True):
    result = 1 if (G is None or weight is None) else G.node[node].get(weight, default)
    if result is not None or not none_error: return result
    raise ValueError('Node ' + repr(node) + ' has no attribute ' + weight)


def largest_connected_component(G):
    """
    Returns the largest connected component of graph G.
    Ties are broken by choosing the component with the lexically earliest node.
    """

    key_function = lambda G: (-len(G), min(G))
    return min(nx.connected_component_subgraphs(G), key=key_function)
