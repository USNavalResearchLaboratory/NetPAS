# This file modifies the gnm_random_graph, barabasi_albert_graph,
# single_source_shortest_path, and all_pairs_shortest_path
# functions, and incorporates parts of the _random_subset function,
# from NetworkX 1.10, which is distributed under the following license:
# 
# =====================Begin NetworkX 1.10 License Text==============
# Copyright (C) 2004-2012, NetworkX Developers
# Aric Hagberg <hagberg@lanl.gov>
# Dan Schult <dschult@colgate.edu>
# Pieter Swart <swart@lanl.gov>
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# 
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
# 
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
# 
#   * Neither the name of the NetworkX Developers nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
# 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# =====================End NetworkX 1.10 License Text================

import networkx as nx
import random
import sys
import copy

def jr_gnm_random_graph(n, m, seed=None, directed=False):
    """Returns a `G_{n,m}` random graph.

    In the `G_{n,m}` model, a graph is chosen uniformly at random from the set
    of all graphs with `n` nodes and `m` edges.
    
    Modifies original version of gnm_random_graph to instantiate a separate
    RNG that is used within this function.

    Parameters
    ----------
    n : int
        The number of nodes.
    m : int
        The number of edges.
    seed : int, optional
        Seed for random number generator (default=None).
    directed : bool, optional (default=False)
        If True return a directed graph

    """
    if seed is not None:
        local_rng = random.Random(seed)
    else:
        local_rng = random.Random()
    
    if directed:
        G=nx.DiGraph()
    else:
        G=nx.Graph()
    G.add_nodes_from(range(n))
    G.name="jr_gnm_random_graph(%s,%s,%s)"%(n,m,seed)

    if n==1:
        return G
    max_edges=n*(n-1)
    if not directed:
        max_edges/=2.0
    if m>=max_edges:
        return nx.complete_graph(n,create_using=G)

    nlist=G.nodes()
    edge_count=0
    while edge_count < m:
        # generate random edge,u,v
        u = local_rng.choice(nlist)
        v = local_rng.choice(nlist)
        if u==v or G.has_edge(u,v):
            continue
        else:
            G.add_edge(u,v)
            edge_count=edge_count+1
    return G

    
def jr_barabasi_albert_graph(n, m, seed=None):
    """Returns a random graph according to the Bar\'abasi--Albert preferential
    attachment model.

    A graph of ``n`` nodes is grown by attaching new nodes each with ``m``
    edges that are preferentially attached to existing nodes with high degree.
    
    Modifies original version of barabasi_albert_graph to instantiate a separate
    RNG that is used within this function.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : int, optional
        Seed for random number generator (default=None).

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If ``m`` does not satisfy ``1 <= m < n``.

    References
    ----------
    .. [1] A. L. Barab\'asi and R. Albert "Emergence of scaling in
       random networks", Science 286, pp 509-512, 1999.
    """
    if seed is not None:
        local_rng = random.Random(seed)
    else:
        local_rng = random.Random()
    

    if m < 1 or  m >=n:
        raise nx.NetworkXError("Barab\'asi--Albert network must have m >= 1"
                               " and m < n, m = %d, n = %d" % (m, n))

    # Add m initial nodes (m0 in barabasi-speak)
    G=nx.empty_graph(m)
    G.name="jr_barabasi_albert_graph(%s,%s,%s)"%(n,m,seed)
    # Target nodes for new edges
    targets=list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes=[]
    # Start adding the other n-m nodes. The first node is m.
    source=m
    while source<n:
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source]*m,targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source]*m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachement)
        #
        # This modifies the code from the NetworkX _random_subset function
        # to be inline and use our local RNG
        #
        targets=set()
        while len(targets)<m:
            x=local_rng.choice(repeated_nodes)
            targets.add(x)
        #
        # End of _random_subset replacement
        #
        source += 1
    return G


def jr_fractional_barabasi_albert_graph(n, m, seed=None):
    """Returns a random graph according to a fractional variant of 
    the Bar\'abasi--Albert preferential attachment model.

    A graph of ``n`` nodes is grown by attaching new nodes that are
    preferentially attached to existing nodes with high degree.  For a
    real-valued attachment parameter ``m`` > 1, decompose m into its integer
    part k and its fractional part p.  Each new node is connected to k+1
    existing nodes with probability p and to k existing nodes with probability
    1-p.
    
    Also instantiates a separate RNG that is used within this function.

    Parameters
    ----------
    n : int
        Number of nodes
    m : float
        Number of edges to attach from a new node to existing nodes
    seed : int, optional
        Seed for random number generator (default=None).

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If ``m`` does not satisfy ``1 <= m < n``.

    References
    ----------
    .. [1] A. L. Barab\'asi and R. Albert "Emergence of scaling in
       random networks", Science 286, pp 509-512, 1999.
    """
    if seed is not None:
        local_rng = random.Random(seed)
    else:
        local_rng = random.Random()
    

    if m < 1 or  m >n-1:
        raise nx.NetworkXError("Fractional Barab\'asi--Albert network must have m >= 1 and m <= n-1, m = %d, n = %d" % (m, n))

    k = int(m)
    p = m - k
    
    r = local_rng.random()
    if r < p:
        num_new = k+1
    else:
        num_new = k

    # Add m initial nodes (m0 in barabasi-speak)
    G=nx.empty_graph(num_new)
    G.name="jr_fractional_barabasi_albert_graph(%s,%s,%s)"%(n,m,seed)
    # Target nodes for new edges
    targets=list(range(num_new))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes=[]
    # Start adding the other n-m nodes. The first node is m.
    source=num_new
    while source<n:
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source]*num_new,targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source]*num_new)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachement)
        #
        # This modifies the code from the NetworkX _random_subset function
        # to be inline and use our local RNG
        #
        targets=set()
        r = local_rng.random()
        if r < p:
            num_new = k+1
        else:
            num_new = k
        while len(targets)<num_new:
            x=local_rng.choice(repeated_nodes)
            targets.add(x)
        #
        # End of _random_subset replacement
        #
        source += 1

    # Should never wind up with disconnected G, but flag it if it happens
    if not nx.is_connected(G):
        raise RuntimeError('G is unexpectedly disconnected!')
    
    return G


def jr_single_source_shortest_path(G,source,seed=None,cutoff=None):
    """Compute shortest path between source
    and all other nodes reachable from source.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    lengths : dictionary
        Dictionary, keyed by target, of shortest paths.

    Examples
    --------
    >>> G=nx.path_graph(5)
    >>> path=nx.single_source_shortest_path(G,0)
    >>> path[4]
    [0, 1, 2, 3, 4]

    Notes
    -----
    The shortest path is not necessarily unique. So there can be multiple
    paths between the source and each target node, all of which have the
    same 'shortest' length. For each target node, this function returns
    only one of those paths.

    See Also
    --------
    shortest_path
    """

    if seed is not None:
        local_rng = random.Random(seed)
    else:
        local_rng = random.Random()
    

    level=0                  # the current level
    nextlevel={source:1}       # list of nodes to check at next level
    paths={source:[source]}  # paths dictionary  (paths to key from source)
    if cutoff==0:
        return paths
    while nextlevel:
        thislevel=nextlevel
        nextlevel={}
        shuff_level_keys =  sorted(thislevel.keys())
        local_rng.shuffle(shuff_level_keys,local_rng.random)
        for v in shuff_level_keys:
            shuff_nbrs = sorted(G[v])
            local_rng.shuffle(shuff_nbrs,local_rng.random)
            for w in shuff_nbrs:
                if w not in paths:
                    paths[w]=paths[v]+[w]
                    nextlevel[w]=1
        level=level+1
        if (cutoff is not None and cutoff <= level):  break
    return paths


def jr_all_pairs_shortest_path(G, seed=None, cutoff=None):
    """Compute shortest paths between all nodes.

    Parameters
    ----------
    G : NetworkX graph

    cutoff : integer, optional
        Depth at which to stop the search. Only paths of length at most
        ``cutoff`` are returned.

    Returns
    -------
    lengths : dictionary
        Dictionary, keyed by source and target, of shortest paths.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> path = nx.all_pairs_shortest_path(G)
    >>> print(path[0][4])
    [0, 1, 2, 3, 4]

    See Also
    --------
    floyd_warshall()

    """
    
    if seed is not None:
        local_rng = random.Random(seed)
    else:
        local_rng = random.Random()

    node_list = sorted(G.nodes())
    seed_list = [local_rng.randint(-sys.maxint - 1, sys.maxint) for i in range(len(node_list))]
    
    path_dict = dict()
    for i in range(len(node_list)):
        n = node_list[i]
        tmp_paths = jr_single_source_shortest_path(G, node_list[i], seed=seed_list[i], cutoff=cutoff)
        path_dict[n] = copy.deepcopy(tmp_paths)
    
    return path_dict
    
