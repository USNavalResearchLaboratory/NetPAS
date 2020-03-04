import networkx as nx

from .. import util as ps


def add_edge_weights_from_paths(G, P, edge_attr_name='weight', use_default=True, default=0):
    """
    Assigns weights to edges in G, such that the weight of edge e is the number of paths in P in which e appears.
    Optionally control how weights are set for edges that do not appear in P.  By default, those edges are assigned a weight of zero.

    :param G: graph topology
    :param P: path dictionary on `G`
    :param edge_attr_name: string for name of edge attribute to which weights are assigned
    :param use_default: True (default) if edges that do not appear in `P` are to be assigned a weight; False if no assignment should occur for those edges
    :param default: Value to assign to edges that do not appear in `P`; 0 by default
    :return: None (mutates `G`)
    """

    if use_default:
        nx.set_edge_attributes(G, name=edge_attr_name, values=default)
    nx.set_edge_attributes(G, name=edge_attr_name, values=ps.edgesinpd(P, histo=True, directed=nx.is_directed(G)))


def add_node_weights_from_edge_sum(G, node_attr_name='weight', edge_attr_name='weight', use_default=True, default=0):
    """
    Assigns weights to nodes in G equal to the sum of weight values of incident edges.
    Optionally control how weights are set for nodes that are not incident to weighted edges; by default those nodes are assigned a weight of zero.

    :param G: graph topology
    :param node_attr_name: string for name of node attribute to which weights are assigned
    :param edge_attr_name: string for name of edge attribute from which weights are calculated
    :param use_default: True (default) if nodes not incident to weighted edges are assigned a weight; False if no assignment should occur for those nodes
    :param default: Value to assign to nodes not incident to weighted edges; 0 by default
    :return: None (mutates G)
    """

    for n in G.nodes_iter():
        adjacent_weights = [it[2] for it in G.edges_iter(n, data=edge_attr_name) if it[2] is not None]
        if len(adjacent_weights) > 0:
            G.node[n][node_attr_name] = sum(adjacent_weights)
        elif use_default:
            G.node[n][node_attr_name] = default


def add_node_weights_from_edge_mean(G, node_attr_name='weight', edge_attr_name='weight', use_default=True, default=0):
    """
    Assigns weights to nodes in G equal to the mean of weight values of incident edges.
    Optionally control how weights are set for nodes that are not incident to weighted edges; by default those nodes are assigned a weight of zero.

    :param G: graph topology
    :param node_attr_name: string for name of node attribute to which weights are assigned
    :param edge_attr_name: string for name of edge attribute from which weights are calculated
    :param use_default: True (default) if nodes not incident to weighted edges are assigned a weight; False if no assignment should occur for those nodes
    :param default: Value to assign to nodes not incident to weighted edges; 0 by default
    :return: None (mutates G)
    """

    for n in G.nodes_iter():
        adjacent_weights = [it[2] for it in G.edges_iter(n, data=edge_attr_name) if it[2] is not None]
        if len(adjacent_weights) > 0:
            G.node[n][node_attr_name] = float(sum(adjacent_weights))/len(adjacent_weights)
        elif use_default:
            G.node[n][node_attr_name] = default


def add_node_weights_from_edge_max(G, node_attr_name='weight', edge_attr_name='weight', use_default=True, default=0):
    """
    Assigns weights to nodes in G equal to the maximum of weight values of incident edges.
    Optionally control how weights are set for nodes that are not incident to weighted edges; by default those nodes are assigned a weight of zero.

    :param G: graph topology
    :param node_attr_name: string for name of node attribute to which weights are assigned
    :param edge_attr_name: string for name of edge attribute from which weights are calculated
    :param use_default: True (default) if nodes not incident to weighted edges are assigned a weight; False if no assignment should occur for those nodes
    :param default: Value to assign to nodes not incident to weighted edges; 0 by default

    :return: None (mutates G)
    """

    for n in G.nodes_iter():
        adjacent_weights = [it[2] for it in G.edges_iter(n, data=edge_attr_name) if it[2] is not None]
        if len(adjacent_weights) > 0:
            G.node[n][node_attr_name] = max(adjacent_weights)
        elif use_default:
            G.node[n][node_attr_name] = default


def add_reciprocal_weights(G, src_attr_name='weight', dst_attr_name='inv_weight', components='both', use_default=True, default=0):
    """
    Given a graph G with components that have attribute values named `src_attr_name`,
    add or replace for each component an attribute named `dest_attr_name` with a value
    that is the reciprocal of the value for `src_attr_name`.

    A component with value zero for `src_attr_name` will be assigned a value of zero for `dst_attr_name`.
    A component with no value or the value None for `src_attr_name` will be assigned a value of `default` for `dst_attr_name` if `use_default` is True and will not be assigned a value at all if `use_default` is False.

    :param G: graph toplogy, an NetworkX graph object
    :param src_attr_name: name of the attribute from which reciprocal values should be computed
    :param dst_attr_name: name of the attribute to which reciprocal values should be assigned
    :param components: for which components of the graph should values be computed? valid argument values are:
        - 'both': nodes and edges
        - 'nodes': nodes only
        - 'edges': edges only
    :param use_default: If True (default), then components with no value or None for `src_attr_name` will be assigned the provided `default` value.  If False, no value will be assigned for such components.
    :param default: Default value to assign to components with no value or None for `src_attr_name` (zero by default).
    :param default: 
    :return: None (mutates G)
    """

    if components=='edges' or components=='both':
        for u,v,w in G.edges_iter(data=src_attr_name):
            if w is None and use_default:
                G[u][v][dst_attr_name] = default
            elif w == 0:
                G[u][v][dst_attr_name] = 0
            elif w is not None:
                G[u][v][dst_attr_name] = 1.0/w
    
    if components=='nodes' or components=='both':
        for _, d in G.nodes_iter(data=True):
            w = d.get(src_attr_name)
            if w is None and use_default:
                d[dst_attr_name] = default
            elif w == 0:
                d[dst_attr_name] = 0
            elif w is not None:
                d[dst_attr_name] = 1.0/w
