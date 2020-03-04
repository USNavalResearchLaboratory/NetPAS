import random
from sys import maxint as MAXINT

from .. import util as ps

#------------------

def path_edge_dict(P):
    paths_to_edges = dict()
    # P is a probing-path dictionary;
    # this iteration is dependent on hash order but that does not affect results
    for s in P:
        for d in P[s]:
            if d != s:
                for edge in ps.edgesin(P[s][d]):
                    paths_to_edges.setdefault((s, d), set()).add(edge)
    return paths_to_edges


def compute_path_weights(current_link_weights, paths_to_edges):
    """
    This function returns a dictionary {number: set}, in which the keys are path-weight values and the values are sets of (src,dst) pairs
    corresponding to sets that have that weight.
    
    :param current_link_weights: dict {edge: weight} giving the importance weight of edges in probing paths
    :param paths_to_edges: dict {(src, dst): set of edges} giving the edges in each probing path

    :return: dict mapping weights to sets of paths
    """
    # iteration order does not affect results because we are independently analyzing each path,
    # and the returned data structure is assumed to be unordered (dict with values as sets)

    weight_paths_dict = dict()
    for path in paths_to_edges:
        weight = sum([current_link_weights[edge] for edge in paths_to_edges[path]])
        weight_paths_dict.setdefault(weight, set()).add(path)

    return weight_paths_dict


def max_weight_paths(k, weight_paths_dict, paths_to_edges, overlap=True, norev=False, rgen=None):
    """
    Returns k largest-weight paths.  The function proceeds in order through descending weight values to select paths,
    and if an entire weight class of paths is not to be included, applies a predictable order to the paths
    before checking them for inclusion (by sorting the (src,dst) pairs in the class and then applying rgen.shuffle).

    :param k: number of paths to select
    :param weight_paths_dict: a dict {weight: set} where keys are weight values, and corresponding values are sets of paths having that weight
    :param paths_to_edges: a dict {(src, dst): set} where the path with endpoints (src, dst) consists of the edges in the corresponding set
    :param overlap: True if max-weight paths selected can share links; False if selected paths should be disjoint
    :param norev: True if (dst,src) should be excluded if (src,dst) is chosen
    :param rgen: RNG object with .shuffle() method to use; random.shuffle() is used if not provided

    :return: (set, set): the set of (src,dst) pairs for paths selected, and the set of edges covered by those paths
    """
    if rgen is None: rgen = random
    
    selected_paths = set()
    probed_edges = set()

    weight_list = sorted(weight_paths_dict, reverse=True)
    weight_index = -1
    while k > 0 and weight_index < len(weight_list)-1:
        weight_index += 1
        weight_class = weight_paths_dict[weight_list[weight_index]]
        l = len(weight_class)
        if k-l >= 0 and overlap and not norev:
            # in this case we will use all paths in the weight class, so no ordering is necessary
            selected_paths.update(weight_class)
            k -= l
        else:
            # in this case we may not use all paths in the weight class, either because:
            # (1) we don't need them all (including them all will result in more than k paths); or
            # (2) we have to check for reverses and/or overlap and thus may skip over some paths in the class
            # so we first sort the paths (deterministic) and then rgen.shuffle them so the weight tie is broken randomly but reproducibly given the RNG
            paths = sorted(weight_class)
            rgen.shuffle(paths)
            if overlap and not norev:
                selected_paths.update(paths[:k])
                k = 0
            elif overlap and norev:
                # paths is an ordered list, so iteration is deterministic
                for s, d in paths:
                    if (d,s) not in selected_paths:
                        selected_paths.add((s,d))
                        k -= 1
                        if k <= 0: break
            else:
                # paths is an ordered list, so iteration is deterministic
                for path in paths:
                    if norev and (path[1],path[0]) in selected_paths: continue
                    path_edges = paths_to_edges[path]
                    if probed_edges.isdisjoint(path_edges):
                        selected_paths.add(path)
                        probed_edges |= path_edges
                        k -= 1
                        if k <= 0: break
    
    # if overlap==True, then we didn't have to update probed_edges, so do it now
    if overlap:
        for path in selected_paths:
            probed_edges |= paths_to_edges[path]

    return selected_paths, probed_edges


def getImportance(edge, G=None, importance=None):
    return 1 if (G is None or importance is None) else G[edge[0]][edge[1]].get(importance, 0)

def update_edge_weights(G, k, N, importance, current_link_weights, probed_edges):
    # this iteration depends on hash order, but each edge is treated independently
    for edge in current_link_weights:
        if edge in probed_edges:
            current_link_weights[edge] = 0
        else:
            W = getImportance(edge, G, importance)
            current_link_weights[edge] = min(W, current_link_weights[edge]+W*k/float(N-1))


#------------------

def bdrs_step(G, k, current_link_weights, paths_to_edges, importance=None, overlap=True, norev=False, rgen=None):
    """
    Runs one timestep of the BDRS strategy, which selects a number of the highest-weight paths, returns that set of paths,
    and updates link weights accordingly.

    :param G: the graph topology, a NetworkX graph object
    :param k: the number of max-weight paths to probe per timestep
    :param current_link_weights: a dict {(u,v): w} where the edge (u,v) has dynamic weight w
    :param paths_to_edges: a dict {(src, dst): set} where the path with endpoints (src, dst) consists of the edges in the corresponding set
    :param importance: the name of the attribute in G containing link importance weights (if None, use 1)
    :param overlap: True if max-weight paths selected can share links; False if selected paths should be disjoint
    :param norev: True if (dst,src) should be excluded if (src,dst) is chosen
    :param rgen: RNG object with .shuffle() method to use; random.shuffle() is used if not provided

    :return: a set of (src, dst) pairs corresponding to the endpoints of paths chosen to be probed; note that current_link_weights is also mutated
    """

    # compute current path weights
    weight_paths_dict = compute_path_weights(current_link_weights, paths_to_edges)

    # randomly choose k paths of largest weight
    N = len(paths_to_edges) # number of probeable paths
    k = min(k, N) # can't probe more paths than N

    selected_paths, probed_edges = max_weight_paths(k, weight_paths_dict, paths_to_edges, overlap, norev, rgen)
    update_edge_weights(G, k, N, importance, current_link_weights, probed_edges)

    return selected_paths


def simulate(graph, paths, *args, **kwargs):
    """
    Runs one trial of the BDRS [INFOCOM'09] probing strategy.

    :param graph: Network topology
    :param paths: Probing paths
    :param args: Captures additional arguments
    :param kwargs: The following optional keyword arguments are supported:
                -- BDRS specific --
                k: Number of max-weight paths to probe per timestep (as in paper)
                importance: Name of edge attribute in the graph for importance weights.
                            If omitted, all edges are assigned equal importance.
                overlap: Set to False to prevent paths in a single timestep to share edges (limits timestep edge load to 1)
                         By default, overlap==True, allowing edge load > 1
                norev: Set to True to exclude the path (d,s) from a timestep if (s,d) was already chosen
                       By default, norev==False, allowing both (s,d) and (d,s) to be chosen if otherwise permitted

                -- Simulation options --
                trialseed: random seed for the trial (randomly chosen if none is provided)
                stopOnceCovered: end the trial when all target edges are covered
                totalcov: target edges to cover (all probeable edges by default)
                maxsteps: maximum number of timesteps (50 by default)

    :return: a list of sets of (src,dst) pairs indicating which paths are probed at which time
    """
    G = graph
    P = paths

    all_edges = ps.all_edge_set(P)

    k = kwargs.get('k', 1)
    importance = kwargs.get('importance', None)
    overlap = kwargs.get('overlap', True)
    norev = kwargs.get('norev', False)
    trialseed = kwargs.get('trialseed', random.randint(-MAXINT-1, MAXINT))
    stopOnceCovered = kwargs.get('stopOnceCovered', False)
    if stopOnceCovered:
        totalcov = kwargs.get('totalcov', all_edges)
    else:
        totalcov = None

    maxsteps = kwargs.get('maxsteps', 50)

    # G has the graph
    # P has the set of probing paths
    # k is the number of max-weight paths to probe per timestep
    # importance has the name of the attribute in G containing link importance weights

    rgen = random.Random(trialseed)
    seq = []
    
    # create helper dictionaries
    #
    # paths_to_edges:
    # keys are probeable (src,dst) pairs, values are sets of canonical edges in those paths
    paths_to_edges = path_edge_dict(P)

    current_link_weights = dict.fromkeys(all_edges, 0) # current link weights, initially zero

    for _ in range(maxsteps):
        r = bdrs_step(G, k, current_link_weights, paths_to_edges, importance, overlap, norev, rgen)
        seq.append(r)
        
        if stopOnceCovered:
            new_edges = ps.all_edge_set([P[s][d] for s, d in r])
            totalcov -= new_edges
            if len(totalcov) == 0:
                break
                
    return seq