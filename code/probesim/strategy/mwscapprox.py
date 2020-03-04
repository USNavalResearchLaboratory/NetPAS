import random
import sys
from .. import util as ps
from ..heapqup import heapqup


# ---- helper functions ----

def number_path_set(path_list):
    '''
    This function takes a path list and numbers it (that is, add
    a unique integer ID to each of the path in the set).

    This function returns a hash map (dictionary).
    '''

    id_path_dict = dict()

    i = 1
    for path in path_list:
        id_path_dict[i] = path
        i += 1

    return id_path_dict


def create_edge_dictionaries(id_path_dict):
    id_edge_set_dict = dict()
    all_edge_dict = dict()

    for pid in id_path_dict:
        for edge in ps.edgesin(id_path_dict[pid]):
            id_edge_set_dict.setdefault(pid, set()).add(edge)
            all_edge_dict.setdefault(edge, set()).add(pid)

    return id_edge_set_dict, all_edge_dict


# ---- main simulation function ----

def simulate(graph, paths, *args, **kwargs):
    """
    Runs the min-wt-set-cover probing strategy.

    :param graph: Network topology
    :param paths: Possible probing paths
    :param args: Captures extra arguments
    :param kwargs: The following optional keyword arguments are supported:
                -- Strategy specific --
                alpha: Probe/load tradeoff parameter [ISCC'17]
                       A setting of alpha=0 [default] will minimize probe (path) count
                       A setting of alpha=1 will minimize average edge load
                k: Number of paths to probe per timestep in expectation
                   If specified, each path will be probed with probability min(1, float(k)/len(cover))
                   The default value is None, which will probe the entire cover for a single timestep (ignoring maxsteps), returning [cover]

                -- Simulation options --
                trialseed: random seed for the trial (randomly chosen if none is provided)
                stopOnceCovered: end the trial when all target edges are covered
                totalcov: target edges to cover (all probeable edges by default)
                maxsteps: maximum number of timesteps (50 by default)

    :return: a list of lists of (src,dst) pairs indicating which paths are probed at which time
             if k is None, the list has length 1 and the set at index 0 contains paths in the computed cover
    """
    alpha = float(kwargs.get('alpha', 0))
    if alpha < 0 or alpha > 1:
        raise ValueError

#     trialseed = kwargs.get('trialseed', random.randint(-sys.maxint - 1, sys.maxint))
    trialseed = kwargs.get('trialseed', None)
    if trialseed is None:
        trialseed = random.randint(-sys.maxint - 1, sys.maxint)

    all_edges = ps.all_edge_set(paths)

    stopOnceCovered = kwargs.get('stopOnceCovered', False)
    if stopOnceCovered:
        totalcov = kwargs.get('totalcov', all_edges)
    else:
        totalcov = None

    maxsteps = kwargs.get('maxsteps', 50)

    # FIRST, establish set cover using min-wt set-cover approximation algorithm
    cover = list()
    rgen = random.Random(trialseed)

    path_list = ps.plfrompd(paths)
    # assigning unique integer ID to each path -- keys are path IDs, values are lists of nodes (paths) -OK
    id_path_dict = number_path_set(path_list)

    id_edge_set_dict, all_edge_dict = create_edge_dictionaries(id_path_dict)  # OK
    score = lambda p: len(id_edge_set_dict[p]) / float(alpha * (len(id_path_dict[p]) - 1) + 1)
    # - the heap is created from a dictionary, assumed to be unordered
    # - the heap itself contains scores, which are totally & predictably order-able items
    # - extraction of a path of max score is done using a reproducible tiebreaker using the seeded RNG
    #      (use of RNG object with lambda function parameter was tested)
    max_PQ = heapqup({path_id: score(path_id) for path_id in id_path_dict}, reverse=True, tiebreak=lambda pathset: rgen.choice(sorted(pathset)))

    all_path_edge_set = ps.all_edge_set(paths)
    # while there are still edges to cover
    while len(all_path_edge_set) > 0:

        # get the path that has the most uncovered edges (uses reproducible tie-break)
        curr_path_id = max_PQ.poll()

        # this set will be used to update
        # the priorities of the paths that were
        # affected by the removal of the path selected above
        affected_path_set = set()

        # add path to probing sequence
        curr_path = id_path_dict[curr_path_id]
        cover.append((curr_path[0], curr_path[-1]))

        # iterate through all edges in this path
        # (iteration order doesn't matter here)
        for edge in id_edge_set_dict[curr_path_id].copy():

            # if the edge is in the set of uncovered
            # edges, remove it
            if edge in all_path_edge_set:
                all_path_edge_set.remove(edge)

            # get the path id set that corresponds to this edge
            path_id_set_of_edge = all_edge_dict[edge]

            # for each path in all the paths that contain this
            # edge, delete the edge from the path's uncovered edge set
            # and note this happened in an affected-path set
            for pid in path_id_set_of_edge:
                if edge in id_edge_set_dict[pid]:
                    id_edge_set_dict[pid].remove(edge)
                    affected_path_set.add(pid)

        # update priorities for paths affected
        for path_id in affected_path_set:
            max_PQ[path_id] = score(path_id)

    # NEXT, run the simulation to generate a probing sequence
    # OR, simply return the cover as a single timestep if k is None
    k = kwargs.get('k', None)

    if k is None:
        return [list(cover)]
    
    seq = []
    p = float(k)/len(cover)
    cover.sort()  # sort cover so that a predictable order of paths is used in random path selection

    for _ in range(maxsteps):
        # iteration here should proceed in order through the list cover
        current = list(filter(lambda _: rgen.random() < p, cover))
        seq.append(current)

        if stopOnceCovered:
            new_edges = ps.all_edge_set([paths[s][d] for s, d in current])
            totalcov -= new_edges
            if len(totalcov) == 0:
                break
                
    return seq