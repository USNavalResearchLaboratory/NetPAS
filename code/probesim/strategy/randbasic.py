import sys
import random
from .. import util as ps

def path_iteration_list(pd, rgen):
    """
    This function creates a predictable iteration order on probing paths
    so that the mapping between random numbers and paths during simulation is reproducible.

    :param pd: Probing-path dictionary
    :param rgen: Random-number generator with shuffle function

    :return: a list of (src, dst) tuples from the path dictionary
    """

    src_dst_list = []
    for s in pd:
        for d in pd[s]:
            if d != s:
                src_dst_list.append((s, d))

    src_dst_list.sort()
    rgen.shuffle(src_dst_list)
    return src_dst_list


def simulate(graph, paths, p, *args, **kwargs):
    """
    Runs randomized probing strategy.

    :param graph: Network topology
    :param paths: Probing-path dictionary
    :param p: Probing probability
    :param args: Captures other arguments
    :param kwargs: The following optional arguments are supported:
                -- Strategy specific --
                maxdelay: for version with history, maximum delay for any edge to be probed

                -- Simulation options --
                trialseed: random seed for the trial (randomly chosen if none is provided)
                stopOnceCovered: end the trial when all target edges are covered
                totalcov: target edges to cover (all probeable edges by default)
                maxsteps: maximum number of timesteps (50 by default)

    :return: a list of sets of (src,dst) pairs indicating which paths are probed at which time
    """
    trialseed = kwargs.get('trialseed', None)
    if trialseed is None:
        trialseed = random.randint(-sys.maxint-1, sys.maxint)
    stopOnceCovered = kwargs.get('stopOnceCovered', False)
    if stopOnceCovered:
        totalcov = kwargs.get('totalcov', ps.all_edge_set(paths))
    else:
        totalcov = None

    maxsteps = kwargs.get('maxsteps', 50)
    maxdelay = kwargs.get('maxdelay', 0)
    # history dictionary; iteration to build depends on hash order but doesn't affect results
    if maxdelay > 0:
        history = {src: dict.fromkeys(paths[src], -1) for src in paths}

    rgen = random.Random(trialseed)
    src_dst_list = path_iteration_list(paths, rgen)
    seq = []

    for step in range(maxsteps):
        r = set(filter(lambda _: rgen.random() < p, src_dst_list))

        # loops in this section depend on hash order but don't affect results,
        # because paths are included based on their independent probing history,
        # and not on the order in which histories are examined.
        if maxdelay > 0:
            for src in history:
                for dst in history[src]:
                    if (src, dst) in r:
                        history[src][dst] = step
                    else:
                        if step - history[src][dst] >= maxdelay:
                            r.add((src,dst))
                            history[src][dst] = step

        seq.append(r)

        if stopOnceCovered:
            new_edges = ps.all_edge_set([paths[s][d] for s, d in r])
            totalcov -= new_edges
            if len(totalcov) == 0:
                break

    return seq