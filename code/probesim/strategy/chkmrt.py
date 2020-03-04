import random
import bisect

from sys import maxint as MAXINT

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

import cvxpy as cp

from .. import util as ps

# ---- pulling weights from graph ----
def edgeset_per_test(testset, directed=False):
    """
    Computes, given a list of probing tests (each a list of probing paths), the edges covered by each test.
    The order of the list returned is the same as the order of tests in the input list; each component is a Python set (and thus unordered).

    :param testset: a list of sets of probing paths
    :param directed: False (default) if (u,v) and (v,u) represent the same edge; True if edges are directed

    :return: a list of sets of edges covered by each test, in the same order as the input testset
    """
    return [set(ps.edgesinpaths(pl, directed=directed)) for pl in testset]

def combine_edgesets(edgesets):
    """
    Takes a collection of sets of edges and returns their union.
    The return type is a Python set, and thus unordered.

    :param edgesets: collection of sets of edges

    :return: a set of edges covered by all sets given
    """
    return set().union(*edgesets)

def priorities_from_graph(graph, testset, importance):
    """
    Returns a mapping from probeable edges to their priorities.
    The return type is a Python dict, and thus unordered.

    :param graph: The graph topology (a NetworkX graph object)
    :param testset: A list of probing tests, each a list of probing paths
    :param importance: Name of edge attribute to use for priorities

    :return: dict {edge: priority} mapping edges covered by the probing tests to their priorities
    """
    probeable_edges = combine_edgesets(edgeset_per_test(testset, graph.is_directed()))
    return {(u,v): graph[u][v][importance] for u, v in probeable_edges}

def priorities_from_value(graph, testset, value):
    """
    Returns a mapping from probeable edges to priorities, where each edge is assigned to the same provided value.
    The return type is a Python dict, and thus unordered.

    :param graph: The graph topology (a NetworkX graph object)
    :param testset: A list of probing tests, each a list of probing paths
    :param value: Priority to assign to each probeable edge

    :return: dict {edge: value} mapping edges covered by the probing tests to value
    """
    return dict.fromkeys(combine_edgesets(edgeset_per_test(testset, graph.is_directed())), value)

# ---- functions to generate sets of tests ----

def testset_single(paths, rgen):
    """
    Returns a test set where each test corresponds to a single probing path.
    This function is reproducible: iteration proceeds through the probing-path dictionary
    in sorted order, and a random shuffle is applied before returning.
    """

    testset = list()
    for src in sorted(paths):
        for dst in sorted(paths[src]):
            if src != dst:
                testset.append( [ paths[src][dst] ] )
    
    rgen.shuffle(testset)
    return testset

test_generators = {'singlepath': testset_single}

def generate_tests(paths, mode='singlepath', seed=None):
    """
    Return a list of probing tests given a probing-path dictionary.
    Each test is a list of paths to probe as part of the test.
    Supported modes have been tested for reproducibility given a seed.

    :param paths: Path dictionary
    :param mode: How to organize tests.
                 Currently supported modes:
                 'singlepath' (default): all probing paths are included and each test consists of one path
    :param seed: optional seed to random.Random object used in test generation
    """
    test_generator = test_generators[mode]
    rgen = random.Random(seed)
    return test_generator(paths, rgen)
    

# ---- goal-oriented optimization and priority-scaling functions ----

def uniform(graph, testset, priorities):
    """
    This function produces a uniform probability distribution for the testset.
    The function does not use the graph nor priorities parameters and
    is deterministic based on the length of the testset.

    :param graph: unused
    :param testset: list of probing tests
    :param priorities: unused

    :return: list of length m=len(testset), where each probability is 1.0/m
    """
    m = len(testset)
    return [1.0/m] * m


def max_cvxpy(graph, testset, priorities, debug=None):
    """
    Generates and solves the Cohen et al. MAX objective linear program
    for the provided list of tests and dictionary of edge priorities.
    This function uses the CVXPY solver.

    :param graph: topology, a NetworkX graph object
    :param testset: a list of probing tests, each a list of probing paths
    :param priorities: a dictionary mapping edges to their priorities.
                       We assume that priorities are scaled such that max(priorities.values()) == 1
                       and that no edge covered by the testset has a priority of zero.

    :return: list q of length m=len(testset), where, for each timestep, q[i] is the probability that test[i] should be probed
    """

    n = len(priorities)     # number of probeable edges
    m = len(testset)        # number of probing tests

    # produce a dictionary tests_for_edge that maps edges to a list of indices corresponding to tests in the testset list that contain that edge
    # as a dictionary, keys are unordered and iteration through it needs to be made predictable (later in code)
    # the test indices are iterated in ascending order, and so each edge's list is ordered
    edges_in_tests = edgeset_per_test(testset, graph.is_directed())
    tests_for_edge = dict()
    for i in range(m):
        for edge in edges_in_tests[i]:
            tests_for_edge.setdefault(edge, list()).append(i)

    z = cp.Variable()
    Q = cp.Variable(m)

    number_of_zero_priority_edges = 0
    edge_constraints = []
    for e in sorted(priorities):
        if priorities[e] == 0:
            # zero-priority edges should contribute no constraints
            number_of_zero_priority_edges += 1
        else:
            this_constraint = ( z <= (1.0/priorities[e]) * sum([Q[i] for i in tests_for_edge[e]]) )
            edge_constraints.append(this_constraint)

    assert(n == len(edge_constraints) + number_of_zero_priority_edges)
    constraints = [Q >= 0, cp.sum(Q) == 1] + edge_constraints
    
    objective = cp.Maximize(z)

    prob = cp.Problem(objective, constraints)
    prob.solve()
    if debug is not None:
        debug['testset'] = testset
        debug['tests_for_edge'] = tests_for_edge
        debug['priorities'] = priorities
        debug['constraints'] = constraints
        debug['objective'] = objective
        debug['Q'] = Q
        debug['z'] = z
    return list(Q.value)


def maxlp(graph, testset, priorities, debug=None):
    """
    Generates and solves the Cohen et al. MAX objective linear program
    for the provided list of tests and dictionary of edge priorities.
    This function uses the CVXOPT solver.

    :param graph: topology, a NetworkX graph object
    :param testset: a list of probing tests, each a list of probing paths
    :param priorities: a dictionary mapping edges to their priorities.
                       We assume that priorities are scaled such that max(priorities.values()) == 1
                       and that no edge covered by the testset has a priority of zero.

    :return: list q of length m=len(testset), where, for each timestep, q[i] is the probability that test[i] should be probed
    """
    edges_in_tests = edgeset_per_test(testset, graph.is_directed())

    n = len(priorities)     # number of probeable edges
    m = len(testset)        # number of probing tests

    c = matrix([0]*m + [-1], tc='d')
    G = [list() for _ in range(m+1)]

    # first constrain sum of probabilities to 1
    for i in range(m):
        G[i].extend([1, -1])
    G[-1].extend([0, 0])

    # next constrain all probabilities to be nonnegative
    for i in range(m):
        for col in range(len(G)):
            G[col].append(-1 if col==i else 0)
    
    # finally, put in constraints to optimize test-choice based on MAX objective
    number_of_edge_constraints = 0
    number_of_zero_priority_edges = 0
    for e in sorted(priorities):
        if priorities[e] == 0:
            # zero-priority edges should contribute no constraints
            number_of_zero_priority_edges += 1
        else:
            number_of_edge_constraints += 1

            for i in range(m):
                if e in edges_in_tests[i]:
                    G[i].append(-1.0/priorities[e])
                else:
                    G[i].append(0)
            G[-1].append(1)

    G = matrix(G, tc='d')

    assert(n == number_of_edge_constraints + number_of_zero_priority_edges)
    h = matrix([1, -1] + [0]*(m + number_of_edge_constraints), tc='d')

    sol = solvers.lp(c, G, h)
    if debug is not None:
        debug['testset'] = testset
        debug['edges_in_tests'] = edges_in_tests
        debug['priorities'] = priorities
        debug['c'] = c
        debug['G'] = G
        debug['h'] = h
        debug['sol'] = sol
    return list(sol['x'][:-1])

def sumcvxp(graph, testset, priorities, debug=None):
    """
    Generates and solves the Cohen et al. SUM objective convex program
    for the provided list of tests and dictionary of edge priorities.

    :param graph: topology, a NetworkX graph object
    :param testset: a list of probing tests, each a list of probing paths
    :param priorities: a dictionary mapping edges to their priorities.
                       We assume that priorities are scaled such that sum(priorities.values()) == 1.

    :return: list q of length m=len(testset), where, for each timestep, q[i] is the probability that test[i] should be probed
    """
    m = len(testset)

    # produce a dictionary tests_for_edge that maps edges to a list of indices corresponding to tests in the testset list that contain that edge
    # as a dictionary, keys are unordered and iteration through it needs to be made predictable (later in code)
    # the test indices are iterated in ascending order, and so each edge's list is ordered
    edges_in_tests = edgeset_per_test(testset, graph.is_directed())
    tests_for_edge = dict()
    for i in range(m):
        for edge in edges_in_tests[i]:
            tests_for_edge.setdefault(edge, list()).append(i)

    Q = cp.Variable(m)
    constraints = [Q >= 0, cp.sum(Q) == 1]
    # below, the inline interation through keys of tests_for_edge is done through a sorted list of those keys so that the constraint is formulated in a deterministic way
    objective = cp.Minimize(cp.sum([priorities[e]*cp.inv_pos(cp.sum([Q[i] for i in tests_for_edge[e]])) for e in sorted(tests_for_edge)]))

    prob = cp.Problem(objective, constraints)
    prob.solve()
    if debug is not None:
        debug['constraints'] = constraints
        debug['objective'] = objective
        debug['Q'] = Q
    return list(Q.value)

distribution_generators = {'max': maxlp,
                           'max-cvxpy': max_cvxpy,
                           'sum': sumcvxp,
                           'uniform': uniform}



def unused(graph, testset, importance=None):
    """
    Produces a priority mapping containing None values, intended to raise type errors if priorities are used.
    The mapping is a dict, and thus unordered, and the function is deterministic.

    :param graph: topology, a NetworkX graph object
    :param testset: a list of probing tests
    :param importance: unused

    :return: dict {edge: None for all edges covered by the testset}
    """
    return priorities_from_value(graph, testset, None)

def maxone(graph, testset, importance=None):
    """
    Produces a priority mapping for testset edges such that the maximum priority value is 1.
    The mapping is a dict, and thus unordered, and the function is deterministic.

    :param graph: topology, a NetworkX graph object
    :param testset: a list of probing tests
    :param importance: the name of the edge attribute in the graph containing importance values to scale for priorities.
                       by default None, which assigns a priority of 1.0 to every edge covered by the testset

    :return: dict {edge: priority for all edges covered by the testset} such that max(dict.values())==1
    """
    if importance is None:
        return priorities_from_value(graph, testset, 1.0)
    else:
        priorities = priorities_from_graph(graph, testset, importance)
        max_importance = float(max(priorities.values()))
        priorities = {edge: p/max_importance for edge, p in priorities.iteritems()}
        return priorities

def sumone(graph, testset, importance=None):
    """
    Produces a priority mapping for testset edges such that the all priority values sum to 1.
    The mapping is a dict, and thus unordered, and the function is deterministic.

    :param graph: topology, a NetworkX graph object
    :param testset: a list of probing tests
    :param importance: the name of the edge attribute in the graph containing importance values to scale for priorities.
                       by default None, which assigns a priority of 1.0/<number of edges covered> to every edge covered by the testset

    :return: dict {edge: priority for all edges covered by the testset} such that sum(dict.values())==1
    """
    if importance is None:
        probeable_edges = combine_edgesets(edgeset_per_test(testset, graph.is_directed()))
        m = len(probeable_edges)
        return dict.fromkeys(probeable_edges, 1.0/m)
    else:
        priorities = priorities_from_graph(graph, testset, importance)
        sum_importance = float(sum(priorities.values()))
        priorities = {edge: p/sum_importance for edge, p in priorities.iteritems()}
        return priorities

priority_scalers = {'max': maxone,
                    'max-cvxpy': maxone,
                    'sum': sumone,
                    'uniform': unused}

# ---- per-step simulation functions ----

# deterministic
def accum(q):
    total = 0.0
    for x in q:
        total += x
        yield total

def cdf(q):
    return list(accum(q))

def chkmrt_step(testset, cum_q, rgen):
    r = rgen.random()
    i = bisect.bisect_left(cum_q, r)
    if i != len(cum_q):
        test = testset[i]
        return {(path[0], path[-1]) for path in test}
    else:
        raise ValueError

def chkmrt_distributed(testset, scaled_q, rgen):
    paths = set()
    for i in range(len(testset)):
        if rgen.random() < scaled_q[i]:
            paths |= {(path[0], path[-1]) for path in testset[i]}
    return paths

# ---- main simulation functions ----

def compute_distribution(graph, testset, importance=None, goal='max', debug=None):
    """
    Computes the probability distribution using the Cohen et al. convex programming strategy.
    This function is deterministic (modulo external libraries).

    :param graph: Network topology
    :param testset: List of probing tests
    :param importance: name of graph edge attribute to use for importance weights
                       default: None, which assigns equal weight to all edges
    :param goal: specifies which optimization program to run
                 currently supported options:
                    'max': LP (paper #7)
                    'sum': CVXP (paper #6)
                    default: 'max'

    :return: a probability distribution over the testset
             this is a list q of the same length as testset, such that:
                 q[i] is the probability that testset[i] should be probed;
                 sum_i q[i] = 1
    """
    

    priority_scaler = priority_scalers[goal]
    priorities = priority_scaler(graph, testset, importance)

    distribution_generator = distribution_generators[goal]
    q = distribution_generator(graph, testset, priorities, debug)

    return q


def chkmrt_trial(testset, q, trialseed=None, stopOnceCovered=False, paths=None, totalcov=None, maxsteps=100, k=None):
    """
    Runs a randomized trial of CHKMRT probing, given a set of tests and a probability distribution over those tests.

    :param testset: A list of tests, each of which is a list of paths to be probed for that test
    :param q: A probability distribution over the tests
    :param trialseed: random seed for the trial (randomly chosen if none is provided)
    :param stopOnceCovered: end the trial when all target edges are covered (False by default)
    :param paths: the probing-path dictionary for the network, required only if stopOnceCovered is True
    :param totalcov: target edges to cover (all probeable edges by default)
    :param maxsteps: maximum number of timesteps (50 by default)
    :param k: expected number of tests to probe per timestep (None by default)
              if specified, each test i is independently probed per timestep with probability min(1,k*q[i])
              if None, one test is chosen per timestep from the distribution q (process described in the original paper)

    :return: a list of sets of (src,dst) pairs indicating which paths are probed at which time
    """

    if trialseed is None: trialseed = random.randint(-MAXINT-1, MAXINT)
    if stopOnceCovered:
        if paths is None: raise ValueError('path dictionary must be provided when stopOnceCovered==True')
        if totalcov is None: totalcov = ps.all_edge_set(paths)
    else:
        totalcov = None

    rgen = random.Random(trialseed)
    seq = []

    if k is None:
        scaled_q = cdf(q)
        testing_method = chkmrt_step
    else:
        scaled_q = [float(k)*prob for prob in q]
        testing_method = chkmrt_distributed

    for _ in range(maxsteps):
        probes = testing_method(testset, scaled_q, rgen)
        seq.append(probes)
        
        if stopOnceCovered:
            new_edges = ps.all_edge_set([paths[s][d] for s, d in probes])
            totalcov -= new_edges
            if len(totalcov) == 0:
                break
                
    return seq


def simulate(graph, paths, *args, **kwargs):
    """
    Runs the CHKMRT convex programming probing strategy.

    :param graph: Network topology
    :param paths: Possible probing paths
    :param args: Captures extra arguments
    :param kwargs: The following optional keyword arguments are supported:
                -- Strategy specific --
                importance: name of graph edge attribute to use for importance weights
                            default: None, which assigns equal weight to all edges

                goal: specifies which optimization program to run
                      currently supported options:
                      'max': LP (paper #7)
                      'sum': CVXP (paper #6)
                      default: 'max'

                tests: how to generate the set of probing tests S
                       currently supported options:
                       'singlepath': each test is a single probing path given in the input
                       default: 'singlepath'

                -- Simulation options --
                trialseed: random seed for the trial (randomly chosen if none is provided)
                stopOnceCovered: end the trial when all target edges are covered
                totalcov: target edges to cover (all probeable edges by default)
                maxsteps: maximum number of timesteps (50 by default)
                k: expected number of tests to probe per timestep (None by default)
                   if specified, each test i is independently probed per timestep with probability min(1,k*q[i])
                   if None, one test is chosen per timestep from the distribution q (process described in the original paper)

    :return: a list of sets of (src,dst) pairs indicating which paths are probed at which time
    """

    trialseed = kwargs.get('trialseed', random.randint(-MAXINT-1, MAXINT))
    rng = random.Random(trialseed)
    testseed = rng.randint(-MAXINT-1, MAXINT)
    simseed = rng.randint(-MAXINT-1, MAXINT)

    testset = generate_tests(paths, mode=kwargs.get('tests', 'singlepath'), seed=testseed)

    q = compute_distribution(graph, testset, kwargs.get('importance', None), kwargs.get('goal', 'max'))

    stopOnceCovered = kwargs.get('stopOnceCovered', False)
    if stopOnceCovered:
        totalcov = kwargs.get('totalcov', ps.all_edge_set(paths))
    else:
        totalcov = None

    maxsteps = kwargs.get('maxsteps', 50)
    k = kwargs.get('k', None)

    return chkmrt_trial(testset, q, simseed, stopOnceCovered, paths, totalcov, maxsteps, k)
