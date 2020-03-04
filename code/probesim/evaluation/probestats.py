'''
This module contains statistics helper functions
to be used with probesim simulations.
'''

import math
from .. import util as ps

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

from collections import namedtuple


def freqtodist(freq):
    '''
    Takes a frequency count and returns a histogram.
    This means, in the returned dictionary, the keys are frequency counts
    and the values are the number of times that frequency count occurred.
    '''
    ldd = {}

    for v in itervalues(freq):
        ldd[v] = ldd.get(v, 0) + 1

    return ldd


def disttolist(h):
    '''
    Takes a histogram (as in the return value from the method above)
    and returns a single list of all the values, appearing as many
    times as their counts.
    '''

    L = []
    for count, num in iteritems(h):
        L += [count] * num

    return L


def add_zero_counts(freqcounts, total_set=None):
    if total_set is not None:
        for item in total_set:
            freqcounts.setdefault(item, 0)

    return freqcounts


def avgload(h):
    '''
    This function takes a histogram dictionary
    and returns an average.

    Tested, faster than numpy.average
    '''
    s = 0.0
    t = 0.0

    for count in h:
        s = s + h[count] * count
        t = t + h[count]

    return s / t


def sumhist(h, hp):
    '''
    This function sums two histogram dictionaries.
    '''

    s = {}

    if len(h) == 0 and len(hp) == 0:
        return s

    if len(h) == 0:
        return hp

    if len(hp) == 0:
        return h

    for k in range(min(min(h), min(hp)), max(max(h), max(hp)) + 1):
        v = h.get(k, 0) + hp.get(k, 0)
        if v > 0:
            s[k] = v

    return s


def adjust_edge_fc(FC, total_edge_set=None, total_path_dict=None):
    if total_edge_set is None:
        if total_path_dict is None:
            return FC
        else:
            total_edge_set = ps.all_edge_set(total_path_dict)

    return add_zero_counts(FC, total_edge_set)


def eload(plist, total_edge_set=None, total_path_dict=None):
    '''
    This function returns a frequency count of edges in a path list
    as a dictionary.
    '''

    return adjust_edge_fc(ps.edgesinpaths(plist, histo=True), total_edge_set, total_path_dict)


def eloadpd(pd, total_edge_set=None, total_path_dict=None):
    '''
    This function returns a frequency count of edges in a path dictionary
    as a dictionary.
    '''

    return adjust_edge_fc(ps.edgesinpd(pd, histo=True), total_edge_set, total_path_dict)


def eloaddist(plist, total_edge_set=None, total_path_dict=None):
    '''
    This function returns a histogram of edge frequency in a path list.
    '''

    return freqtodist(eload(plist, total_edge_set, total_path_dict))


def adjust_node_fc(FC, total_node_set=None, total_path_dict=None):
    if total_node_set is None:
        if total_path_dict is None:
            return FC
        else:
            total_node_set = set(ps.nodesinpd(total_path_dict, histo=True))

    return add_zero_counts(FC, total_node_set)


def nload(paths, total_node_set=None, total_path_dict=None):
    '''
    This function returns a frequency count of nodes in a path list
    as a dictionary.
    '''
    ld = {}

    for p in paths:
        for n in p:
            ld[n] = ld.get(n, 0) + 1

    return adjust_node_fc(ld, total_node_set, total_path_dict)


def nloaddist(paths, total_node_set=None, total_path_dict=None):
    '''
    This function returns a histogram of node counts in a path set.
    '''
    return freqtodist(nload(paths, total_node_set, total_path_dict))


def adjust_probe_fc(FC, total_probe_set=None, total_path_dict=None):
    if total_probe_set is None:
        if total_path_dict is None:
            return FC
        else:
            total_probe_set = ps.probesinpd(total_path_dict)

    return add_zero_counts(FC, total_probe_set)


def sload(paths, total_probe_set=None, total_path_dict=None):
    '''
    This function returns the frequency of nodes appearing as sources
    or destinations of probes in a path list.
    '''
    ld = {}
    for p in paths:
        src = p[0]
        dst = p[-1]
        ld[src] = ld.get(src, 0) + 1
        ld[dst] = ld.get(dst, 0) + 1

    return adjust_probe_fc(ld, total_probe_set, total_path_dict)


def sloaddist(paths, total_probe_set=None, total_path_dict=None):
    '''
    This function returns a histogram of probing node counts in a path set.
    '''
    return freqtodist(sload(paths, total_probe_set, total_path_dict))


def times_to_intervals(timedict, maxtime=None):
    '''
    This function takes a dictionary of probe times (keys are components, values are lists of times)
    and returns a dictionary of time interval lengths between probes (keys are components, values are lists of interval lengths).

    The parameter maxtime is used to determine the length of the last interval.
    If maxtime is None, then the last interval length is not reported in the returned result.
    This means that an item that is never probed has an empty list as a value in the returned dictionary.
    '''
    intervals = dict()
    for comp, times in iteritems(timedict):
        D = []
        if len(times) > 0:
            D = [times[0]]
            for i in range(1, len(times)):
                if times[i] != times[i-1]:
                    D.append(times[i] - times[i-1])

        if maxtime is not None:
            if len(times) > 0 and times[-1] != maxtime:
                D.append(maxtime - times[-1])
            else:
                D.append(maxtime)

        intervals[comp] = D

    return intervals


def times_to_delays(timedict):
    '''
    This function takes a dictionary of probe times (keys are components, values are lists of times)
    and returns a dictionary of delays (keys are components, values are lists of delays).
    '''
    delays = dict()
    for comp, times in iteritems(timedict):
        if len(times) <= 0:
            D = [float("inf")]
        else:
            D = [times[0]]
            for i in range(len(times) - 1):
                if times[i + 1] != times[i]:
                    D.append(times[i + 1] - times[i])

        delays[comp] = D

    return delays


def timelist_to_timeloaddict(timelist):
    '''
    This function takes a list of times and produces a dictionary of loads per timestep.
    '''
    TLD = dict()

    for t in timelist:
        TLD[t] = TLD.get(t, 0) + 1

    return TLD


def timelist_to_timeloads(timelist, maxtime=None):
    '''
    This function takes a list of times and produces a list of loads per timestep.
    The output is a list of loads from 0 up to maxtime.

    Note that if maxtime is not provided, the last recorded time in the input list
    will be used as the length of the returned list.
    '''

    TLD = timelist_to_timeloaddict(timelist)

    if maxtime is None:
        maxtime = max(TLD) + 1

    TL = []

    for i in range(maxtime):
        TL.append(TLD.get(i, 0))

    return TL


def stdev(valuelist, mean=None):
    """ Returns standard deviation of a list.
        Tested equivalent to but faster than numpy.std

        If the mean is already available it can be passed in as a parameter to avoid recomputation"""
    N = float(len(valuelist))
    if mean is None: mean = math.fsum(valuelist) / N
    VarList = [(val - mean) ** 2 for val in valuelist]
    return math.sqrt(math.fsum(VarList) / N)


def avgDictValue(d):
    '''
    This function returns the average value in a dictionary.

    It is computed as sum(all values) / len(dictionary).
    '''

    return math.fsum(itervalues(d)) / float(len(d))


def pctDictValue(d, pct=None):
    return percentile(itervalues(d), pct)


def pctHist(h, pct=None):
    return percentile(disttolist(h), pct)


def NIST_helper(L, N=None, p=None):
    """
    Computes percentile p of sorted list L of length N using definition in NIST handbook Sec 7.2.5.2.
    :param L: sorted list of values
    :param N: length of list
    :param p: percentile desired, in the interval [0,1], or None for median
    :return: computer percentile p of sorted list L
    """

    if N is None: N = len(L)

    if N < 1:
        return None
    elif N < 2:
        return L[0]

    if p is None:
        p = 0.5

    weight, base = math.modf(float(p) * (N + 1))
    base = int(base) - 1
    if base < 0:
        return L[0]
    elif base >= N - 1:
        return L[-1]
    else:
        return float(L[base]) + weight * (float(L[base + 1]) - float(L[base]))


class nan_sort_wrapper:
    def __init__(self, value, least=True, altkey=None):
        self.value = value
        self.least = least
        if altkey is None: altkey = lambda x: x
        self.altkey = altkey
    def __lt__(self, other):
        if not math.isnan(self.value) and not math.isnan(other.value):
            return self.altkey(self.value) < self.altkey(other.value)
        elif math.isnan(self.value) and math.isnan(other.value):
            return False
        else:
            return math.isnan(self.value) == self.least

def nan_sort_key(least=True, altkey=None):
    """
    This function returns a sorting key for data, guaranteeing that NaN values appear at the beginning (or end) of the sorted order.
    The default (least==True) is NaNs are at the beginning; set least=False to have NaNs appear at the end.
    The additional optional parameter altkey is for specifying a key parameter to be used on non-NaN values.
    """
    return lambda value: nan_sort_wrapper(value, least, altkey)


def percentile(I, pct=None, nosort=False):
    '''
    This function returns the value closest to the pct-percentile.

    If pct is None, the median value is returned.
    Here, median of an even-sized set of values is the average of the two middle values.

    To increase efficiency, if pct is a sequence type, then a sequence
    of percentile values is returned from a single sort of the
    values in the provided iterable.  (This sequence can include None,
    in which case the median is returned as defined above.)

    Definition of percentile is taken from NIST handbook Sec 7.2.5.2.
    '''

    if nosort and isinstance(I, (list, tuple)):
        L = I
    elif nosort:
        L = list(I)
    else:
        L = sorted(I, key=nan_sort_key())

    N = len(L)

    ptl = (pct,) if not hasattr(pct, '__iter__') else pct

    PL = []
    for p in ptl:
        PL.append(NIST_helper(L, N, p))

    return PL[0] if not hasattr(pct, '__iter__') else PL


def mmmPercentile(I, pct=None, nosort=False):
    """This function is like percentile above, but also returns the mean, min, and max of the values provided"""

    if nosort and isinstance(I, (list, tuple)):
        L = I
    elif nosort:
        L = list(I)
    else:
        L = sorted(I, key=nan_sort_key())

    Vsum = math.fsum(L)
    N = len(L)

    ptl = (pct,) if not hasattr(pct, '__iter__') else pct

    PL = []
    for p in ptl:
        PL.append(NIST_helper(L, N, p))

    return [Vsum / float(N), L[0], L[-1]] + PL


def mmm_helper(I):
    first = True
    vmin = None
    vmax = None
    vsum = 0
    count = 0
    for val in I:
        if first:
            vmin = val
            vmax = val
            first = False
        else:
            if val < vmin: vmin = val
            if val > vmax: vmax = val
        vsum += val
        count += 1

    return float(vsum), count, vmin, vmax

def mmm(I):
    "Returns the mean, min, and max of an iterable of values as a named tuple"
    mmm_tuple = namedtuple("mmm_tuple", ["mean", "min", "max"])
    vsum, count, vmin, vmax = mmm_helper(I)
    return mmm_tuple(vsum / count, vmin, vmax)


def mms(I):
    "Returns the sum, min, and max of an iterable of values as a named tuple"
    mms_tuple = namedtuple("mms_tuple", ["sum", "min", "max"])
    vsum, _, vmin, vmax = mmm_helper(I)
    return mms_tuple(vsum, vmin, vmax)
