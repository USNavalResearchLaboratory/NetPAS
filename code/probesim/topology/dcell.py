import sys
sys.dont_write_bytecode = True

import networkx as nx

# Construct a DCell graph

# suffix for switches
# Want to know this elsewhere to check which nodes are switches
DCellSwitchSuffixList = ['s']
DCellSwitchSuffixTuple = tuple(DCellSwitchSuffixList)
DCellSwitchSuffixLen = len(DCellSwitchSuffixList)

def SwitchForPrefix(prefix):
    '''
    Return the name of the switch for the DCell_0 named by prefix.
    
    For a prefix list that names a DCell_0, return the name (as a tuple) of the switch in that DCell_0.
    '''
    
    return tuple(prefix + DCellSwitchSuffixList)


def IsNotSwitch(name):
    '''
    Return True if name is not the name of a switch.
    '''
    
    if tuple(name[-DCellSwitchSuffixLen:]) == DCellSwitchSuffixTuple:
        return False
    else:
        return True
    

def BuildDCells(dcellgraph, prefix, n, l, tTuple, gTuple):
    '''
    Implement code for constructing DCells from Guo et al. paper
    
    prefix is the network prefix (as a list) of the DCell_l
    n is the number of nodes in a DCell_0
    l is the level of DCell we're building here
    tTuple describes the number of servers in each size DCell
    gTuple describes the number of smaller DCells used to make a larger DCell
    '''
    
    if l == 0:
        # SwitchForPrefix(prefix) gives the name of the switch in this DCell_0
        switchTuple = SwitchForPrefix(prefix)
        dcellgraph.add_node(switchTuple)
        for i in range(n):
            nodeTuple = tuple(prefix + [i])
            dcellgraph.add_node(nodeTuple)
            dcellgraph.add_edge(nodeTuple, switchTuple)

    else:
        for i in range(gTuple[-l-1]):
            BuildDCells(dcellgraph, prefix + [i], n, l-1, tTuple, gTuple)
        for i in range(tTuple[-(l-1)-1]):
            newEdges = []
            for j in range(i+1, gTuple[-l-1]):
                # These are integer IDs within their respective DCell_{l-1}s
                u1 = j-1
                u2 = i
                # Convert integer IDs to suffixes of node names
                suff1 = []
                suff2 = []
                for z in range(l):
                    az1 = u1 % tTuple[-z-1]
                    suff1 = [az1] + suff1
                    u1 = (u1 - az1) / tTuple[-z-1]
                    az2 = u2 % tTuple[-z-1]
                    suff2 = [az2] + suff2
                    u2 = (u2 - az2) / tTuple[-z-1]
                dcellgraph.add_edge(tuple(prefix + [i] + suff1), tuple(prefix + [j] + suff2))
                newEdges.append([tuple(prefix + [i] + suff1), tuple(prefix + [j] + suff2)])



def DCellGraph(n, k):
    '''
    Create and return a DCell with n-port switches and levels 0,...,k.
    
    WARNING: The number of servers grows as c^{2^k} for c > n + 1/2.  For k = 3 and n = 6, there are 3.26 million servers.
    
    Nodes are named with their labels from the paper.  For a DCell_k, these are (k+1)-tuples.  Switches are included; these names may be distinctive, depending on the value returned by SwitchForPrefix().
    
    DCellGraph does setup and then calls the recursive function BuildDCells.  The returned object is a networkX graph.
    '''
    
    # Create empty nx graph
    # Pass the graph to BuildDCells for recursive construction
    
    G = nx.Graph()
    
    # tList[-k-1] has # of servers in DCell_k
    # gList[-k-1] has # of DCell_{k-1}s in a DCell_k
    # Construct these from back to front (so that ordering is same
    # as in the paper)
    tList = [n]
    gList = [1]
    for i in range(k+1):
        gList = [tList[0] + 1] + gList
        tList = [(tList[0] + 1) * tList[0]] + tList
    tTuple = tuple(tList)
    gTuple = tuple(gList)
    
    prefix = []

    BuildDCells(G, prefix, n, k, tTuple, gTuple)
    
    return G

########################
# Construct DCell routing paths

def dCellSuffix(uid, l, tTuple):
    '''
    Convert a uid in a DCell_{l-1} to an l tuple
    '''
    
    suff = []
    for i in range(l):
        ai = uid % tTuple[-i-1]
        suff = [ai] + suff
        uid = (uid - ai) / tTuple[-i-1]

    return suff
    

def GetLink(dcellgraph, prefix, sm, dm, tTuple):
    '''
    Return the (non-empty) edge in the middle of the recursively constructed routing path
    
    s and d are the first values that differ in the (k+1) tuples for the source and destination servers.
    prefix is the longest common prefix for the source and destination (k+1) tuples
    
    
    '''
    # Return a pair of tuples (nodes)
    
    # sm and dm are different so the elements returned are always different
    
    # If s < d, link goes between [s, d-1] and [d, s]
    
    m = len(prefix)
    k = len(tTuple) - 1
    
    if sm < dm:
        # convert d-1 and s to suffixes
        # return (prefix + [sm] + suff(dm-1), prefix + [dm] + suff(sm))
        return (tuple(prefix + [sm] + dCellSuffix(dm - 1, k-m, tTuple)), tuple(prefix + [dm] + dCellSuffix(sm, k-m, tTuple)))
    else: # d < s
        # convert s-1 and d to suffixes
        # return (prefix + [sm] + suff(dm), prefix + [dm] + suff(sm-1))
        return (tuple(prefix + [sm] + dCellSuffix(dm, k-m, tTuple)), tuple(prefix + [dm] + dCellSuffix(sm - 1, k-m, tTuple)))
        
    

def DCellRouting(dcellgraph, src, dst, tTuple):
    '''
    Return the DCell routing path from src to dst (assumed to be servers, not switches)
    
    If src == dst, return an empty list
    Otherwise, return a list of nodes [src, ..., dst] forming a path
    
    src and dst are (k+1) tuples
    '''
    
    if len(src) != len(dst):
        raise RuntimeError('src and dst have different lengths!')
    if src == dst:
        return []
    
    k = len(src) - 1
    
    prefix = []
    for i in range(len(src)):
        if src[i] == dst[i]:
            prefix.append(src[i])
        else:
            break
    m = len(prefix)
    if m == len(src) - 1:
        return [src, SwitchForPrefix(prefix), dst]
    (n1, n2) = GetLink(dcellgraph, prefix, src[m], dst[m], tTuple)
    if src != n1:
        # path1 will end with n1 in this case
        path1 = DCellRouting(dcellgraph, src, n1, tTuple)
    else:
        path1 = [n1]
    if n2 != dst:
        # path2 will start at n2 in this case
        path2 = DCellRouting(dcellgraph, n2, dst, tTuple)
    else:
        path2 = [n2]
    return path1 + path2



def RouteInDCellServers(dcellgraph, src, dst, n, k):
    '''
    Do routing in a DCell(n,k) from src to dst (assumed to be servers, not switches)
    
    Return a list of nodes, including switches, in a path from src to dst.  This is obtained using the algorithm in the DCell paper.  If src and dst are in the same DCell_{k+1} but separate DCell_{k}s, it will include the edge connecting their DCell_{k}s; the paths to and from this edge are constructed recursively.
    
    If src and dst are the same node, then 
    
    This does general setup and then calls DCellRouting, which then calls itself recursively.
    '''
    
    if src == dst:
        return [src]
    
    # tList[-k-1] has # of servers in DCell_k
    # gList[-k-1] has # of DCell_{k-1}s in a DCell_k
    # Construct these from back to front (so that ordering is same
    # as in the paper)
    tList = [n]
#     gList = [1]
    for i in range(k):
        tList = [(tList[0] + 1) * tList[0]] + tList
    tTuple = tuple(tList)

    return DCellRouting(dcellgraph, src, dst, tTuple)


def LowServerForSwitch(node):
    '''
    Return the name of the lowest-indexed server in the same DCell_0 as the switch node.
    '''
    
    return tuple(list(node[:-1]) + DCellSwitchSuffixList)
    
    

def RouteInDCell(dcellgraph, src, dst, n, k):
    '''
    Do routing in a DCell(n,k) from src to dst (allowed to be switches)
    
    Generalize to allow src and dst to be switches and not just servers.  If one of these is a switch, pick the lowest-indexed server in its DCell_0 as the src/dst.  Do the routing recursively.  If the routing goes through the switch, drop the server at the end of the path; if it does not, (ap/pre)pend the switch to the path.
    
    Return a list of nodes, including switches, in a path from src to dst.  This is obtained using the algorithm in the DCell paper.  If src and dst are in the same DCell_{k+1} but separate DCell_{k}s, it will include the edge connecting their DCell_{k}s; the paths to and from this edge are constructed recursively.
    
    If src and dst are the same node, then 
    
    This does general setup and then calls DCellRouting, which then calls itself recursively.
    '''
    
    if src == dst:
        return [src]
    
    # tList[-k-1] has # of servers in DCell_k
    # gList[-k-1] has # of DCell_{k-1}s in a DCell_k
    # Construct these from back to front (so that ordering is same
    # as in the paper)
    tList = [n]
#     gList = [1]
    for i in range(k):
        tList = [(tList[0] + 1) * tList[0]] + tList
    tTuple = tuple(tList)

    if not IsNotSwitch(src):
        tmpSrc = LowServerForSwitch(src)
        srcSwitch = True
    else:
        tmpSrc = src
        srcSwitch = False
    
    if not IsNotSwitch(dst):
        tmpDst = LowServerForSwitch(dst)
        dstSwitch = True
    else:
        tmpDst = dst
        dstSwitch = False
    
    tmpPath = DCellRouting(dcellgraph, tmpSrc, tmpDst, tTuple)
    # tmpPath goes from tmpSrc to tmpDst

    if (not srcSwitch) and (not dstSwitch):
        path = tmpPath
    elif (not srcSwitch) and dstSwitch:
        if dst == tmpPath[-2]:
            path = tmpPath[:-1]
        elif dst in tmpPath:
            raise RuntimeError('Switch dst appears at unexpected place in tmpPath!')
        else:
            path = tuple(list(tmpPath) + [dst])
    elif srcSwitch and (not dstSwitch):
        if src == tmpPath[1]:
            path = tmpPath[1:]
        elif src in tmpPath:
            raise RuntimeError('Switch src appears at unexpected place in tmpPath!')
        else:
            path = tuple([src] + list(tmpPath))
    elif srcSwitch and dstSwitch:
        # We know that src != dst, else we would have returned above
        if src == tmpPath[1]:
            newTmpPath = tmpPath[1:]
        elif src in tmpPath:
            raise RuntimeError('Switch src appears at unexpected place in tmpPath!')
        else:
            newTmpPath = tuple([src] + list(tmpPath))

        if dst == newTmpPath[-2]:
            path = newTmpPath[:-1]
        elif dst in newTmpPath:
            raise RuntimeError('Switch dst appears at unexpected place in tmpPath!')
        else:
            path = tuple(list(newTmpPath) + [dst])
    
    return path


def all_pairs_DCell_routes(dcellgraph, n, k):
    '''
    Compute the routing dictionary in a DCell
    
    
    '''
    
    paths = {}
    
    for s in dcellgraph.nodes():
        paths[s] = {}
        for d in dcellgraph.nodes():
            paths[s][d] = RouteInDCell(dcellgraph, s, d, n, k)
            
    return paths

def all_server_pairs_DCell_routes(dcellgraph, n, k):
    '''
    Compute the routing dictionary in a DCell only using servers as endpoints
    
    
    '''
    
    paths = {}
    
    for s in dcellgraph.nodes():
        if not IsNotSwitch(s):
            continue
        paths[s] = {}
        for d in dcellgraph.nodes():
            if not IsNotSwitch(d):
                continue
            paths[s][d] = RouteInDCellServers(dcellgraph, s, d, n, k)
            
    return paths
