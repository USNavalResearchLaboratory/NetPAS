import networkx as nx
from numpy import prod

# Construct Extended Generalized Fat Trees and special cases thereof
# Follow \"{O}hring et al.'s paper ('95 IPPS) for details of the construction

def recurseGFT(G, h, m, w, prefix, leftxpos, xwidth):
    '''
    Recursive function for constructing a GFT
    '''

    if h < 0:
        raise RuntimeError('recurseGFT called with h < 0!')
    
    # Label of first parent node is (h + 1, tuple(list(firstParentPrefix) + [0]))
    firstParentPrefix = prefix[:-1]
    
    for k in range(w**h):
        newLabel = (h,tuple(prefix + [k]))
#         xval = 0.0
#         for i in range(1,len(newLabel[1])+1):
#             xval += newLabel[1][-i] * w**(i+h) +k
#         G.add_node(newLabel, xpos=xval, ypos=h)
        G.add_node(newLabel, xpos=leftxpos + k * float(xwidth) / w**h, ypos=h)
        for i in range(w):
            # Add w edges up to previous level
            parentLabel = (h + 1, tuple(list(firstParentPrefix) + [k*w + i]))
            G.add_edge(newLabel, parentLabel)
            
    
    # For h = 0, just add w^h nodes (above); no subgraphs below them
    if h == 0:
        return
        
    for j in range(m):
        recurseGFT(G, h-1, m, w, prefix + [j], leftxpos + j * float(xwidth) / m, float(xwidth) / m)



def GFTgraph(h, m, w):
    '''
    Constuct a GFT graph (networkX object)
    
    GFTgraph does setup and then calls recurseGFT to do the recursive construction.
    
    '''
    
    G = nx.Graph()
    
    for i in range(w**h):
        G.add_node((h,tuple([i])), xpos=i, ypos=h)
    
    for j in range(m):
        recurseGFT(G, h-1, m, w, [j], j * float(w**h) / m, float(w**h)/m)
    
    return G
    
def drawGFT(G):
    '''
    Draw a GFT
    '''
    
    posDict = {}
    xpos=nx.get_node_attributes(G,'xpos')
    ypos=nx.get_node_attributes(G,'ypos')
    for n in G.nodes():
        posDict[n] = (xpos[n],ypos[n])
    nx.draw_networkx(G, posDict)
#     plt.show()

###############
# XGFT versions that generalize GFT code
###############

def recurseXGFT(G, h, mList, wList, prefix, leftxpos, xwidth):
    '''
    Recursive function for constructing a XGFT
    '''

    if h < 0:
        raise RuntimeError('recurseXGFT called with h < 0!')
    
    # Label of first parent node is (h + 1, tuple(list(firstParentPrefix) + [0]))
    firstParentPrefix = prefix[:-1]
    

    if abs(int(prod(wList[:-1])) - prod(wList[:-1])) > 0.01:
        raise RuntimeError('int(prod(wList[:-1])) and prod(wList[:-1]) differ by more than 0.01!')
    
    for k in range(int(prod(wList[:-1]))):
        newLabel = (h,tuple(prefix + [k]))
#         xval = 0.0
#         for i in range(1,len(newLabel[1])+1):
#             xval += newLabel[1][-i] * w**(i+h) +k
#         G.add_node(newLabel, xpos=xval, ypos=h)
        G.add_node(newLabel, xpos=leftxpos + k * float(xwidth) / prod(wList[:-1]), ypos=h)
        for i in range(wList[-1]):
            # Add wList[-1] edges up to previous level
            parentLabel = (h + 1, tuple(list(firstParentPrefix) + [k*wList[-1] + i]))
            G.add_edge(newLabel, parentLabel)
            
    
    # For h = 0, just add w^h nodes (above); no subgraphs below them
    if h == 0:
        return
    
    # We've been passed a longer mList than we need---use the penultimate entry
    for j in range(mList[-2]):
        recurseXGFT(G, h-1, mList[:-1], wList[:-1], prefix + [j], leftxpos + j * float(xwidth) / mList[-1], float(xwidth) / mList[-1])



def XGFTgraph(h, mList, wList):
    '''
    Constuct an XGFT graph (networkX object)
    
    XGFTgraph does setup and then calls recurseXGFT to do the recursive construction.
    
    '''
    
    G = nx.Graph()
    
    for i in range(prod(wList)):
        G.add_node((h,tuple([i])), xpos=i, ypos=h)
    
    for j in range(mList[-1]):
        recurseXGFT(G, h-1, mList, wList, [j], j * float(prod(wList)) / mList[-1], float(prod(wList))/mList[-1])
    
    return G
    


    
def drawXGFT(G):
    '''
    Draw an XGFT
    '''
    
    posDict = {}
    xpos=nx.get_node_attributes(G,'xpos')
    ypos=nx.get_node_attributes(G,'ypos')
    for n in G.nodes():
        posDict[n] = (xpos[n],ypos[n])
    nx.draw_networkx(G, posDict)
#     plt.show()

