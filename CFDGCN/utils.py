import numpy as np
import math
import matplotlib.pyplot as plt
from os import PathLike
from typing import Dict, Union, Tuple, List

SU2_SHAPE_IDS = {
    'line': 3,
    'triangle': 5,
    'quad': 9,
}

def get_mesh_graph(mesh_filename: Union[str, PathLike],
               dtype: np.dtype = np.float32
               ) -> Tuple[np.ndarray, np.ndarray, List[List[List[int]]], Dict[str, List[List[int]]]]:
    def get_rhs(s: str) -> str:
        return s.split('=')[-1]

    marker_dict = {}
    with open(mesh_filename) as f:
        for line in f:
            if line.startswith('NPOIN'):
                num_points = int(get_rhs(line))
                mesh_points = [[float(p) for p in f.readline().split()[:2]]
                               for _ in range(num_points)]
                nodes = np.array(mesh_points, dtype=dtype)

            if line.startswith('NMARK'):
                num_markers = int(get_rhs(line))
                for _ in range(num_markers):
                    line = f.readline()
                    assert line.startswith('MARKER_TAG')
                    marker_tag = get_rhs(line).strip()
                    num_elems = int(get_rhs(f.readline()))
                    marker_elems = [[int(e) for e in f.readline().split()[-2:]]
                                    for _ in range(num_elems)]
                    # marker_dict[marker_tag] = np.array(marker_elems, dtype=np.long).transpose()
                    marker_dict[marker_tag] = marker_elems

            if line.startswith('NELEM'):
                edges = []
                triangles = []
                quads = []
                num_edges = int(get_rhs(line))
                for _ in range(num_edges):
                    elem = [int(p) for p in f.readline().split()]
                    if elem[0] == SU2_SHAPE_IDS['triangle']:
                        n = 3
                        triangles.append(elem[1:1+n])
                    elif elem[0] == SU2_SHAPE_IDS['quad']:
                        n = 4
                        quads.append(elem[1:1+n])
                    else:
                        raise NotImplementedError
                    elem = elem[1:1+n]
                    #[0 1] [1 2] [2 0]
                    edges += [[elem[i], elem[(i+1) % n]] for i in range(n)]
                edges = np.array(edges, dtype=np.compat.long).transpose() #dtype=np.long).transpose()
                # triangles = np.array(triangles, dtype=np.long)
                # quads = np.array(quads, dtype=np.long)
                elems = [triangles, quads]
    return nodes, edges, elems, marker_dict

# (ΔTriangle) = (1/2) |x1(y2 − y3) + x2(y3 - y1) + x3(y1 − y2)|
# (Δquad)     = 1/2 {(x1y2 + x2y3 + x3y4 + x4y1) − (x2y1 + x3y2 + x4y3 + x1y4)}
# Centroid of triangle = ((x1+x2+x3)/3 , (y1+y2+y3)/3)
# Centroid of quad     = ((x1+x2+x3+x4)/4 , (y1+y2+y3+y4)/4)
def calcFVMesh(nodeIndex: 'nodeIndex', nodeList: 'coordinatesList'):
    if len(nodeIndex) == 3:
        x1 = nodeList[nodeIndex[0]][0]
        y1 = nodeList[nodeIndex[0]][1]
        x2 = nodeList[nodeIndex[1]][0]
        y2 = nodeList[nodeIndex[1]][1]
        x3 = nodeList[nodeIndex[2]][0]
        y3 = nodeList[nodeIndex[2]][1]
        volume=0.5*abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
        centroid = [(x1+x2+x3)/3 , (y1+y2+y3)/3]
    elif len(nodeIndex) == 4:
        x1 = nodeList[nodeIndex[0]][0]
        y1 = nodeList[nodeIndex[0]][1]
        x2 = nodeList[nodeIndex[1]][0]
        y2 = nodeList[nodeIndex[1]][1]
        x3 = nodeList[nodeIndex[2]][0]
        y3 = nodeList[nodeIndex[2]][1]
        x4 = nodeList[nodeIndex[3]][0]
        y4 = nodeList[nodeIndex[3]][1]
        volume=0.5*((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (x2*y1 + x3*y2 + x4*y3 + x1*y4))
        centroid = [(x1+x2+x3+x4)/4 , (y1+y2+y3+y4)/4]
    else:
        raise NotImplementedError 
    return volume,centroid

def calcFVFields(nodeIndex, fields):
    if len(nodeIndex) == 3:
        var1 = fields[nodeIndex[0]]
        var2 = fields[nodeIndex[1]]
        var3 = fields[nodeIndex[2]]
        FVFields = (var1+var2+var3)/3
    elif len(nodeIndex) == 4:
        var1 = fields[nodeIndex[0]]
        var2 = fields[nodeIndex[1]]
        var3 = fields[nodeIndex[2]]
        var4 = fields[nodeIndex[4]]
        FVFields = (var1+var2+var3+var4)/4
    else:
        raise NotImplementedError 
    return FVFields

def calcEdge(edges: 'edgeConnectivity', nodeList: 'coordinatesList'):
    magSf = [] #edge length
    Cf    = [] #midpoint of edge
    for i in edges:
        x1 = nodeList[i[0]][0]
        y1 = nodeList[i[0]][1]
        x2 = nodeList[i[1]][0]
        y2 = nodeList[i[1]][1]
        magSf.append(math.sqrt((x2-x1)**2+(y2-y1)**2)) #area/length of edge
        Cf.append([(x2+x1)*0.5, (y2+y1)*0.5]) # midpoint of edge
    return magSf,Cf

## Getting nearbourElementIndex
def getNeighbour(elems,boundaryPatch):
    trielemsEdge= []
    quadelemsEdge= []
    for items in elems[0]:
        trielemsEdge.append(sorted(items))
    for items in elems[1]:
        quadelemsEdge.append(sorted(items))
    elemsMasterlist = trielemsEdge+quadelemsEdge
    
    bcEdge=[]
    bcEdge_str=[]
    for items in boundaryPatch:
        bcEdge.append(sorted(items))
        bcEdge_str.append(str(items[0])+'_'+str(items[1]))

    trielemsEdge_str=[]
    for items in trielemsEdge: #01 12 02
        edge01 = str(items[0])+'_'+str(items[1])
        edge02 = str(items[1])+'_'+str(items[2])
        edge03 = str(items[0])+'_'+str(items[2])
        trielemsEdge_str.append([edge01,edge02,edge03])

    quadelemsEdge_str=[]
    for items in quadelemsEdge: #01 12 23 02 03 13
        edge01 = str(items[0])+'_'+str(items[1])
        edge02 = str(items[1])+'_'+str(items[2])
        edge03 = str(items[2])+'_'+str(items[3])
        edge04 = str(items[0])+'_'+str(items[2])
        edge05 = str(items[0])+'_'+str(items[3])
        edge06 = str(items[1])+'_'+str(items[3])
        quadelemsEdge_str.append([edge01,edge02,edge03,edge04,edge05,edge06])
    elemsEdge_strMaster = trielemsEdge_str + quadelemsEdge_str
    
    Nei_elemsMasterlist=[]
    n=0
    for indexI, items in enumerate(elemsEdge_strMaster):
        neighbour=[]
        for indexi, i in enumerate(items):
            isInternalEdge = False;
            for indexS, search in enumerate(elemsEdge_strMaster):
                for indexj, j in enumerate(search):
                    if indexI!=indexS and i==j:
                        neighbour.append(indexS)
                        isInternalEdge = True;
                        n = n + 1
                        break
                if isInternalEdge:
                    break
            if not isInternalEdge:
                neighbour.append(-1)
                #m = m + 1
        Nei_elemsMasterlist.append(neighbour)
    print('internal edges = ',n)
    
    BCneighbour=[]   
    m=0
    for indexI, i in enumerate(bcEdge_str):
        isBoundaryEdge = False
        for indexS, search in enumerate(elemsEdge_strMaster):
            for indexj, j in enumerate(search):
                if i==j:
                    BCneighbour.append(indexS)
                    isBoundaryEdge = True
                    m = m + 1
                    break
            if isBoundaryEdge:
                break
        if not isBoundaryEdge:
            BCneighbour.append(-1)
            print(i)
    print('boundary edges = ',m)            
    return elemsMasterlist, Nei_elemsMasterlist, BCneighbour

