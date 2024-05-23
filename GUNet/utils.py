import numpy as np
import math
import matplotlib.pyplot as plt
from os import PathLike
from typing import Dict, Union, Tuple, List
import pickle

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
    if len(nodeIndex) == 3: # TRIANGLE
        x1 = nodeList[nodeIndex[0]][0]
        y1 = nodeList[nodeIndex[0]][1]
        x2 = nodeList[nodeIndex[1]][0]
        y2 = nodeList[nodeIndex[1]][1]
        x3 = nodeList[nodeIndex[2]][0]
        y3 = nodeList[nodeIndex[2]][1]
        volume=0.5*abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
        centroid = [(x1+x2+x3)/3 , (y1+y2+y3)/3]
    elif len(nodeIndex) == 4: # RECTANGLE
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
        var4 = fields[nodeIndex[3]]
        FVFields = (var1+var2+var3+var4)/4
    else:
        raise NotImplementedError 
    return FVFields
    
def calcFVFlagsTri(nodeIndex, flags):
    if len(nodeIndex) == 3:
        var1 = flags[nodeIndex[0]]
        var2 = flags[nodeIndex[1]]
        var3 = flags[nodeIndex[2]]
        FVFields = (np.sum([var1,var2,var3]) == 2) # If a triangle contains an airfoil edge, return True. False otherwise.
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
# def getNeighbour(elems,boundaryPatch):
#     trielemsEdge= []
#     quadelemsEdge= []
#     for items in elems[0]:
#         trielemsEdge.append(sorted(items))
#     for items in elems[1]:
#         quadelemsEdge.append(sorted(items))
#     elemsMasterlist = trielemsEdge+quadelemsEdge
    
#     bcEdge=[]
#     bcEdge_str=[]
#     for items in boundaryPatch:
#         bcEdge.append(sorted(items))
#         bcEdge_str.append(str(items[0])+'_'+str(items[1]))

#     trielemsEdge_str=[]
#     for items in trielemsEdge: #01 12 02
#         edge01 = str(items[0])+'_'+str(items[1])
#         edge02 = str(items[1])+'_'+str(items[2])
#         edge03 = str(items[0])+'_'+str(items[2])
#         trielemsEdge_str.append([edge01,edge02,edge03])

#     quadelemsEdge_str=[]
#     for items in quadelemsEdge: #01 12 23 02 03 13
#         edge01 = str(items[0])+'_'+str(items[1])
#         edge02 = str(items[1])+'_'+str(items[2])
#         edge03 = str(items[2])+'_'+str(items[3])
#         edge04 = str(items[0])+'_'+str(items[2])
#         edge05 = str(items[0])+'_'+str(items[3])
#         edge06 = str(items[1])+'_'+str(items[3])
#         quadelemsEdge_str.append([edge01,edge02,edge03,edge04,edge05,edge06])
#     elemsEdge_strMaster = trielemsEdge_str + quadelemsEdge_str
    
#     Nei_elemsMasterlist=[]
#     n=0
#     for indexI, items in enumerate(elemsEdge_strMaster):
#         neighbour=[]
#         for indexi, i in enumerate(items):
#             isInternalEdge = False;
#             for indexS, search in enumerate(elemsEdge_strMaster):
#                 for indexj, j in enumerate(search):
#                     if indexI!=indexS and i==j:
#                         neighbour.append(indexS)
#                         isInternalEdge = True;
#                         n = n + 1
#                         break
#                 if isInternalEdge:
#                     break
#             if not isInternalEdge:
#                 neighbour.append(-1)
#                 #m = m + 1
#         Nei_elemsMasterlist.append(neighbour)
#     print('internal edges = ',n)
    
#     BCneighbour=[]   
#     m=0
#     for indexI, i in enumerate(bcEdge_str):
#         isBoundaryEdge = False
#         for indexS, search in enumerate(elemsEdge_strMaster):
#             for indexj, j in enumerate(search):
#                 if i==j:
#                     BCneighbour.append(indexS)
#                     isBoundaryEdge = True
#                     m = m + 1
#                     break
#             if isBoundaryEdge:
#                 break
#         if not isBoundaryEdge:
#             BCneighbour.append(-1)
#             print(i)
#     print('boundary edges = ',m)            
#     return elemsMasterlist, Nei_elemsMasterlist, BCneighbour

def getNeighbour(elems):
    x = elems
    owner = {}
    neighbor = {}
    index = {}
    for i,elem in enumerate(x):
        n = len(elem)
        edges = [[elem[i], elem[(i+1) % n]] for i in range(n)]
        for e in edges:
            ord_e = (min(e[0],e[1]), max(e[0],e[1]) )
            if ord_e not in index:
                index[ ord_e ] = [i]
            else:
                index[ ord_e ] += [i]
                
    for k, value in index.items():
        if len(value) == 2:
            neighbor[value[0]] = neighbor.get(value[0],[]) + [value[1]]
            neighbor[value[1]]  = neighbor.get(value[1],[]) + [value[0]]
    owner = x
    
    return index, owner, neighbor

def getFVattrs(su2meshpath):
    """
    Extracts node and edge-attributes from a given su2 file mesh,
    Returns:  a dictinary {"nodeattrs": (Tuple of numpy arrays), "edgeattrs": (Tuple of numpy arrays)}
    """
    ## Calculate cell volume and centriod
    return_dict = {"nodeattrs": None, "edgeattrs": None}
    # ------ Compute  Node attributes (cellcenter coordinates, cell volumes, cellcenter u,v,p )---------- 
    triVol = []
    quadVol = []
    triC = []
    quadC = []
    trifields = []
    quadfields = []
    nodes, edges, elems, marker_dict = get_mesh_graph(su2meshpath)
    for tri in elems[0]:
        triVol.append(calcFVMesh(tri,nodes)[0])
        triC.append(calcFVMesh(tri,nodes)[1])
        trifields.append(calcFVFields(tri,nodes))

    for quad in elems[1]:
        quadVol.append(calcFVMesh(quad,nodes)[0])
        quadC.append(calcFVMesh(quad,nodes)[1]) 
        quadfields.append(calcFVFields(quad,nodes))

    volume = triVol+quadVol #[triVol,quadVol]
    centroid = triC+quadC #[triC,quadC] 
    centroildFvfields = trifields+quadfields
    return_dict["nodeattrs"] = (np.array(centroid),np.array(volume),np.array(centroildFvfields))

    # ---- compute edge attributes --------------
    magSf,Cf = calcEdge(edges.transpose(),nodes)
    edge_indices = { (min(x,y),max(x,y)) :i for i,(x,y) in enumerate(zip(edges[0,:],edges[1,:])) } 
    # for x,y in edge_indices:
    #     edge_indices[(y,x)] = edge_indices[(x,y)]

    # coo_u = []
    # coo_v = []
    # Eattr = []
    # for i,(x,y) in enumerate(edge_indices):
    #     coo_u.append(x); coo_u.append(y)
    #     coo_v.append(y); coo_v.append(x)
    #     Eattr.append(np.vstack((np.array(magSf[i]),np.array(Cf[i]))))
    # E = np.hstack((coo_v,coo_v))
    # print(E.shape)
    # print(np.array(Eattr).shape)
    # airfoilPatch = marker_dict['airfoil']
    # magBf,CBf = calcEdge(airfoilPatch,nodes)
    # elemsOwner,neighbour,BCneighbour = getNeighbour(elems,airfoilPatch)
    edge_commoncells, _,_ = getNeighbour(elems[0]+elems[1])
    coo_u = []
    coo_v = []
    Eattr = []
    for k, value in edge_commoncells.items():
        if len(value) == 2:
            coo_u.append(value[0]); coo_u.append(value[1])
            coo_v.append(value[1]); coo_v.append(value[0])
            common_e_idx = edge_indices[k]
            Eattr.append((np.array(magSf[common_e_idx]),np.array(Cf[common_e_idx])))
            Eattr.append((np.array(magSf[common_e_idx]),np.array(Cf[common_e_idx])))

    # for i,cu in enumerate(elemsOwner):
    #     len_cu = len(cu)
    #     for j,cv in enumerate(neighbour[i]):
    #         if cv == -1: # (cu is an on-boundary cell.)
    #             continue
    #         if j<len_cu-1:
    #             common_e = (cu[j],cu[j+1])
    #         else:
    #             common_e = (cu[j],cu[0])
    #         common_e = (min(common_e[0],common_e[1]), max(common_e[0],common_e[1]))
    #         try:
    #             common_e_idx = edge_indices[common_e]
    #             Eattr.append((np.array(magSf[common_e_idx]),np.array(Cf[common_e_idx])))
    #             Eattr.append((np.array(magSf[common_e_idx]),np.array(Cf[common_e_idx])))
    #             coo_u.append(i); coo_u.append(cv)
    #             coo_v.append(cv); coo_v.append(i)
    #         except Exception as e:
    #             print(cu)
    #             print(cv)
    #             for v in cv:
    #                 print(v,' => ',elemsOwner[v])
    #             raise(e)

    # d['nodeattrs'][0] => data.x[0,1] (n x 2)
    # d['nodeattrs'][1] => cell volume (n x 1)
    # d['nodeattrs'][2] => data.y  (n x 2)
    # d['edgeattrs'][0] => Edgelist in coo format (2x|E|)
    # d['edgeattrs'][1] => Surf Area, Surf center coordinates (|E| x 3)
    E = np.array([coo_u,coo_v])
    Eattr = np.array(Eattr)
    return_dict["edgeattrs"] = (E,Eattr)
    return return_dict

def save_pickle(ob, fname):
    with open (fname, 'wb') as f:
        #Use the dump function to convert Python objects into binary object files
        pickle.dump(ob, f)

def load_pickle(fname):
    with open (fname, 'rb') as f:
        #Convert binary object to Python object
        ob = pickle.load(f)
        return ob