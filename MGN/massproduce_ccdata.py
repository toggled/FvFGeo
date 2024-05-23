import math
import argparse, yaml, os, json, glob,re
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import numpy as np
from math import *
import openfoamparser as Ofpp
from process_vtk import *
import pyvista as pv
from time import time

def GcomputeSDF(pos,surf_bool):
    boundarycells = pos[surf_bool]
    # dist = cdist(boundarycells,pos)
    dist = torch.cdist(boundarycells,pos,p=2)
    # print(dist.size())
    return torch.min(dist,dim=0)[0]

def GcomputeSAF2(pos,surf_bool): # Assume pos, surf_bool are torch tensors
    def get_closest_points(internalcells, boundarycells):
        dist = torch.cdist(boundarycells,internalcells,p=2)
        # print('dist size: ',dist.size())
        closests = torch.argmin(dist,dim=0)
        return closests, boundarycells[closests]
        
    closest_point,closest_points_xyz = get_closest_points(pos,pos[surf_bool])
    saf_tmp = pos[:,:2] - closest_points_xyz[:,:2] 
    return saf_tmp

def getDSDF(pos, bd, theta_rot, theta_seg, inf=5):
    '''get dSDF representation of geometry given rotation angle, segment angle.'''
    # print('----- DSDF ---------')
    # print('type(pos) & shape(pos): ', type(pos), pos.size())
    # print('type(bd) & shape(bd): ',type(bd), bd.size())
    def order_clockwise(bd_xy):
        c = torch.mean(bd_xy,dim=0)
        h = bd_xy - c
        theta = torch.atan2(h[:,1],h[:,0]) #size MxN
        return torch.flipud(bd_xy[theta.sort()[1]])
    def sameSide(j,i):
        '''Check if boundary points j (Mx2) and all points i (Nx2)
        fall on the same side of the geometry (bool NxM)'''
        # c = torch.tensor([[0.5,0]]).to(j.device) #airfoil internal point
        c = torch.mean(j,dim=0).unsqueeze(0)
        j = order_clockwise(j); j1 = torch.cat([j[1:,:],j[0:1,:]])
        indi, indj = torch.meshgrid(torch.arange(i.size(0)), \
                                torch.arange(j.size(0)), indexing='ij')
        p0 = i[indi]; p1 = j[indj]; p2 = j1[indj]
        dir_i = (p1[:,:,1]-p0[:,:,1])*(p2[:,:,0]-p1[:,:,0])
        dir_i -= (p2[:,:,1]-p1[:,:,1])*(p1[:,:,0]-p0[:,:,0])
        dir_c = (j[:,1]-c[:,1])*(j1[:,0]-j[:,0])
        dir_c -= (j1[:,1]-j[:,1])*(j[:,0]-c[:,0])
        side = ((dir_i>0)==(dir_c<=0)).t() #bool size NxM
        return j,side

    def PDF(mean,x):
        '''chosen PDF to used to weigh points'''
        return torch.ones(x.size()).to(d.device) #if just uniform distribution
        # grad = 1/theta_seg**2;
        # w0 = (x-(mean-theta_seg/2))/(theta_seg**2)
        # w1 = (-x+(mean+theta_seg/2))/(theta_seg**2) #linear
        # return w0*(x<=mean)+w1*(x>mean) #if linear distribution
    
    def intPDF(mean,mRange):
        '''integral of chosen PDF used to weigh points'''
        return (mRange[1]-mRange[0]) #for uniform distribution
        # dw = PDF(mean,mRange); grad = 1/theta_seg**2;
        # w0 = 0.5*dw*(mRange-(mean-theta_seg/2))
        # w1 = 1 - (0.5*dw*((mean+theta_seg/2)-mRange)) #linear
        # w = (w0*(mRange<=mean)+w1*(mRange>mean))*(dw>=0)
        # return w[1]-w[0] #if linear distribution

    dSDF = []; j = pos[bd]; j, side = sameSide(j,pos)
    #Compute dSDF for ever segment centre theta_cen
    for theta_cen in torch.arange(0,2*torch.pi,theta_rot):
        #Compute segment range theta_ran
        theta_ran = torch.tensor([theta_cen-(theta_seg/2), theta_cen+(theta_seg/2)])
        dSDF_tCen = []
        
        #Compute dSDF_i for every node i in V
        h = j[:,0:1] - pos.t()[0:1,:] #size MxN
        k = j[:,1:] - pos.t()[1:,:] #size MxN
        theta_ij = torch.atan2(k,h) #size MxN
        theta_ij = theta_ij%(2*torch.pi) if theta_ran[0]>=0 else theta_ij
        
        #Compute distance from point i to every bd point within segment range AND same side
        ind = (theta_ran[0]<=theta_ij)*(theta_ij<=theta_ran[1])*side
        d = torch.sqrt(k*k+h*h); d[d>inf]=inf #size MxN
        w = PDF(theta_cen,theta_ij) #size MxN
        dSDF_i = torch.sum((d*w)*ind,dim=0) #dSDF_i = torch.sum((d*ind),dim=0)
        w = torch.sum((w*ind),dim=0) #w=torch.sum(ind,dim=0) #sum of discrete weights
        
        #minimum angle range from point i to point j
        theta_minR = torch.zeros(2,theta_ij.size(1)).to(dSDF_i.device)
        ind = (theta_ran[0]<=theta_ij)*(theta_ij<=theta_ran[1]) # in segment range (but any side)
        theta_ij[~ind] = 2*torch.pi; theta_minR[0] = torch.min(theta_ij,dim=0)[0]
        theta_ij[~ind] = -2*torch.pi; theta_minR[1] = torch.max(theta_ij,dim=0)[0]
        
        #compute weight of minimum angle segment w_tMin
        w_tMin = intPDF(theta_cen,theta_minR)/intPDF(theta_cen,theta_ran)
        w_tMin[w_tMin<0] = 0
        dSDF_i = torch.nan_to_num(dSDF_i/w,nan=0)
        dSDF_tCen = w_tMin*dSDF_i + (1-w_tMin)*inf
        dSDF.append(dSDF_tCen.clone())
    dSDF = torch.stack(dSDF)
    return dSDF

def get_lastIteration(casepath, default = None, logfilename='log.sol'):
    """ 
    Returns the last iteration of an Openfoam Case. 
    :casepath = path to an openfoam case
    """
    find = r"\sTime = (\d+)"
    # logfile = os.path.join(casepath,'log.simpleFoam')
    # logfile = os.path.join(casepath,'log.sol')
    logfile = os.path.join(casepath,logfilename)
    text = "".join(open(logfile,'r').readlines())
    found = re.findall(find,text)
    if len(found):
        # print(found)
        return found[-1]
    else:
        return None
    
def preprocess(case, isfine = True):
    suffix = ['constant/polyMesh/points','constant/polyMesh/faces','constant/polyMesh/owner','constant/polyMesh/neighbour']
    for s in suffix:
        file = os.path.join(case,s)
        source_file = os.path.join(case,s+'.gz')
        if not os.path.isfile(file):
            os.system('gzip -d '+source_file)
    cC_file = os.path.join(case,'0/C')
    if not os.path.isfile(cC_file):
        cc_gz = 'gzip -d '+cC_file+'.gz'
        if not os.path.isfile(cC_file+'.gz'):
            os.system('postProcess -case '+case+' -func writeCellCentres -time 0')
        os.system(cc_gz)
    if isfine:
        # _iter = get_lastIteration(case,logfilename='log.simpleFoam')
        _iter = get_lastIteration(case)
    else:
        _iter = get_lastIteration(case)
        # print(_iter)
    if isfine:
        Cf_file = os.path.join(os.path.join(case,_iter),'Cf.gz')
        if not os.path.isfile(os.path.join(os.path.join(case,_iter),'Cf')):
            os.system('gzip -d '+Cf_file)
        if not os.path.isfile(os.path.join(os.path.join(case,_iter),'magSf')):
            magSf_file = os.path.join(os.path.join(case,_iter),'magSf.gz')
            os.system('gzip -d '+magSf_file)
        if not os.path.isfile(os.path.join(os.path.join(case,_iter),'Sf')):
            Sf_file = os.path.join(os.path.join(case,_iter),'Sf.gz')
            os.system('gzip -d '+Sf_file)
        
        if not os.path.isfile(os.path.join(os.path.join(case,_iter),'V')):
            V_file = os.path.join(os.path.join(case,_iter),'V.gz')
            os.system('gzip -d '+V_file)
    if not os.path.isfile(os.path.join(os.path.join(case,_iter),'U')):
        U_file = os.path.join(os.path.join(case,_iter),'U.gz')
        os.system('gzip -d '+U_file)
    if not os.path.isfile(os.path.join(os.path.join(case,_iter),'p')):
        p_file = os.path.join(os.path.join(case,_iter),'p.gz')
        os.system('gzip -d '+p_file)
    if not os.path.isfile(os.path.join(os.path.join(case,_iter),'nut')):
        nut_file = os.path.join(os.path.join(case,_iter),'nut.gz')
        os.system('gzip -d '+nut_file)

def get_cellCentres(mesh,case):
    mesh.read_cell_centres(os.path.join(case,'0/C'))
    return mesh.cell_centres

def get_cellVolume(mesh, last_iterationfolder):
    return Ofpp.parse_internal_field(os.path.join(last_iterationfolder+"/V"))

def get_surfaceArea(mesh, last_iterationfolder):
    mesh.read_face_areas(last_iterationfolder+'/magSf')
    return mesh.face_areas
    
def get_surfaceCentre(mesh, last_iterationfolder):
    mesh.read_cell_centres(last_iterationfolder+'/Cf')
    return mesh.cell_centres

def get_surfaceNormal(mesh,last_iterationfolder):
    mesh.read_cell_centres(last_iterationfolder+'/Sf')
    return mesh.cell_centres

def loaddict_json(jsonfile):
    with open(jsonfile, 'r') as f:
        return json.load(f)
    
def get_edges(internal_vtu):
    internal_edges = internal_vtu.extract_feature_edges(boundary_edges=False, feature_edges=False, manifold_edges=True)
    boundary_edges = internal_vtu.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
    return (internal_edges,boundary_edges)

def compute_Cellmarkers_numpy(internal_vtu,bd_edges):
    """ return markers """
    marker = np.array([False]*internal_vtu.cell_data['U'].shape[0],dtype = bool) # mark all points as -1
    marker[bd_edges.cell_data['vtkOriginalCellIds']] = True # mark aerofoil + farfield to 1
    # marker[internal_vtu.cell_data['U'][:, 0]==0] = False # Mark only aerofoil to 0
    return marker

def compute_normalOutward(case,cC, normal_source = False, unit_normal = True, magSf = False):
    wallnormalf = os.path.join(case,'aerofoilPatch.txt')
    normals = np.zeros_like(cC)
    if magSf:
        surf_magSf = np.zeros_like(cC[:,0])
    if normal_source:
        source_n = np.zeros_like(cC)
    with open(wallnormalf,'r') as f:
        f.readline()
        for line in f.readlines():
            split_l = line.split()
            inward_normal = [float(i) for i in split_l[-3:]]
            cell_containedin = int(split_l[0])
            if magSf:
                surf_magSf[cell_containedin] = math.sqrt(inward_normal[0]**2 + inward_normal[1]**2+inward_normal[2]**2)
                
            if unit_normal:
                normals[cell_containedin] = -(np.array(inward_normal)/np.linalg.norm(inward_normal))
            else:
                normals[cell_containedin] = -np.array(inward_normal)
            if normal_source:
                src = [float(i) for i in split_l[4:7]]
                source_n[cell_containedin] = np.array(src)
    if normal_source:
        if magSf:
            return (surf_magSf, source_n[:,:2],normals) # source => origin/mid point of the (normal vectors), 
        else:
            return (source_n[:,:2],normals)
    else:
        normals = normals[:,:2]
        return normals
    
def get_graph(mesh,case,_include, farfieldmarker, cC,V,magSf,Cf,num_nodes,converged_folder, Uinf, \
              alpha, isfine=True, onlysurf = False, surfinfo = False ):
    points = []
    edge_list = []
    edge_attr = [] 
    node_attr = []
    # TO DO: Add airfoil points as graph nodes
    nodemarker = [] # TO DO : FIll up nodemarker: 0 => airfoil points, 1 => farfield cells, -1 => internal cells.
    if surfinfo:
        owner = []
        neighbor = []
        internal_surf_flag = []
        cell_next = []
        sf = []
        magSf = []
    # _include = (cC[:,0]>=xlim[0]) & (cC[:,0]<=xlim[1]) \
    #         & (cC[:,1]>=ylim[0]) & (cC[:,1]<=ylim[1])
    if isfine:
        # cc = cC[_include]
        inverted_index = {}
        count = 0
        for i,boolean in enumerate(_include):
            if boolean:
                if farfieldmarker[i]: # FF
                    nodemarker.append(1)
                else:
                    nodemarker.append(-1) # internal
                inverted_index[i] = count
                count+=1
        U = Ofpp.parse_internal_field(os.path.join(converged_folder,'U'))[:,:2]
        P = Ofpp.parse_internal_field(os.path.join(converged_folder,'p'))
        nut = Ofpp.parse_internal_field(os.path.join(converged_folder,'nut'))
        P = P.reshape(-1,1)
        nut = nut.reshape(-1,1)
        # print(U.shape,' ',P.shape)
        UVPnut = torch.from_numpy(np.concatenate([U,P,nut],axis=1))
        surf_magSf, surf_mid, norm = compute_normalOutward(case,cC,normal_source=True,magSf = True)
        _, _, norm_unnormalized = compute_normalOutward(case,cC,normal_source=True, unit_normal = False, magSf = True)
        n = []
        surf_nodeattr = []
        surf_nodexy = [] # Cf
        surf_edges = []
        surf_edgeattr = []
        surfcount = 0
        surf_UVP = []
        C = []
        indices_C = [] # For extracting P of adjacent cell-center
        n_unnorm = []
        
        for i,boolean in enumerate(_include):
            if (boolean):
                ipos = inverted_index[i] # reindexed position
                bool2 = mesh.is_cell_on_boundary(i,b'aerofoil')
                if bool2:
                    surf_nodexy.append([surf_mid[i][0],surf_mid[i][1]])
                    if surfinfo: 
                        cell_next.append(ipos) # #cells next = #surfpoints
                    n.append([norm[i][0],norm[i][1]])
                    n_unnorm.append([norm_unnormalized[i][0],norm_unnormalized[i][1]])  
                    surf_nodeattr.append(0.0) # volume 0 at af-surface mid-points
                    surf_edges.append([ipos,surfcount+count])
                    surf_edges.append([surfcount+count,ipos])
                    surf_edgeattr.append(np.array([surf_magSf[i], surf_mid[i][0],surf_mid[i][1]])) 
                    surf_edgeattr.append(np.array([surf_magSf[i], surf_mid[i][0],surf_mid[i][1]])) 
                    surf_UVP.append([0.0,0.0,P[i][0],0.0])
                    surfcount+=1
                    C.append([cC[i,0],cC[i,1]])
                    indices_C.append(len(points))
                
                points.append([cC[i,0],cC[i,1]])
                if onlysurf:  continue
                
                f_i = mesh.cell_faces[i]
                # print(i,' => ',f_i)
                nbr_i = mesh.cell_neighbour_cells(i)
                node_attr.append(V[i])
                for x in nbr_i:
                    if x>0:
                        if _include[x]:
                            # print(x,' => ',mesh.cell_faces[x])
                            common_face = set(mesh.cell_faces[x]).intersection(f_i)
                            assert(len(common_face)==1)
                            if len(common_face):
                                common_face = common_face.pop()
                                if surfinfo:
                                    owner.append(inverted_index[mesh.owner[common_face]])
                                    neighbor.append(inverted_index[mesh.neighbour[common_face]])
                                    sf.append(Sf[common_face])

                                try:
                                    xpos = inverted_index[x]
                                    edge_list.append([ipos,xpos])
                                    attrs = np.hstack((magSf[common_face], Cf[common_face][:2]))
                                    edge_attr.append(attrs)
                                except Exception as e:
                                    print('common_face: ',common_face)
                                    print(i,',',x,' => ',ipos,',',xpos)
                                    raise(e)
        if onlysurf:
            surf_data = Data(val = torch.from_numpy(np.concatenate([np.array(C),np.array(surf_nodexy),np.array(n_unnorm)],axis=1)),\
                             indices = torch.tensor(indices_C))
            return surf_data
        elif surfinfo:
            internal_surf_flag = torch.tensor([True]*len(edge_list) + [False]*len(surf_edges))
            data = Data(internal_surf_flag = internal_surf_flag, owner = owner,neighbor = neighbor,cell_next = cell_next, sf = torch.from_numpy(np.array(sf)))
            return data
        else:
            # edge_attr = np.vstack(edge_attr)     
            # data = Data(x= torch.hstack((torch.tensor(points),u[_include,:],n[_include,:])), \
            #                 edge_index = torch.tensor(edge_list,dtype=torch.long).t().contiguous(),\
            #                node_attr=torch.tensor(node_attr),\
            #                edge_attr = torch.tensor(edge_attr),\
            #                surf = torch.tensor(surfbool),\
            #                y = UVPnut[_include]
            #                )
            edge_attr = np.vstack(edge_attr)  
            surf_edgeattr = np.vstack(surf_edgeattr)
            norm_internal = np.zeros_like(points)
            surfbool = np.array([False]*len(points) + [True]*len(surf_nodexy))
            n = np.concatenate([norm_internal,np.array(n)],axis = 0)
            points = np.concatenate([points,np.array(surf_nodexy)],axis = 0) 
            u = torch.tensor((np.array([np.cos(alpha), np.sin(alpha)])*Uinf).reshape(1, 2)*np.ones_like(points))
            edge_list = edge_list + surf_edges 
            node_attr = node_attr + surf_nodeattr
            nodemarker = np.array(nodemarker + [0]*len(surf_nodexy),dtype = int) 
            edge_attr = np.concatenate([edge_attr , surf_edgeattr],axis=0)
            UVPnut_internal = np.concatenate([U,P,nut],axis=1)[_include,:]
            UVPnut = np.concatenate([UVPnut_internal,np.array(surf_UVP,dtype=float)],axis=0)
            # add bidrection edge and edge attributes
            data = Data(x= torch.hstack((torch.tensor(points),torch.tensor(u),torch.tensor(n))), \
                        edge_index = torch.tensor(edge_list,dtype=torch.long).t().contiguous(),\
                        node_attr=torch.tensor(node_attr),\
                        edge_attr = torch.tensor(edge_attr),\
                        surf = torch.tensor(surfbool),\
                        y = torch.tensor(UVPnut),
                        markers = torch.tensor(nodemarker)
                       )
            return data
    else:
        points = []
        for i,boolean in enumerate(_include):
            if (boolean):
                points.append([cC[i,0],cC[i,1]])
        U = Ofpp.parse_internal_field(os.path.join(converged_folder,'U'))[:,:2]
        P = Ofpp.parse_internal_field(os.path.join(converged_folder,'p'))
        nut = Ofpp.parse_internal_field(os.path.join(converged_folder,'nut'))
        P = P.reshape(-1,1)
        nut = nut.reshape(-1,1)
        # print(U.shape,' ',P.shape)
        UVPnut = torch.from_numpy(np.concatenate([U,P,nut],axis=1))
        data = Data(x= torch.tensor(points),y=UVPnut[_include])
    return data

def get_graph_ns(mesh,case,_include, farfieldmarker, cC,V,magSf,Cf,Sf,num_nodes,converged_folder, Uinf, \
              alpha, isfine=True, onlysurf = False):
    points = []
    edge_list = []
    edge_attr = [] 
    node_attr = []
    # TO DO: Add airfoil points as graph nodes
    nodemarker = [] # TO DO : FIll up nodemarker: 0 => airfoil points, 1 => farfield cells, -1 => internal cells.
    # owner = []
    # neighbor = []
    internal_surf_flag = []
    owner_to_neighbor = []
    cell_next = []
    sf = []
    cf = []
    surf_normal_vectors = []
    # _include = (cC[:,0]>=xlim[0]) & (cC[:,0]<=xlim[1]) \
    #         & (cC[:,1]>=ylim[0]) & (cC[:,1]<=ylim[1])
    if isfine:
        # cc = cC[_include]
        inverted_index = {}
        count = 0
        for i,boolean in enumerate(_include):
            if boolean:
                if farfieldmarker[i]: # FF
                    nodemarker.append(1)
                else:
                    nodemarker.append(-1) # internal
                inverted_index[i] = count
                count+=1
        U = Ofpp.parse_internal_field(os.path.join(converged_folder,'U'))[:,:2]
        P = Ofpp.parse_internal_field(os.path.join(converged_folder,'p'))
        nut = Ofpp.parse_internal_field(os.path.join(converged_folder,'nut'))
        P = P.reshape(-1,1)
        nut = nut.reshape(-1,1)
        # print(U.shape,' ',P.shape)
        UVPnut = torch.from_numpy(np.concatenate([U,P,nut],axis=1))
        surf_magSf, surf_mid, norm = compute_normalOutward(case,cC,normal_source=True,magSf = True)
        _, _, norm_unnormalized = compute_normalOutward(case,cC,normal_source=True, unit_normal = False, magSf = True)
        n = []
        surf_nodeattr = []
        surf_nodexy = [] # Cf
        surf_edges = []
        surf_edgeattr = []
        surfcount = 0
        surf_UVP = []
        surf_normal_vectors_bd = []
        C = []
        indices_C = [] # For extracting P of adjacent cell-center
        n_unnorm = []
        # visited_set = set()
        fvattr_tm = 0
        for i,boolean in enumerate(_include):
            if (boolean):
                ipos = inverted_index[i] # reindexed position
                bool2 = mesh.is_cell_on_boundary(i,b'aerofoil')
                if bool2:
                    surf_nodexy.append([surf_mid[i][0],surf_mid[i][1]])
                    cell_next.append(ipos) # #cells next = #surfpoints
                    n.append([norm[i][0],norm[i][1]])
                    n_unnorm.append([norm_unnormalized[i][0],norm_unnormalized[i][1]])  
                    surf_edges.append([ipos,surfcount+count])
                    surf_edges.append([surfcount+count,ipos])
                    
                    s_tm = time() # measure_surface edge attribute compute-time
                    surf_nodeattr.append(0.0) # volume 0 at af-surface mid-points
                    surf_normal_vectors_bd.append([-norm_unnormalized[i][0],-norm_unnormalized[i][1]]) # in-ward normal (ipos -> surfcount+count)
                    surf_normal_vectors_bd.append([norm_unnormalized[i][0],norm_unnormalized[i][1]]) # out-ward normal (surfcount+count -> ipos)
                    surf_edgeattr.append(np.array([surf_magSf[i], surf_mid[i][0],surf_mid[i][1]])) 
                    surf_edgeattr.append(np.array([surf_magSf[i], surf_mid[i][0],surf_mid[i][1]])) 
                    fvattr_tm += (time()-s_tm) # measure_surface edge attribute compute-time
                    
                    surf_UVP.append([0.0,0.0,P[i][0],0.0])
                    surfcount+=1
                    C.append([cC[i,0],cC[i,1]])
                    indices_C.append(len(points))
                
                points.append([cC[i,0],cC[i,1]])
                if onlysurf:  continue
                
                s_tm = time()
                f_i = mesh.cell_faces[i]
                # print(i,' => ',f_i)
                nbr_i = mesh.cell_neighbour_cells(i)
                node_attr.append(V[i])
                for x in nbr_i:
                    if x>0:
                        if _include[x]:
                            # print(x,' => ',mesh.cell_faces[x])
                            common_face = set(mesh.cell_faces[x]).intersection(f_i)
                            assert(len(common_face)==1)
                            if len(common_face):
                                common_face = common_face.pop()
                                # if surfinfo:
                                #     owner.append(inverted_index[mesh.owner[common_face]])
                                #     neighbor.append(inverted_index[mesh.neighbour[common_face]])
                                #     sf.append(Sf[common_face])
                                # if (common_face not in visited_set):
                                #     # owner.append(inverted_index[mesh.owner[common_face]])
                                #     # neighbor.append(inverted_index[mesh.neighbour[common_face]])
                                #     # cf.append(Cf[common_face])
                                #     visited_set.add(common_face)
                                xpos = inverted_index[x]
                                if (inverted_index[mesh.owner[common_face]] == ipos) and (inverted_index[mesh.neighbour[common_face]]==xpos):
                                    owner_to_neighbor.append(True)
                                    sf.append(Sf[common_face])
                                    cf.append(Cf[common_face])
                                    surf_normal_vectors.append(Sf[common_face][:2])
                                else:
                                    owner_to_neighbor.append(False)
                                    # sf.append(-Sf[common_face])
                                    surf_normal_vectors.append(-Sf[common_face][:2])
                                    
                                try:
                                    edge_list.append([ipos,xpos])
                                    attrs = np.hstack((magSf[common_face], Cf[common_face][:2]))
                                    edge_attr.append(attrs)
                                except Exception as e:
                                    print('common_face: ',common_face)
                                    print(i,',',x,' => ',ipos,',',xpos)
                                    print(Cf[common_face])
                                    # print(magSf[common_face])
                                    raise(e)
                fvattr_tm += (time()-s_tm)
        if onlysurf:
            surf_data = Data(val = torch.from_numpy(np.concatenate([np.array(C),np.array(surf_nodexy),np.array(n_unnorm)],axis=1)),\
                             indices = torch.tensor(indices_C))
            return surf_data
        # elif surfinfo:
        #     internal_surf_flag = torch.tensor([True]*len(edge_list) + [False]*len(surf_edges))
        #     data = Data(internal_surf_flag = internal_surf_flag, owner = owner,neighbor = neighbor,cell_next = cell_next, sf = torch.from_numpy(np.array(sf)))
        #     return data
        else:
            # edge_attr = np.vstack(edge_attr)     
            # data = Data(x= torch.hstack((torch.tensor(points),u[_include,:],n[_include,:])), \
            #                 edge_index = torch.tensor(edge_list,dtype=torch.long).t().contiguous(),\
            #                node_attr=torch.tensor(node_attr),\
            #                edge_attr = torch.tensor(edge_attr),\
            #                surf = torch.tensor(surfbool),\
            #                y = UVPnut[_include]
            #                )
            edge_attr = np.vstack(edge_attr)  
            surf_edgeattr = np.vstack(surf_edgeattr)
            norm_internal = np.zeros_like(points)
            surfbool = np.array([False]*len(points) + [True]*len(surf_nodexy))
            n = np.concatenate([norm_internal,np.array(n)],axis = 0)
            points = np.concatenate([points,np.array(surf_nodexy)],axis = 0) 
            u = torch.tensor((np.array([np.cos(alpha), np.sin(alpha)])*Uinf).reshape(1, 2)*np.ones_like(points))
            internal_surf_flag = torch.tensor([True]*len(edge_list) + [False]*len(surf_edges))
            edge_list = edge_list + surf_edges 
            node_attr = node_attr + surf_nodeattr
            nodemarker = np.array(nodemarker + [0]*len(surf_nodexy),dtype = int) 
            edge_attr = np.concatenate([edge_attr , surf_edgeattr],axis=0)
            UVPnut_internal = np.concatenate([U,P,nut],axis=1)[_include,:]
            UVPnut = np.concatenate([UVPnut_internal,np.array(surf_UVP,dtype=float)],axis=0)
            # add bidrection edge and edge attributes
            data = Data(x= torch.hstack((torch.tensor(points),torch.tensor(u),torch.tensor(n))), \
                        edge_index = torch.tensor(edge_list,dtype=torch.long).t().contiguous(),\
                        node_attr=torch.tensor(node_attr),\
                        edge_attr = torch.tensor(edge_attr),\
                        surf = torch.tensor(surfbool),\
                        y = torch.tensor(UVPnut),
                        markers = torch.tensor(nodemarker)
                       )
            # data.internal_surf_flag = internal_surf_flag
            # data.owner = torch.tensor(owner)
            # data.neighbor = torch.tensor(neighbor)
            data.cell_next = torch.tensor(cell_next)
            data.sf = torch.from_numpy(np.array(sf))
            data.cf = torch.from_numpy(np.array(cf))
            data.owner_to_neighbor = torch.tensor(owner_to_neighbor+[False]*len(surf_edges)) # owner=>neighbor edges are True, all others are false including pseudoedges
            surf_normal_vectors = np.array(surf_normal_vectors)
            surf_normal_vectors_bd = np.array(surf_normal_vectors_bd)
            surface_normals = torch.tensor(np.concatenate([surf_normal_vectors,surf_normal_vectors_bd],axis = 0))
            pos = data.x[:,:2]
            s_tm = time()   
            edge_attr = torch.cat((surface_normals,(data.edge_attr[:,1:]-pos[data.edge_index[0,:],:]),(data.edge_attr[:,1:]-pos[data.edge_index[1,:],:])), dim=1)
            fvattr_tm += (time()-s_tm)
            data.edge_attr = edge_attr
            return fvattr_tm, data
    else:
        points = []
        for i,boolean in enumerate(_include):
            if (boolean):
                points.append([cC[i,0],cC[i,1]])
        U = Ofpp.parse_internal_field(os.path.join(converged_folder,'U'))[:,:2]
        P = Ofpp.parse_internal_field(os.path.join(converged_folder,'p'))
        nut = Ofpp.parse_internal_field(os.path.join(converged_folder,'nut'))
        P = P.reshape(-1,1)
        nut = nut.reshape(-1,1)
        # print(U.shape,' ',P.shape)
        UVPnut = torch.from_numpy(np.concatenate([U,P,nut],axis=1))
        data = Data(x= torch.tensor(points),y=UVPnut[_include])
    return None, data


# if __name__=='__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-d', '--device', help = 'dev', default = 'cuda:0', type = str)
#     parser.add_argument('-s', '--start', help = 'start', default = 0, type = int)
#     parser.add_argument('-e', '--end', help = 'end', default = 1000, type = int)
#     args = parser.parse_args()

#     #case = '/home1/OF_dataset/airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32'
#     vtkpath = 'Fine_vtk/'
#     cvtkpath = 'Coarse_vtk/'

#     theta_rot = 45
#     theta_seg = 90
#     dsdf_inf = 4.0
#     device = args.device

#     #task = 'scarce'
#     task = 'full'
#     raw_dir = '/ntuzfs/jessica/Airfrans/OF_dataset/'
#     craw_dir = 'AIRFRANS_coarse/'

#     start = args.start
#     end = args.end
#     # start = 1
#     # end = 2
#     data_fine = 'ccfine/'
#     os.system('mkdir -p '+data_fine)
#     #print(os.listdir(data_fine))
#     data_crs = 'cccoarse/'
#     os.system('mkdir -p '+data_crs)

#     manifest = loaddict_json('manifest.json')
#     manifest_all = (manifest[task + '_train'] + manifest['full_test'])
#     for i,casename in enumerate(tqdm(manifest_all[start:end])):
#         # COARSE
#         case = os.path.join(craw_dir,casename)
#         s = case.split('/')[-1]
#         if os.path.isfile(os.path.join(data_fine,s+'.pkl')):
#             continue

#         fname_c = os.path.join(data_crs,s+'.pkl')
#         if not os.path.isfile(fname_c): # Do not re-generate coarse pickle if it exists already
#             Uinf, alpha = float(s.split('_')[2]), float(s.split('_')[3])*np.pi/180
#             _iter = get_lastIteration(case,logfilename='log.sol')
#             preprocess(case,isfine = False)
#             mesh = Ofpp.FoamMesh(case)
#             cC = get_cellCentres(mesh,case)
#             internal_vtu = pv.read(cvtkpath+s+'/'+s+'_internal.vtu')
#             _include = np.array([False]*cC.shape[0])
#             _include[internal_vtu.cell_data['vtkOriginalCellIds']] = True
            
#             dt_coarse = get_graph(mesh,case,_include, None, cC=cC,V=None,magSf=None,Cf=None,num_nodes=cC.shape[0],\
#                     converged_folder = os.path.join(case,_iter), Uinf=None, alpha=None, isfine=False)
                
#             # Coarse data
#             torch.save(dt_coarse, fname_c)
#             # print('coarse: ',dt_coarse)
        
#         # FINE
#         case = os.path.join(raw_dir,casename)
#         s = case.split('/')[-1]
#         Uinf, alpha = float(s.split('_')[2]), float(s.split('_')[3])*np.pi/180
#         _iter = get_lastIteration(case,logfilename='log.simpleFoam')
#         preprocess(case,isfine = True)
#         mesh = Ofpp.FoamMesh(case)
#         cC = get_cellCentres(mesh,case)
#         V = get_cellVolume(mesh,os.path.join(case,_iter))
#         Cf = get_surfaceCentre(mesh,os.path.join(case,_iter))
#         magSf = get_surfaceArea(mesh,os.path.join(case,_iter))
        
#         # Load internal vtu
#         internal_vtu = pv.read(vtkpath+s+'/'+s+'_internal.vtu')

#         _include = np.array([False]*cC.shape[0])
#         _include[internal_vtu.cell_data['vtkOriginalCellIds']] = True
#         _,bd_edges = get_edges(internal_vtu)
#         markers = compute_Cellmarkers_numpy(internal_vtu,bd_edges)
#         farfieldmarker = np.array([False]*cC.shape[0])
#         farfieldmarker[internal_vtu.cell_data['vtkOriginalCellIds']] = markers
        
#         # Fine data
#         dt_fine = get_graph(mesh,case,_include, farfieldmarker, cC,V,magSf,Cf,num_nodes=V.shape[0],\
#                 converged_folder = os.path.join(case,_iter),Uinf=Uinf, alpha=alpha, isfine = True)
#         dt_fine.uinf = Uinf
#         dt_fine.alpha = alpha
#         dt_fine.sdf = GcomputeSDF(dt_fine.x[:,:2],dt_fine.surf)
#         dt_fine.saf = GcomputeSAF2(pos = dt_fine.x[:,:2].to(device), surf_bool = dt_fine.surf.to(device)).cpu()
#         try:
#             dt_fine.dsdf = getDSDF(pos=dt_fine.x[:,:2].to(device),bd=dt_fine.surf.to(device),theta_rot=radians(theta_rot),theta_seg=radians(theta_seg),inf=dsdf_inf).t().cpu()
#         except Exception as e:
#             raise(e)
#         finally:
#             torch.cuda.empty_cache()
#             continue
#         dt_fine.coarse_path = fname_c
#         fname = os.path.join(data_fine,s+'.pkl')
#         torch.save(dt_fine, fname)

#--used for rebuttal--
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', help = 'dev', default = 'cuda:0', type = str)
    parser.add_argument('-s', '--start', help = 'start', default = 0, type = int)
    parser.add_argument('-e', '--end', help = 'end', default = 1000, type = int)
    args = parser.parse_args()

    #case = '/home1/OF_dataset/airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32'
    vtkpath = 'Fine_vtk/'
    cvtkpath = 'Coarse_vtk/'

    theta_rot = 45
    theta_seg = 90
    dsdf_inf = 4.0
    device = args.device

    #task = 'scarce'
    task = 'full'
    raw = '/home1/' #'OF_dataset/'
    # raw_dir = os.path.join(raw,'OF_dataset/')
    raw_dir = os.path.join(os.getcwd(),'AIRFRANS_coarse/') #coarse

    start = args.start
    end = args.end
    # start = 1
    # end = 2
    data_fine = 'ccfine_ns/'
    os.system('mkdir -p '+data_fine)
    #print(os.listdir(data_fine))
    data_crs = 'cccoarse_ns/'
    os.system('mkdir -p '+data_crs)

    manifest = loaddict_json('manifest.json')
    manifest_all = (manifest[task + '_train'] + manifest['full_test'])
    # casename = manifest_all[start:end][0]
    for i,casename in enumerate(tqdm(manifest_all[start:end], desc = device)):
        case = os.path.join(raw_dir,casename)
        s = case.split('/')[-1]
        Uinf, alpha = float(s.split('_')[2]), float(s.split('_')[3])*np.pi/180
        # _iter = get_lastIteration(case,logfilename='log.simpleFoam')
        _iter = get_lastIteration(case) #coarse
        preprocess(case,isfine = True)
        mesh = Ofpp.FoamMesh(case)
        cC = get_cellCentres(mesh,case)
        V = get_cellVolume(mesh,os.path.join(case,_iter))
        Cf = get_surfaceCentre(mesh,os.path.join(case,_iter))
        Sf = get_surfaceNormal(mesh,os.path.join(case,_iter))
        magSf = get_surfaceArea(mesh,os.path.join(case,_iter))
            
        internal_vtu = pv.read(vtkpath+s+'/'+s+'_internal.vtu')

        _include = np.array([False]*cC.shape[0])
        _include[internal_vtu.cell_data['vtkOriginalCellIds']] = True
        _,bd_edges = get_edges(internal_vtu)
        markers = compute_Cellmarkers_numpy(internal_vtu,bd_edges)
        farfieldmarker = np.array([False]*cC.shape[0])
        farfieldmarker[internal_vtu.cell_data['vtkOriginalCellIds']] = markers
            
        # Fine data
        ccgraph_time , dt_fine = get_graph_ns(mesh,case,_include, farfieldmarker, cC,V,magSf,Cf,Sf,num_nodes=V.shape[0],\
                    converged_folder = os.path.join(case,_iter),Uinf=Uinf, alpha=alpha, isfine = True)      
        # print(ccgraph_time)
        with open('cc_graph_LRmesh.txt','w') as f: #coarse
        # with open('cc_graph_finemesh.txt','w') as f:
            s_tm = time()
            dt_fine.saf = GcomputeSAF2(pos = dt_fine.x[:,:2].to(device), surf_bool = dt_fine.surf.to(device)).cpu()
            fv_time = time() - s_tm
            s_tm = time()
            try:
                dt_fine.dsdf = getDSDF(pos=dt_fine.x[:,:2].to(device),bd=dt_fine.surf.to(device),theta_rot=radians(theta_rot),theta_seg=radians(theta_seg),inf=dsdf_inf).t().cpu()
            except Exception as e:
                raise(e)
            did_time = time() - s_tm
            f.write(str(len(dt_fine.x))+' '+str(fv_time)+' '+str(did_time)+' '+str(ccgraph_time)+'\n')

#------------- used for data generation ------------
# if __name__=='__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-d', '--device', help = 'dev', default = 'cuda:0', type = str)
#     parser.add_argument('-s', '--start', help = 'start', default = 0, type = int)
#     parser.add_argument('-e', '--end', help = 'end', default = 1000, type = int)
#     args = parser.parse_args()

#     #case = '/home1/OF_dataset/airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32'
#     vtkpath = 'Fine_vtk/'
#     cvtkpath = 'Coarse_vtk/'

#     theta_rot = 45
#     theta_seg = 90
#     dsdf_inf = 4.0
#     device = args.device

#     #task = 'scarce'
#     task = 'full'
#     raw = '/home1/' #'OF_dataset/'
#     raw_dir = os.path.join(raw,'OF_dataset/')
#     craw_dir = os.path.join(os.getcwd(),'AIRFRANS_coarse/')

#     start = args.start
#     end = args.end
#     # start = 1
#     # end = 2
#     data_fine = 'ccfine_ns/'
#     os.system('mkdir -p '+data_fine)
#     #print(os.listdir(data_fine))
#     data_crs = 'cccoarse_ns/'
#     os.system('mkdir -p '+data_crs)

#     manifest = loaddict_json('manifest.json')
#     manifest_all = (manifest[task + '_train'] + manifest['full_test'])
#     for i,casename in enumerate(tqdm(manifest_all[start:end], desc = device)):
#         # COARSE
#         case = os.path.join(craw_dir,casename)
#         s = case.split('/')[-1]
#         #if os.path.isfile(os.path.join(data_fine,s+'.pkl')):
#         #    continue

#         fname_c = os.path.join(data_crs,s+'.pkl')
#         if not os.path.isfile(fname_c): # Do not re-generate coarse pickle if it exists already
#             Uinf, alpha = float(s.split('_')[2]), float(s.split('_')[3])*np.pi/180
#             _iter = get_lastIteration(case,logfilename='log.sol')
#             preprocess(case,isfine = False)
#             mesh = Ofpp.FoamMesh(case)
#             cC = get_cellCentres(mesh,case)
#             internal_vtu = pv.read(cvtkpath+s+'/'+s+'_internal.vtu')
#             _include = np.array([False]*cC.shape[0])
#             _include[internal_vtu.cell_data['vtkOriginalCellIds']] = True
            
#             dt_coarse = get_graph_ns(mesh,case,_include, None, cC=cC,V=None,magSf=None,Cf=None,Sf = None, num_nodes=cC.shape[0],\
#                     converged_folder = os.path.join(case,_iter), Uinf=None, alpha=None, isfine=False)
                
#             # Coarse data
#             #torch.save(dt_coarse, fname_c)
#             # print('coarse: ',dt_coarse)
        
#         # FINE
#         case = os.path.join(raw_dir,casename)
#         s = case.split('/')[-1]
#         Uinf, alpha = float(s.split('_')[2]), float(s.split('_')[3])*np.pi/180
#         _iter = get_lastIteration(case,logfilename='log.simpleFoam')
#         preprocess(case,isfine = True)
#         mesh = Ofpp.FoamMesh(case)
#         cC = get_cellCentres(mesh,case)
#         V = get_cellVolume(mesh,os.path.join(case,_iter))
#         Cf = get_surfaceCentre(mesh,os.path.join(case,_iter))
#         Sf = get_surfaceNormal(mesh,os.path.join(case,_iter))
#         magSf = get_surfaceArea(mesh,os.path.join(case,_iter))
#         # print(case)
#         # print('Sf.shape: ',Sf.shape)
#         # print('Cf.shape: ',Cf.shape)
#         # Load internal vtu
#         internal_vtu = pv.read(vtkpath+s+'/'+s+'_internal.vtu')

#         _include = np.array([False]*cC.shape[0])
#         _include[internal_vtu.cell_data['vtkOriginalCellIds']] = True
#         _,bd_edges = get_edges(internal_vtu)
#         markers = compute_Cellmarkers_numpy(internal_vtu,bd_edges)
#         farfieldmarker = np.array([False]*cC.shape[0])
#         farfieldmarker[internal_vtu.cell_data['vtkOriginalCellIds']] = markers
        
#         # Fine data
#         ccgraph_time , dt_fine = get_graph_ns(mesh,case,_include, farfieldmarker, cC,V,magSf,Cf,Sf,num_nodes=V.shape[0],\
#                 converged_folder = os.path.join(case,_iter),Uinf=Uinf, alpha=alpha, isfine = True)
#         dt_fine.uinf = Uinf
#         dt_fine.alpha = alpha
        
#         dt_fine.sdf = GcomputeSDF(dt_fine.x[:,:2],dt_fine.surf)
#         s_tm = time()
#         dt_fine.saf = GcomputeSAF2(pos = dt_fine.x[:,:2].to(device), surf_bool = dt_fine.surf.to(device)).cpu()
#         fv_time = time() - s_tm
#         s_tm = time()
#         try:
#             dt_fine.dsdf = getDSDF(pos=dt_fine.x[:,:2].to(device),bd=dt_fine.surf.to(device),theta_rot=radians(theta_rot),theta_seg=radians(theta_seg),inf=dsdf_inf).t().cpu()
#         except Exception as e:
#             raise(e)
#         did_time = time() - s_tm
#         with open('fine_tm.txt','a') as f:
#             f.write(str(fv_time)+' '+str(did_time)+' '+str(ccgraph_time)+'\n')
#         # finally:
#         #     torch.cuda.empty_cache()
#         #     continue
#         dt_fine.coarse_path = fname_c
#         fname = os.path.join(data_fine,s+'.pkl')
#         #torch.save(dt_fine, fname)
#         torch.cuda.empty_cache()
        
