import os
import pyvista as pv
import numpy as np
import pickle
from torch_geometric.data import Data
import torch
from math import radians
from time import time
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--case', help = 'Openfoam case path', default = '/home1/OF_dataset/airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32', type = str)
# parser.add_argument('-s', '--savedir', help = 'output directory', default = 'Coarse_vtk/', type = str)

# args = parser.parse_args()

def getfoamfile(casepath):
# casepath = '/home1/OF_dataset/airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32'
    curpath = os.getcwd()
    os.chdir(casepath)
    os.system('paraFoam -touch-all')
    take = []
    for f in os.listdir(casepath):
        if f.endswith('.foam'):
            take.append(f)
            break
    os.chdir(curpath)
    return os.path.join(casepath,take[0])

def getMesh(casepath, slice = True):
    foam = getfoamfile(casepath)
    # print(foam)
    reader = pv.POpenFOAMReader(foam)
    # print(f"All patch names: {reader.patch_array_names}")
    # print(f"All patch status: {reader.all_patch_arrays_status}")
    # print(f"Available Time Values: {reader.time_values}")
    reader.set_active_time_value(sorted(reader.time_values)[-1])
    reader.cell_to_point_creation = True  # Need point data for streamlines
    mesh = reader.read()
    return mesh

def slice_z_center(mesh):
    """Slice mesh through center in z normal direction, move to z=0."""
    slice_mesh = mesh.slice(normal='z')
    return slice_mesh

def crop_internal(internal_mesh, crop =  [-2, 4, -1.5, 1.5, 0.5, 0.5]):
    bounds = tuple(crop)
    cropped_internal = internal_mesh.clip_box(bounds = bounds, invert = False, crinkle = True)
    return cropped_internal

def get_airfoil_vtp(boundary,compute_length = True):
    aerofoil_wnorm = boundary['aerofoil'].compute_normals(cell_normals=True, point_normals=True, inplace=True)
    aerofoil_wnorm = slice_z_center(aerofoil_wnorm)
    if compute_length:
        aerofoil_wnorm = aerofoil_wnorm.compute_cell_sizes(length=True, area=False, volume=False)
#     print(aerofoil_wnorm.points)
#     aerofoil_wnorm.point_data['Normals']
    return aerofoil_wnorm

def get_farfield_indices(internal_vtu):
    feat_edges = internal_vtu.extract_feature_edges()
    selected = feat_edges.point_data['vtkOriginalPointIds']
    feat_pts_c = internal_vtu.points[selected,:]
    _filter = ~( (feat_pts_c[:,0]>= -1) & (feat_pts_c[:,0]<=2) & (feat_pts_c[:,1]>= -1) & (feat_pts_c[:,1]<= 1) )
    subselected = selected[_filter]
    return subselected

def save_vtpvtu(casepath,savepath):
    if not os.path.exists(savepath):
        os.system('mkdir -p '+savepath)
    assert os.path.exists(casepath)

    mesh = getMesh(casepath)
    # print(f"Mesh patches: {mesh.keys()}")
    internal_mesh = mesh["internalMesh"]
    boundary = mesh["boundary"]
    internal_mesh.compute_implicit_distance(boundary['aerofoil'], inplace=True) # Compute SDF
    # Remove unnecessary arrays from internal.vtp
    keep0 = ['nut', 'p', 'U', 'implicit_distance', 'vtkOriginalPointIds']
    keep1 = ['nut', 'p', 'U', 'cell_ids', 'vtkOriginalCellIds']
    for key in internal_mesh.point_data.keys():
        if key not in keep0:
            internal_mesh.point_data.remove(key)
    for key in internal_mesh.cell_data.keys():
        if key not in keep1:
            internal_mesh.cell_data.remove(key)
    internal_vtu = crop_internal(slice_z_center(internal_mesh))

    aerofoil_vtp = get_airfoil_vtp(boundary,compute_length = True)
    keep0 = ['nut', 'p', 'U', 'Normals']
    keep1 = ['nut', 'p', 'U', 'Normals', 'Length']
    for key in aerofoil_vtp.point_data.keys():
        if key not in keep0:
            aerofoil_vtp.point_data.remove(key)
    for key in aerofoil_vtp.cell_data.keys():
        if key not in keep1:
            aerofoil_vtp.cell_data.remove(key)

    vtpname = casepath.split('/')[-1]+"_aerofoil.vtp"
    vtuname = casepath.split('/')[-1]+"_internal.vtu"
    subdir = os.path.join(savepath,casepath.split('/')[-1])
    if not os.path.exists(subdir):
        os.system('mkdir -p '+subdir)
    aerofoil_vtp.save(os.path.join(subdir,vtpname))
    internal_vtu.save(os.path.join(subdir,vtuname))
    # print('internal =>')
    # print(internal_vtu)
    # print(internal_vtu.point_data)
    # print(internal_vtu.cell_data)
    # print('aerofoil =>')
    # print(aerofoil_vtp)
    # print(aerofoil_vtp.point_data)
    # print(aerofoil_vtp.cell_data)

def get_points_from_edges(edges):
    return edges.point_data['vtkOriginalPointIds']

def get_edges(internal_vtu):
    internal_edges = internal_vtu.extract_feature_edges(boundary_edges=False, feature_edges=False, manifold_edges=True)
    boundary_edges = internal_vtu.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
    return (internal_edges,boundary_edges)

def get_edges_numpy(edges):
    return np.asarray(edges.lines).reshape((-1, 3))[:, 1:]

def renumber_edges(edges):
    """ Map the node-ids in the edges from their internal-id after edge-extraction to original point-ids """
    _m = {}
    _new_edges = []
    for i,orig_id in enumerate(edges.point_data['vtkOriginalPointIds']):
        _m[i] = orig_id

    for e in get_edges_numpy(edges):
        _new_edges.append([_m[e[0]], _m[e[1]]])
    return _new_edges

def compute_edge_indices(internal_vtu):
    """ edge indices respects the ordering of points in internal_vtu.points"""
    edges,bd_edges = get_edges(internal_vtu)
    edges_list = renumber_edges(edges)
    bd_edges_list = renumber_edges(bd_edges)
    edge_indices = edges_list + bd_edges_list
    return edge_indices

def compute_markers_numpy(internal_vtu,bd_edges):
    """ return markers """
    marker = np.array([-1]*internal_vtu.points.shape[0],dtype = int) # mark all points as -1
    marker[bd_edges.point_data['vtkOriginalPointIds']] = 1 # mark aerofoil + farfield to 1
    marker[internal_vtu.point_data['U'][:, 0]==0] = 0 # Mark only aerofoil to 0
    return marker

def save_node_markers(casepath,savepath):
    if not os.path.exists(savepath):
        os.system('mkdir -p '+savepath)
    assert os.path.exists(casepath)
    vtuname = casepath.split('/')[-1]+"_internal.vtu"
    subdir = os.path.join(savepath,casepath.split('/')[-1])
    internal_vtu = pv.read(os.path.join(subdir,vtuname))
    _,bd_edges = get_edges(internal_vtu)
    marker = compute_markers_numpy(internal_vtu,bd_edges)
    # with open(os.path.join(subdir,'_markers.pkl'),'wb') as f:
    #     pickle.dump(marker,f)
    save_name = os.path.join(subdir,'markers.npy')
    np.save(save_name, marker) # save
    with open('N_fine_numnodes.txt','a') as f:
        num_nodes = len(marker)
        f.write(str(num_nodes)+"\n")

def save_edge_indices(casepath,savepath):
    if not os.path.exists(savepath):
        os.system('mkdir -p '+savepath)
    assert os.path.exists(casepath)
    vtuname = casepath.split('/')[-1]+"_internal.vtu"
    subdir = os.path.join(savepath,casepath.split('/')[-1])
    internal_vtu = pv.read(os.path.join(subdir,vtuname))
    edge_indices = np.array(compute_edge_indices(internal_vtu))
    save_name = os.path.join(subdir,'edges.npy')
    np.save(save_name, edge_indices)

def save_coarse_data(casepath, savepath):
    if not os.path.exists(savepath):
        os.system('mkdir -p '+savepath)
    assert os.path.exists(casepath)
    vtuname = casepath.split('/')[-1]+"_internal.vtu"
    subdir = os.path.join(savepath,casepath.split('/')[-1])
    internal = pv.read(os.path.join(subdir,vtuname))
    coarse_y = np.concatenate([internal.point_data['U'][:, :2], internal.point_data['p'][:, None], internal.point_data['nut'][:, None]], \
                axis = -1)
    coarse_x = internal.points[:,:2]
    #print(torch.from_numpy(coarse_x).type())
    coarse_data = Data(x = torch.from_numpy(coarse_x).double(), y = torch.from_numpy(coarse_y).double())
    # print(coarse_data.x.type(),coarse_data.y.type(),' ',coarse_data.x[0].type())
    save_name = os.path.join(subdir,'data.pt')
    torch.save(coarse_data,save_name)
    with open('N_coarse_numnodes.txt','a') as f:
        num_nodes = len(coarse_data.x)
        f.write(str(num_nodes)+"\n")
    
def GcomputeSAF2(pos,surf_bool): # Assume pos, surf_bool are torch tensors
    """ New SAF computation """
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

def save_SDFSAFDSDF_data(casepath, savepath, device = 'cuda:0'):
    if not os.path.exists(savepath):
        os.system('mkdir -p '+savepath)
    assert os.path.exists(casepath)
    vtuname = casepath.split('/')[-1]+"_internal.vtu"
    subdir = os.path.join(savepath,casepath.split('/')[-1])
    internal = pv.read(os.path.join(subdir,vtuname))
    sdf = -internal.point_data['implicit_distance'][:, None]
    pos = torch.tensor(internal.points[:,:2]).to(device)
    surf_bool = (internal.point_data['U'][:, 0] == 0)
    s_tm = time()
    saf = GcomputeSAF2(pos,surf_bool=surf_bool).cpu()
    fv_time = time() - s_tm
    theta_rot = 45
    theta_seg = 90
    dsdf_inf = 4.0
    s_tm = time()
    dsdf = getDSDF(pos, surf_bool, theta_rot=radians(theta_rot),theta_seg=radians(theta_seg),inf=dsdf_inf).t().cpu()
    did_time =time()-s_tm
    geomfeats = Data(sdf = sdf, saf = saf, dsdf = dsdf)
    save_name = os.path.join(subdir,'geom.ft')
    torch.save(geomfeats,save_name)
    with open('coarse_tm.txt','a') as f:
        f.write(str(fv_time)+" "+str(did_time)+"\n")

def plot_edges(internal_vtu,edges,bd_edges):
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    
    points = internal_vtu.points[:,:2]
    edges = np.array(edges)
    lc = LineCollection(points[edges])
    fig = plt.figure(figsize=(14,6))
    plt.gca().add_collection(lc)
    
    points = internal_vtu.points[:,:2]
    edges = np.array(bd_edges)
    lc = LineCollection(points[edges],color='r')
    plt.gca().add_collection(lc)
    
    # plt.xlim((-0.15,1))
    # plt.ylim((-.1,.1))
    # plt.xlim((3,4.4))
    # plt.ylim((0,points[:,1].max()))
    plt.xlim(points[:,0].min(), points[:,0].max())
    plt.ylim(points[:,1].min(), points[:,1].max())
    plt.show()

# if __name__ == '__main__':
#     casepath = args.case
#     savepath = args.savedir
#     save_vtpvtu(casepath=casepath,savepath=savepath)