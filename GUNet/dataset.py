import numpy as np
import pyvista as pv
from reorganize import reorganize
from math import *
import torch
from torch_geometric.data import Data
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import tqdm
import os 

def cell_sampling_2d(cell_points, cell_attr = None):
    '''
    Sample points in a two dimensional cell via parallelogram sampling and triangle interpolation via barycentric coordinates. The vertices have to be ordered in a certain way.

    Args:
        cell_points (array): Vertices of the 2 dimensional cells. Shape (N, 4) for N cells with 4 vertices.
        cell_attr (array, optional): Features of the vertices of the 2 dimensional cells. Shape (N, 4, k) for N cells with 4 edges and k features. 
            If given shape (N, 4) it will resize it automatically in a (N, 4, 1) array. Default: ``None``
    '''
    # Sampling via triangulation of the cell and parallelogram sampling
    v0, v1 = cell_points[:, 1] - cell_points[:, 0], cell_points[:, 3] - cell_points[:, 0]
    v2, v3 = cell_points[:, 3] - cell_points[:, 2], cell_points[:, 1] - cell_points[:, 2]  
    a0, a1 = np.abs(np.linalg.det(np.hstack([v0[:, :2], v1[:, :2]]).reshape(-1, 2, 2))), np.abs(np.linalg.det(np.hstack([v2[:, :2], v3[:, :2]]).reshape(-1, 2, 2)))
    p = a0/(a0 + a1)
    index_triangle = np.random.binomial(1, p)[:, None]
    u = np.random.uniform(size = (len(p), 2))
    sampled_point = index_triangle*(u[:, 0:1]*v0 + u[:, 1:2]*v1) + (1 - index_triangle)*(u[:, 0:1]*v2 + u[:, 1:2]*v3)
    sampled_point_mirror = index_triangle*((1 - u[:, 0:1])*v0 + (1 - u[:, 1:2])*v1) + (1 - index_triangle)*((1 - u[:, 0:1])*v2 + (1 - u[:, 1:2])*v3)
    reflex = (u.sum(axis = 1) > 1)
    sampled_point[reflex] = sampled_point_mirror[reflex]

    # Interpolation on a triangle via barycentric coordinates
    if cell_attr is not None:
        t0, t1, t2 = np.zeros_like(v0), index_triangle*v0 + (1 - index_triangle)*v2, index_triangle*v1 + (1 - index_triangle)*v3
        w = (t1[:, 1] - t2[:, 1])*(t0[:, 0] - t2[:, 0]) + (t2[:, 0] - t1[:, 0])*(t0[:, 1] - t2[:, 1])
        w0 = (t1[:, 1] - t2[:, 1])*(sampled_point[:, 0] - t2[:, 0]) + (t2[:, 0] - t1[:, 0])*(sampled_point[:, 1] - t2[:, 1])
        w1 = (t2[:, 1] - t0[:, 1])*(sampled_point[:, 0] - t2[:, 0]) + (t0[:, 0] - t2[:, 0])*(sampled_point[:, 1] - t2[:, 1])
        w0, w1 = w0/w, w1/w
        w2 = 1 - w0 - w1
        
        if len(cell_attr.shape) == 2:
            cell_attr = cell_attr[:, :, None]
        attr0 = index_triangle*cell_attr[:, 0] + (1 - index_triangle)*cell_attr[:, 2]
        attr1 = index_triangle*cell_attr[:, 1] + (1 - index_triangle)*cell_attr[:, 1]
        attr2 = index_triangle*cell_attr[:, 3] + (1 - index_triangle)*cell_attr[:, 3]
        sampled_attr = w0[:, None]*attr0 + w1[:, None]*attr1 + w2[:, None]*attr2

    sampled_point += index_triangle*cell_points[:, 0] + (1 - index_triangle)*cell_points[:, 2]    

    return np.hstack([sampled_point[:, :2], sampled_attr]) if cell_attr is not None else sampled_point[:, :2]

def cell_sampling_1d(line_points, line_attr = None):
    '''
    Sample points in a one dimensional cell via linear sampling and interpolation.

    Args:
        line_points (array): Edges of the 1 dimensional cells. Shape (N, 2) for N cells with 2 edges.
        line_attr (array, optional): Features of the edges of the 1 dimensional cells. Shape (N, 2, k) for N cells with 2 edges and k features.
            If given shape (N, 2) it will resize it automatically in a (N, 2, 1) array. Default: ``None``
    '''
    # Linear sampling
    u = np.random.uniform(size = (len(line_points), 1))
    sampled_point = u*line_points[:, 0] + (1 - u)*line_points[:, 1]

    # Linear interpolation
    if line_attr is not None:   
        if len(line_attr.shape) == 2:
            line_attr = line_attr[:, :, None]
        sampled_attr = u*line_attr[:, 0] + (1 - u)*line_attr[:, 1]

    return np.hstack([sampled_point[:, :2], sampled_attr]) if line_attr is not None else sampled_point[:, :2]

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

def computeSAF(pos,surf_bool,aerofoil):
    # def angle_trunc(a):
    #     while (a < 0.0).any():
    #         a[a<0.0] += (pi * 2)
    #     return a
    def getAngleBetweenPoints(xy_orig, xy_landmark):
        """ Returns angle in radians """
        x_landmark,y_landmark = xy_landmark[:,0],xy_landmark[:,1]
        x_orig,y_orig = xy_orig[:,0],xy_orig[:,1]
        deltaY = y_landmark - y_orig
        deltaX = x_landmark - x_orig
        # return angle_trunc(np.arctan2(deltaY, deltaX)) # Do not use this
        return np.arctan2(deltaY,deltaX)

    def get_closest_points(internalcells, boundarycells):
        dist = cdist(boundarycells,internalcells)
        closests = np.argmin(dist,axis=0)
        return closests, boundarycells[closests]
        
    # closest_point= aerofoil.find_closest_cell(pos)
    closest_point,closest_points_xyz = get_closest_points(pos,pos[surf_bool])
    # print(closest_point)
    # print('shape: ',closest_point.shape,' aerofoil points shape: ',aerofoil.points.shape)
    # closest_points_xyz = aerofoil.points[closest_point]
    # print('closest_points_xyz: ',closest_points_xyz.shape)
    saf_tmp = getAngleBetweenPoints(closest_points_xyz[:,:2],pos[:,:2]) # Angle in radian
    saf_tmp = (saf_tmp)/1.8128
    return saf_tmp

def GcomputeSAF(pos,surf_bool): # Assume pos, surf_bool are torch tensors
    # def angle_trunc(a):
    #     while (a < 0.0).any():
    #         a[a<0.0] += (pi * 2)
    #     return a
    def getAngleBetweenPoints(xy_orig, xy_landmark):
        """ Returns angle in radians """
        x_landmark,y_landmark = xy_landmark[:,0],xy_landmark[:,1]
        x_orig,y_orig = xy_orig[:,0],xy_orig[:,1]
        deltaY = y_landmark - y_orig
        deltaX = x_landmark - x_orig
        return torch.atan2(deltaY,deltaX)

    def get_closest_points(internalcells, boundarycells):
        dist = torch.cdist(boundarycells,internalcells,p=2)
        # print('dist size: ',dist.size())
        closests = torch.argmin(dist,dim=0)
        return closests, boundarycells[closests]
        
    # closest_point= aerofoil.find_closest_cell(pos)
    closest_point,closest_points_xyz = get_closest_points(pos,pos[surf_bool])
    # print(closest_point)
    # print('shape: ',closest_point.shape,' aerofoil points shape: ',aerofoil.points.shape)
    # closest_points_xyz = aerofoil.points[closest_point]
    # print('closest_points_xyz: ',closest_points_xyz.shape)
    saf_tmp = getAngleBetweenPoints(closest_points_xyz[:,:2],pos[:,:2]) # Angle in radian
    # saf_tmp = (saf_tmp)/1.8128 # normalization => subtract mean([-pi,pi]), divide by std([-pi,pi])
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
  

def Dataset(set, norm = False, coef_norm = None, crop = None, sample = None, n_boot = int(5e5), surf_ratio = .1, \
    use_saf=False,use_dsdf=False, manifest_dsdf=None,manifest_saf=None):
    '''
    Create a list of simulation to input in a PyTorch Geometric DataLoader. Simulation are transformed by keeping vertices of the CFD mesh or 
    by sampling (uniformly or via the mesh density) points in the simulation cells.

    Args:
        set (list): List of geometry names to include in the dataset.
        norm (bool, optional): If norm is set to ``True``, the mean and the standard deviation of the dataset will be computed and returned. 
            Moreover, the dataset will be normalized by these quantities. Ignored when ``coef_norm`` is not None. Default: ``False``
        coef_norm (tuple, optional): This has to be a tuple of the form (mean input, std input, mean output, std ouput) if not None. 
            The dataset generated will be normalized by those quantites. Default: ``None``
        crop (list, optional): List of the vertices of the rectangular [xmin, xmax, ymin, ymax] box to crop simulations. Default: ``None``
        sample (string, optional): Type of sampling. If ``None``, no sampling strategy is applied and the nodes of the CFD mesh are returned.
            If ``uniform`` or ``mesh`` is chosen, uniform or mesh density sampling is applied on the domain. Default: ``None``
        n_boot (int, optional): Used only if sample is not None, gives the size of the sampling for each simulation. Defaul: ``int(5e5)``
        surf_ratio (float, optional): Used only if sample is not None, gives the ratio of point over the airfoil to sample with respect to point
            in the volume. Default: ``0.1``
    '''
    if norm and coef_norm is not None:
        raise ValueError('If coef_norm is not None and norm is True, the normalization will be done via coef_norm')

    dataset = []

    for k, s in enumerate(tqdm(set,desc = 'loading GUNet dataset')):
        # Get the 3D mesh, add the signed distance function and slice it to return in 2D
        internal = pv.read('Dataset/' + s + '/' + s + '_internal.vtu')
        aerofoil = pv.read('Dataset/' + s + '/' + s + '_aerofoil.vtp')
        internal = internal.compute_cell_sizes(length = False, volume = False)
        # Cropping if needed, crinkle is True.
        if crop is not None:
            bounds = (crop[0], crop[1], crop[2], crop[3], 0, 1)
            internal = internal.clip_box(bounds = bounds, invert = False, crinkle = True)

        # If sampling strategy is chosen, it will sample points in the cells of the simulation instead of directly taking the nodes of the mesh.
        if sample is not None:
            # Sample on a new point cloud
            if sample == 'uniform': # Uniform sampling strategy
                p = internal.cell_data['Area']/internal.cell_data['Area'].sum()
                sampled_cell_indices = np.random.choice(internal.n_cells, size = n_boot, p = p)
                surf_p = aerofoil.cell_data['Length']/aerofoil.cell_data['Length'].sum()
                sampled_line_indices = np.random.choice(aerofoil.n_cells, size = int(n_boot*surf_ratio), p = surf_p)
            elif sample == 'mesh': # Sample via mesh density
                sampled_cell_indices = np.random.choice(internal.n_cells, size = n_boot)
                sampled_line_indices = np.random.choice(aerofoil.n_cells, size = int(n_boot*surf_ratio))

            cell_dict = internal.cells.reshape(-1, 5)[sampled_cell_indices, 1:]
            cell_points = internal.points[cell_dict]            
            line_dict = aerofoil.lines.reshape(-1, 3)[sampled_line_indices, 1:]
            line_points = aerofoil.points[line_dict]

            # Geometry information
            geom = -internal.point_data['implicit_distance'][cell_dict, None] # Signed distance function
            Uinf, alpha = float(s.split('_')[2]), float(s.split('_')[3])*np.pi/180
            # u = (np.array([np.cos(alpha), np.sin(alpha)])*Uinf).reshape(1, 2)*(internal.point_data['U'][cell_dict, :1] != 0)
            u = (np.array([np.cos(alpha), np.sin(alpha)])*Uinf).reshape(1, 2)*np.ones_like(internal.point_data['U'][cell_dict, :1])
            normal = np.zeros_like(u)

            surf_geom = np.zeros_like(aerofoil.point_data['U'][line_dict, :1])
            # surf_u = np.zeros_like(aerofoil.point_data['U'][line_dict, :2])
            surf_u = (np.array([np.cos(alpha), np.sin(alpha)])*Uinf).reshape(1, 2)*np.ones_like(aerofoil.point_data['U'][line_dict, :1])
            surf_normal = -aerofoil.point_data['Normals'][line_dict, :2]

            attr = np.concatenate([u, geom, normal, internal.point_data['U'][cell_dict, :2], 
                internal.point_data['p'][cell_dict, None], internal.point_data['nut'][cell_dict, None]], axis = -1)
            surf_attr = np.concatenate([surf_u, surf_geom, surf_normal, aerofoil.point_data['U'][line_dict, :2], 
                aerofoil.point_data['p'][line_dict, None], aerofoil.point_data['nut'][line_dict, None]], axis = -1)
            sampled_points = cell_sampling_2d(cell_points, attr)
            surf_sampled_points = cell_sampling_1d(line_points, surf_attr)

            # Define the inputs and the targets
            pos = sampled_points[:, :2]
            init = sampled_points[:, :7]
            target = sampled_points[:, 7:]
            surf_pos = surf_sampled_points[:, :2]
            surf_init = surf_sampled_points[:, :7]
            surf_target = surf_sampled_points[:, 7:]

            # if cell_centers:
            #     centers = internal.ptc().cell_centers()
            #     surf_centers = aerofoil.cell_centers()

            #     geom = -centers.cell_data['implicit_distance'][:, None] # Signed distance function
            #     Uinf, alpha = float(s.split('_')[2]), float(s.split('_')[3])*np.pi/180
            #     u = (np.array([np.cos(alpha), np.sin(alpha)])*Uinf).reshape(1, 2)*np.ones_like(internal.cell_data['U'][:, :1])
            #     normal = np.zeros_like(u)

            #     surf_geom = np.zeros_like(surf_centers.cell_data['U'][:, :1])
            #     # surf_u = np.zeros_like(surf_centers.cell_data['U'][:, :2])
            #     surf_u = (np.array([np.cos(alpha), np.sin(alpha)])*Uinf).reshape(1, 2)*np.ones_like(surf_centers.cell_data['U'][:, :1])
            #     surf_normal = -aerofoil.cell_data['Normals'][:, :2]

            #     attr = np.concatenate([u, geom, normal,
            #         internal.cell_data['U'][:, :2], internal.cell_data['p'][:, None], internal.cell_data['nut'][:, None]], axis = -1)
            #     surf_attr = np.concatenate([surf_u, surf_geom, surf_normal,
            #         aerofoil.cell_data['U'][:, :2], aerofoil.cell_data['p'][:, None], aerofoil.cell_data['nut'][:, None]], axis = -1)

            #     bool_centers = np.concatenate([np.ones_like(centers.points[:, 0]), np.zeros_like(pos[:, 0])], axis = 0)
            #     surf_bool_centers = np.concatenate([np.ones_like(surf_centers.points[:, 0]), np.zeros_like(surf_pos[:, 0])], axis = 0)
            #     pos = np.concatenate([centers.points[:, :2], pos], axis = 0)
            #     init = np.concatenate([np.concatenate([centers.points[:, :2], attr[:, :6]], axis = 1), init], axis = 0)
            #     target = np.concatenate([attr[:, 6:], target], axis = 0)
            #     surf_pos = np.concatenate([surf_centers.points[:, :2], surf_pos], axis = 0)
            #     surf_init = np.concatenate([np.concatenate([surf_centers.points[:, :2], surf_attr[:, :6]], axis = 1), surf_init], axis = 0)
            #     surf_target = np.concatenate([surf_attr[:, 6:], surf_target], axis = 0)

            #     centers = torch.cat([torch.tensor(bool_centers), torch.tensor(surf_bool_centers)], dim = 0)

            # Put everything in tensor
            surf = torch.cat([torch.zeros(len(pos)), torch.ones(len(surf_pos))], dim = 0)
            pos = torch.cat([torch.tensor(pos, dtype = torch.float), torch.tensor(surf_pos, dtype = torch.float)], dim = 0) 
            x = torch.cat([torch.tensor(init, dtype = torch.float), torch.tensor(surf_init, dtype = torch.float)], dim = 0)
            y = torch.cat([torch.tensor(target, dtype = torch.float), torch.tensor(surf_target, dtype = torch.float)], dim = 0)            

        else: # Keep the mesh nodes
            surf_bool = (internal.point_data['U'][:, 0] == 0)

            # print('surf_bool: ',surf_bool)
            geom = -internal.point_data['implicit_distance'][:, None] # Signed distance function
            # print('sdf: ',geom)
            Uinf, alpha = float(s.split('_')[2]), float(s.split('_')[3])*np.pi/180
            # print('Uinf: ',Uinf)
            # print('alpha: ',alpha)
            # u = (np.array([np.cos(alpha), np.sin(alpha)])*Uinf).reshape(1, 2)*(internal.point_data['U'][:, :1] != 0)
            u = (np.array([np.cos(alpha), np.sin(alpha)])*Uinf).reshape(1, 2)*np.ones_like(internal.point_data['U'][:, :1])
            # print('u: ',u)
            normal = np.zeros_like(u)
            normal[surf_bool] = reorganize(aerofoil.points[:, :2], internal.points[surf_bool, :2], -aerofoil.point_data['Normals'][:, :2])
            # print('internal.point_data keys: ',internal.point_data.keys())

            # print('u , geom, normal, internal.point_data[u] [p] [nut] shapes:')
            # print(u.shape)
            # print(geom.shape)
            # print(normal.shape)
            # print(internal.point_data['U'].shape)
            # print(internal.point_data['p'].shape)
            # print(internal.point_data['nut'].shape)
            attr = np.concatenate([u, geom, normal,
                internal.point_data['U'][:, :2], internal.point_data['p'][:, None], internal.point_data['nut'][:, None]], axis = -1)

            # print('attr shape: ',attr.shape)
            # print('first 5 columns are x')
            # print(attr.shape[1]-5,' columns are y')
            pos = internal.points[:, :2] # cx,cy
            init = np.concatenate([pos, attr[:, :5]], axis = 1) # cx,cy, ux,uy,sdf,nx,ny
            target = attr[:, 5:] # U
            # saf = computeSAF(internal.points[:,:3],surf_bool,aerofoil)
            # Plotting SDF
            # if k==0:
            #     F = ( (pos[:,0]<2) & (pos[:,0]>-1)  & (pos[:,1]>-0.5) & (pos[:,1]<0.5))
            #     xyF = pos[F]
            #     sdfF = geom[F]
            #     plt.scatter(xyF[:,0],xyF[:,1],c=sdfF,s=1,cmap = 'viridis'); plt.colorbar(); plt.savefig(s+'_SDF.png'); plt.clf()
            # # Plotting SAF
            # if k<4:
            #     F = ( (pos[:,0]<2) & (pos[:,0]>-1)  & (pos[:,1]>-0.5) & (pos[:,1]<0.5))
            #     xyF = pos[F]
            #     safF = saf[F]
            #     plt.scatter(xyF[:,0],xyF[:,1],c=safF,s=1,cmap = 'viridis'); plt.colorbar(); plt.savefig(s+'_SAF.png'); plt.clf()
            # Put everything in tensor
            surf = torch.tensor(surf_bool)
            pos = torch.tensor(pos, dtype = torch.float)
            x = torch.tensor(init, dtype = torch.float)
            y = torch.tensor(target, dtype = torch.float) # ux,uy,sdf,nx,ny
            #print('target y :',y.shape)

        if norm and coef_norm is None:
            if k == 0:
                old_length = init.shape[0]
                mean_in = init.mean(axis = 0, dtype = np.double)
                mean_out = target.mean(axis = 0, dtype = np.double)
            else:
                new_length = old_length + init.shape[0]
                mean_in += (init.sum(axis = 0, dtype = np.double) - init.shape[0]*mean_in)/new_length
                mean_out += (target.sum(axis = 0, dtype = np.double) - init.shape[0]*mean_out)/new_length
                old_length = new_length 

        # Graph definition
        # if cell_centers:
        #     data = Data(pos = pos, x = x, y = y, surf = surf.bool(), centers = centers.bool())
        # else:
        #     data = Data(pos = pos, x = x, y = y, surf = surf.bool())
        if use_saf:
            saf = torch.load(manifest_saf[s])
        else:
            saf = None 
        if (use_dsdf):
            dsdf = torch.load(manifest_dsdf[s])
        else:
            dsdf = None
        data = Data(pos = pos, x = x, y = y, surf = surf.bool(),saf = saf,dsdf=dsdf)
        dataset.append(data)

    if norm and coef_norm is None:
        # Compute normalization
        mean_in = mean_in.astype(np.single)
        mean_out = mean_out.astype(np.single)
        # Umean = np.linalg.norm(data.x[:, 2:4], axis = 1).mean()     
        for k, data in enumerate(dataset):
            # data.x = data.x/torch.tensor([6, 6, Umean, Umean, 6, 1, 1], dtype = torch.float)
            # data.y = data.y/torch.tensor([Umean, Umean, .5*Umean**2, Umean], dtype = torch.float)

            if k == 0:
                old_length = data.x.numpy().shape[0]
                std_in = ((data.x.numpy() - mean_in)**2).sum(axis = 0, dtype = np.double)/old_length
                std_out = ((data.y.numpy() - mean_out)**2).sum(axis = 0, dtype = np.double)/old_length
            else:
                new_length = old_length + data.x.numpy().shape[0]
                std_in += (((data.x.numpy() - mean_in)**2).sum(axis = 0, dtype = np.double) - data.x.numpy().shape[0]*std_in)/new_length
                std_out += (((data.y.numpy() - mean_out)**2).sum(axis = 0, dtype = np.double) - data.x.numpy().shape[0]*std_out)/new_length
                old_length = new_length
        
        std_in = np.sqrt(std_in).astype(np.single)
        std_out = np.sqrt(std_out).astype(np.single)

        # Normalize
        for data in dataset:
            data.x = (data.x - mean_in)/(std_in + 1e-8)
            data.y = (data.y - mean_out)/(std_out + 1e-8)

        coef_norm = (mean_in, std_in, mean_out, std_out)     
        dataset = (dataset, coef_norm)   
    
    elif coef_norm is not None:
        # Normalize
        for data in dataset:
            # data.x = data.x/torch.tensor([6, 6, coef_norm[-1], coef_norm[-1], 6, 1, 1], dtype = torch.float)
            # data.y = data.y/torch.tensor([coef_norm[-1], coef_norm[-1], .5*coef_norm[-1]**2, coef_norm[-1]], dtype = torch.float)
            data.x = (data.x - coef_norm[0])/(coef_norm[1] + 1e-8)
            data.y = (data.y - coef_norm[2])/(coef_norm[3] + 1e-8)
    
    return dataset
    
def FVDatasetreplaceSAF(set,manifest_fv, norm = False, coef_norm = None):
    print('FVDatasetreplaceSAF')
    dataset = []
    for k, s in enumerate(tqdm(set)):
        data = torch.load(manifest_fv[s])
        
        #data.saf = GcomputeSAF2(data.x[:,:2],surf_bool=data.surf)
        # data.saf = torch.load(manifest_saf[s])
        # Put everything in tensor
        #x = torch.hstack((data.x,data.sdf.unsqueeze(1))) # Appending sdf to data.x because sdf needs to be normalized as well.
        x = torch.hstack((data.x,data.saf,data.dsdf)).to(dtype=torch.float32)
        y = data.y.to(dtype=torch.float32)
        data.edge_attr = data.edge_attr.to(dtype=torch.float32)
        data.node_attr = data.node_attr.to(dtype=torch.float32)

        if norm and coef_norm is None:
            if k == 0:
                old_length = x.shape[0]
                mean_in = torch.mean(x,0)
                mean_out = torch.mean(y,0)
            else:
                new_length = old_length + x.shape[0]
                mean_in += (torch.sum(x,0) - x.shape[0]*mean_in)/new_length
                mean_out += (torch.sum(y,0) - x.shape[0]*mean_out)/new_length
                old_length = new_length 

        
        data.x = x # Appending sdf to data.x because sdf needs to be normalized as well.
        
        pos = data.x[:,0:2]
        #edge_attr = [Face area, (face centroid - ni) coor, (face centroid - nj) coor)]
        edge_attr = torch.cat((data.edge_attr[:,0:1],(data.edge_attr[:,1:]-pos[data.edge_index[0,:],:]),(data.edge_attr[:,1:]-pos[data.edge_index[1,:],:])), dim=1)
        node_attr = torch.cat((pos,data.node_attr.unsqueeze(1)),dim=1)
        
        A1 = [torch.sparse_coo_tensor(data.edge_index,edge_attr[:,d],\
              size=(x.size(0),x.size(0))) for d in range(edge_attr.size(1))]
        A2 = [torch.sparse.mm(A1[d],A1[d]) for d in range(len(A1))]
        data.edge_indexA2 = A2[0].coalesce().indices()
        edge_attrA2 = torch.stack([A2[d].coalesce().values() for d in range(len(A2))],dim=1)
        
        rel_Nattr = node_attr[data.edge_index[0,:],:] - node_attr[data.edge_index[1,:],:]
        data.edge_attr = torch.cat((rel_Nattr,edge_attr), dim=1)
        rel_NattrA2 = node_attr[data.edge_indexA2[0,:],:] - node_attr[data.edge_indexA2[1,:],:]
        data.edge_attrA2 = torch.cat((rel_NattrA2,edge_attrA2), dim=1)
        data.node_attr = None 
        data.sdf = None
        data.pos = pos
        if data.dsdf.shape[0]>data.dsdf.shape[1]:
            data.dsdf = data.dsdf.t()
        data.y = y
        dataset.append(data)
        
    if norm and coef_norm is None:
        # Compute normalization
        # print('mean_in: ',mean_in)
        # print('mean_out: ',mean_out)
        mean_in = mean_in.numpy().astype(np.single)
        mean_out = mean_out.numpy().astype(np.single)
        for k, data in enumerate(dataset):
            if k == 0:
                old_length = data.x.numpy().shape[0]
                std_in = ((data.x.numpy() - mean_in)**2).sum(axis = 0, dtype = np.double)/old_length
                std_out = ((data.y.numpy() - mean_out)**2).sum(axis = 0, dtype = np.double)/old_length
            else:
                new_length = old_length + data.x.numpy().shape[0]
                std_in += (((data.x.numpy() - mean_in)**2).sum(axis = 0, dtype = np.double) - data.x.numpy().shape[0]*std_in)/new_length
                std_out += (((data.y.numpy() - mean_out)**2).sum(axis = 0, dtype = np.double) - data.x.numpy().shape[0]*std_out)/new_length
                old_length = new_length
        
        std_in = np.sqrt(std_in).astype(np.single)
        std_out = np.sqrt(std_out).astype(np.single)

        # Normalize
        # print('mean: ',mean_in,' ',mean_out)
        # print('std: ',std_in,' ',std_out)
        for data in dataset:
            data.x = (data.x - mean_in)/(std_in + 1e-8)
            data.y = (data.y - mean_out)/(std_out + 1e-8)

        coef_norm = (mean_in, std_in, mean_out, std_out)     
        dataset = (dataset, coef_norm)   
    
    elif coef_norm is not None:
        # print('normalizing coefficients: ')
        # print('mean: ',coef_norm[0],' ',coef_norm[2])
        # print('std: ',coef_norm[1],' ',coef_norm[3])
        # Normalize
        for data in dataset:
            # data.x = data.x/torch.tensor([6, 6, coef_norm[-1], coef_norm[-1], 6, 1, 1], dtype = torch.float)
            # data.y = data.y/torch.tensor([coef_norm[-1], coef_norm[-1], .5*coef_norm[-1]**2, coef_norm[-1]], dtype = torch.float)
            data.x = (data.x - coef_norm[0])/(coef_norm[1] + 1e-8)
            data.y = (data.y - coef_norm[2])/(coef_norm[3] + 1e-8)
    
    return dataset

def FVDataset(set, manifest_fv, norm = False, coef_norm = None):
    '''
    Create a list of simulation to input in a PyTorch Geometric DataLoader. Simulation are transformed by keeping vertices of the CFD mesh or 
    by sampling (uniformly or via the mesh density) points in the simulation cells.

    Args:
        set (list): List of geometry names to include in the dataset.
        norm (bool, optional): If norm is set to ``True``, the mean and the standard deviation of the dataset will be computed and returned. 
            Moreover, the dataset will be normalized by these quantities. Ignored when ``coef_norm`` is not None. Default: ``False``
        coef_norm (tuple, optional): This has to be a tuple of the form (mean input, std input, mean output, std ouput) if not None. 
            The dataset generated will be normalized by those quantites. Default: ``None``
        crop (list, optional): List of the vertices of the rectangular [xmin, xmax, ymin, ymax] box to crop simulations. Default: ``None``
        surf_ratio (float, optional): Used only if sample is not None, gives the ratio of point over the airfoil to sample with respect to point
            in the volume. Default: ``0.1``
    '''
    # if norm and coef_norm is not None:
    #     raise ValueError('If coef_norm is not None and norm is True, the normalization will be done via coef_norm')

    dataset = []

    for k, s in enumerate(tqdm(set)):
        data = torch.load(manifest_fv[s])
        # Put everything in tensor
        #x = torch.hstack((data.x,data.sdf.unsqueeze(1))) # Appending sdf to data.x because sdf needs to be normalized as well.
        x = torch.hstack((data.x,data.saf,data.dsdf)).to(dtype=torch.float32)
        y = data.y.to(dtype=torch.float32)
        data.edge_attr = data.edge_attr.to(dtype=torch.float32)
        data.node_attr = data.node_attr.to(dtype=torch.float32)

        if norm and coef_norm is None:
            if k == 0:
                old_length = x.shape[0]
                mean_in = torch.mean(x,0)
                mean_out = torch.mean(y,0)
            else:
                new_length = old_length + x.shape[0]
                mean_in += (torch.sum(x,0) - x.shape[0]*mean_in)/new_length
                mean_out += (torch.sum(y,0) - x.shape[0]*mean_out)/new_length
                old_length = new_length 

        # data = Data(pos = pos, x = x, y = y, surf = surf.bool(),saf = saf,dsdf=dsdf)
        #data.dsdf = x[:,8:16]; data.saf = x[:,6:8]; #data.sdf = x[:,16:]
        #data.x = x[:,:6]
        
        data.x = x # Appending sdf to data.x because sdf needs to be normalized as well.
        
        pos = data.x[:,0:2]
        #edge_attr = [Face area, (face centroid - ni) coor, (face centroid - nj) coor)]
        edge_attr = torch.cat((data.edge_attr[:,0:1],(data.edge_attr[:,1:]-pos[data.edge_index[0,:],:]),(data.edge_attr[:,1:]-pos[data.edge_index[1,:],:])), dim=1)
        node_attr = torch.cat((pos,data.node_attr.unsqueeze(1)),dim=1)
        
        A1 = [torch.sparse_coo_tensor(data.edge_index,edge_attr[:,d],\
              size=(x.size(0),x.size(0))) for d in range(edge_attr.size(1))]
        A2 = [torch.sparse.mm(A1[d],A1[d]) for d in range(len(A1))]
        data.edge_indexA2 = A2[0].coalesce().indices()
        edge_attrA2 = torch.stack([A2[d].coalesce().values() for d in range(len(A2))],dim=1)
        
        rel_Nattr = node_attr[data.edge_index[0,:],:] - node_attr[data.edge_index[1,:],:]
        data.edge_attr = torch.cat((rel_Nattr,edge_attr), dim=1)
        rel_NattrA2 = node_attr[data.edge_indexA2[0,:],:] - node_attr[data.edge_indexA2[1,:],:]
        data.edge_attrA2 = torch.cat((rel_NattrA2,edge_attrA2), dim=1)
        data.node_attr = None 
        data.sdf = None
        data.saf, data.dsdf = None, None
        # data.pos = pos
        
        data.y = y
        dataset.append(data)
        # print(dataset)

    if norm and coef_norm is None:
        # Compute normalization
        # print('mean_in: ',mean_in)
        # print('mean_out: ',mean_out)
        mean_in = mean_in.numpy().astype(np.single)
        mean_out = mean_out.numpy().astype(np.single)
        for k, data in enumerate(dataset):
            if k == 0:
                old_length = data.x.numpy().shape[0]
                std_in = ((data.x.numpy() - mean_in)**2).sum(axis = 0, dtype = np.double)/old_length
                std_out = ((data.y.numpy() - mean_out)**2).sum(axis = 0, dtype = np.double)/old_length
            else:
                new_length = old_length + data.x.numpy().shape[0]
                std_in += (((data.x.numpy() - mean_in)**2).sum(axis = 0, dtype = np.double) - data.x.numpy().shape[0]*std_in)/new_length
                std_out += (((data.y.numpy() - mean_out)**2).sum(axis = 0, dtype = np.double) - data.x.numpy().shape[0]*std_out)/new_length
                old_length = new_length
        
        std_in = np.sqrt(std_in).astype(np.single)
        std_out = np.sqrt(std_out).astype(np.single)

        # Normalize
        # print('mean: ',mean_in,' ',mean_out)
        # print('std: ',std_in,' ',std_out)
        for data in dataset:
            data.x = (data.x - mean_in)/(std_in + 1e-8)
            data.y = (data.y - mean_out)/(std_out + 1e-8)

        coef_norm = (mean_in, std_in, mean_out, std_out)     
        dataset = (dataset, coef_norm)   
    
    elif coef_norm is not None:
        # print('normalizing coefficients: ')
        # print('mean: ',coef_norm[0],' ',coef_norm[2])
        # print('std: ',coef_norm[1],' ',coef_norm[3])
        # Normalize
        for data in dataset:
            # data.x = data.x/torch.tensor([6, 6, coef_norm[-1], coef_norm[-1], 6, 1, 1], dtype = torch.float)
            # data.y = data.y/torch.tensor([coef_norm[-1], coef_norm[-1], .5*coef_norm[-1]**2, coef_norm[-1]], dtype = torch.float)
            data.x = (data.x - coef_norm[0])/(coef_norm[1] + 1e-8)
            data.y = (data.y - coef_norm[2])/(coef_norm[3] + 1e-8)
    
    return dataset

def GUNetDataset(set, norm = False, coef_norm = None, manifest_dsdf=None, manifest_saf=None):
    '''
    Create a list of simulation to input in a PyTorch Geometric DataLoader. Simulation are transformed by keeping vertices of the CFD mesh or 
    by sampling (uniformly or via the mesh density) points in the simulation cells.

    Args:
        set (list): List of geometry names to include in the dataset.
        norm (bool, optional): If norm is set to ``True``, the mean and the standard deviation of the dataset will be computed and returned. 
            Moreover, the dataset will be normalized by these quantities. Ignored when ``coef_norm`` is not None. Default: ``False``
        coef_norm (tuple, optional): This has to be a tuple of the form (mean input, std input, mean output, std ouput) if not None. 
            The dataset generated will be normalized by those quantites. Default: ``None``
        crop (list, optional): List of the vertices of the rectangular [xmin, xmax, ymin, ymax] box to crop simulations. Default: ``None``
        surf_ratio (float, optional): Used only if sample is not None, gives the ratio of point over the airfoil to sample with respect to point
            in the volume. Default: ``0.1``
    '''
    # if norm and coef_norm is not None:
    #     raise ValueError('If coef_norm is not None and norm is True, the normalization will be done via coef_norm')

    dataset = []
    for k, s in enumerate(tqdm(set,desc = 'loading GUnet+Geo dataset')):
        internal = pv.read('Dataset/' + s + '/' + s + '_internal.vtu')
        aerofoil = pv.read('Dataset/' + s + '/' + s + '_aerofoil.vtp')
        internal = internal.compute_cell_sizes(length = False, volume = False)
        
        surf_bool = (internal.point_data['U'][:, 0] == 0)

        # geom = -internal.point_data['implicit_distance'][:, None] # Signed distance function
        Uinf, alpha = float(s.split('_')[2]), float(s.split('_')[3])*np.pi/180

        u = (np.array([np.cos(alpha), np.sin(alpha)])*Uinf).reshape(1, 2)*np.ones_like(internal.point_data['U'][:, :1])

        normal = np.zeros_like(u)
        normal[surf_bool] = reorganize(aerofoil.points[:, :2], internal.points[surf_bool, :2], -aerofoil.point_data['Normals'][:, :2])
        attr = np.concatenate([u, normal,
            internal.point_data['U'][:, :2], internal.point_data['p'][:, None], internal.point_data['nut'][:, None]], axis = -1)

        pos = internal.points[:, :2] # cx,cy
        init = np.concatenate([pos, attr[:, :4]], axis = 1) # cx,cy, ux,uy,nx,ny
        target = attr[:, 4:] # U

        # Put everything in tensor
        surf = torch.tensor(surf_bool)
        pos = torch.tensor(pos, dtype = torch.float)
        x = torch.tensor(init, dtype = torch.float)
        if manifest_saf is not None:
            saf = torch.load(manifest_saf[s])
        if manifest_dsdf is not None:
            dsdf = torch.load(manifest_dsdf[s])

        x = torch.hstack((x,saf,dsdf.t())).to(dtype=torch.float) # cx,cy, ux,uy,nx,ny, saf,dsdf
        y = torch.tensor(target, dtype = torch.float) 

        if norm and coef_norm is None:
            init = x.clone().numpy()
            if k == 0:
                old_length = init.shape[0]
                mean_in = init.mean(axis = 0, dtype = np.double)
                mean_out = target.mean(axis = 0, dtype = np.double)
            else:
                new_length = old_length + init.shape[0]
                mean_in += (init.sum(axis = 0, dtype = np.double) - init.shape[0]*mean_in)/new_length
                mean_out += (target.sum(axis = 0, dtype = np.double) - init.shape[0]*mean_out)/new_length
                old_length = new_length 

        data = Data(pos = pos, x = x, y = y, surf = surf.bool(), node_attr = pos, saf = saf, dsdf = dsdf)
        dataset.append(data)

    if norm and coef_norm is None:
        # Compute normalization
        mean_in = mean_in.astype(np.single)
        mean_out = mean_out.astype(np.single)
        # Umean = np.linalg.norm(data.x[:, 2:4], axis = 1).mean()     
        for k, data in enumerate(dataset):
            # data.x = data.x/torch.tensor([6, 6, Umean, Umean, 6, 1, 1], dtype = torch.float)
            # data.y = data.y/torch.tensor([Umean, Umean, .5*Umean**2, Umean], dtype = torch.float)

            if k == 0:
                old_length = data.x.numpy().shape[0]
                std_in = ((data.x.numpy() - mean_in)**2).sum(axis = 0, dtype = np.double)/old_length
                std_out = ((data.y.numpy() - mean_out)**2).sum(axis = 0, dtype = np.double)/old_length
            else:
                new_length = old_length + data.x.numpy().shape[0]
                std_in += (((data.x.numpy() - mean_in)**2).sum(axis = 0, dtype = np.double) - data.x.numpy().shape[0]*std_in)/new_length
                std_out += (((data.y.numpy() - mean_out)**2).sum(axis = 0, dtype = np.double) - data.x.numpy().shape[0]*std_out)/new_length
                old_length = new_length
        
        std_in = np.sqrt(std_in).astype(np.single)
        std_out = np.sqrt(std_out).astype(np.single)

        # Normalize
        for data in dataset:
            data.x = (data.x - mean_in)/(std_in + 1e-8)
            data.y = (data.y - mean_out)/(std_out + 1e-8)

        coef_norm = (mean_in, std_in, mean_out, std_out)     
        dataset = (dataset, coef_norm)   
    
    elif coef_norm is not None:
        # Normalize
        for data in dataset:
            # data.x = data.x/torch.tensor([6, 6, coef_norm[-1], coef_norm[-1], 6, 1, 1], dtype = torch.float)
            # data.y = data.y/torch.tensor([coef_norm[-1], coef_norm[-1], .5*coef_norm[-1]**2, coef_norm[-1]], dtype = torch.float)
            data.x = (data.x - coef_norm[0])/(coef_norm[1] + 1e-8)
            data.y = (data.y - coef_norm[2])/(coef_norm[3] + 1e-8)
    
    return dataset

def NSResDataset(set, norm = False, coef_norm = None, res_est = None):
    #print('NSResDataset_noappend')
    print('NSResDataset')
    dataset = []
    for k, s in enumerate(tqdm(set)):
        data = torch.load(os.path.join('nsresd/',s+'.pkl'))
        data.x = torch.hstack((data.x,data.saf,data.dsdf)).to(dtype=torch.float32)
        if res_est is not None:
            #data.x = torch.hstack((data.x,res_est[k].to(dtype=torch.float32))) # comment for no_append
            data.estimate = res_est[k].to(dtype=torch.float32)
        data.y = data.y.to(dtype = torch.float32)
        
        x = data.x
        # del data.saf 
        del data.dsdf 
        del data.sdf 
        del data.internal_surf_flag
        y = data.y.to(dtype=torch.float32)
        data.edge_attr = data.edge_attr.to(dtype=torch.float32)
        data.node_attr = data.node_attr.unsqueeze(1).to(dtype=torch.float32)

        if norm and coef_norm is None:
            if k == 0:
                old_length = x.shape[0]
                mean_in = torch.mean(x,0)
                mean_out = torch.mean(y,0)
            else:
                new_length = old_length + x.shape[0]
                mean_in += (torch.sum(x,0) - x.shape[0]*mean_in)/new_length
                mean_out += (torch.sum(y,0) - x.shape[0]*mean_out)/new_length
                old_length = new_length 

        dataset.append(data)
        
    if norm and coef_norm is None:
        # Compute normalization
        mean_in = mean_in.numpy().astype(np.single)
        mean_out = mean_out.numpy().astype(np.single)
        for k, data in enumerate(dataset):
            if k == 0:
                old_length = data.x.numpy().shape[0]
                std_in = ((data.x.numpy() - mean_in)**2).sum(axis = 0, dtype = np.double)/old_length
                std_out = ((data.y.numpy() - mean_out)**2).sum(axis = 0, dtype = np.double)/old_length
            else:
                new_length = old_length + data.x.numpy().shape[0]
                std_in += (((data.x.numpy() - mean_in)**2).sum(axis = 0, dtype = np.double) - data.x.numpy().shape[0]*std_in)/new_length
                std_out += (((data.y.numpy() - mean_out)**2).sum(axis = 0, dtype = np.double) - data.x.numpy().shape[0]*std_out)/new_length
                old_length = new_length
        
        std_in = np.sqrt(std_in).astype(np.single)
        std_out = np.sqrt(std_out).astype(np.single)

        # Normalize
        for k,data in enumerate(dataset):
            data.x = (data.x - mean_in)/(std_in + 1e-8)
            data.y = (data.y - mean_out)/(std_out + 1e-8)
            if res_est is not None:
            #    data.estimate = (data.estimate - mean_out)/(std_out + 1e-8)
                data.x = torch.hstack((data.x,res_est[k].to(dtype=torch.float32)))
            
        coef_norm = (mean_in, std_in, mean_out, std_out)     
        dataset = (dataset, coef_norm)   
    
    elif coef_norm is not None:
        # Normalize
        for k,data in enumerate(dataset):
            data.x = (data.x - coef_norm[0])/(coef_norm[1] + 1e-8)
            data.y = (data.y - coef_norm[2])/(coef_norm[3] + 1e-8)
            if res_est is not None:
            #    data.estimate = (data.estimate -coef_norm[2])/(coef_norm[3] + 1e-8)
                data.x = torch.hstack((data.x,res_est[k].to(dtype=torch.float32)))
    #print(dataset[0][0])
    #print(dataset[0][0].x)
    #print(dataset[0][0].y)
    #print(dataset[0][0].estimate)
    return dataset