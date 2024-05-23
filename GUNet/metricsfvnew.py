import numpy as np
import scipy as sc
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import os 
import pyvista as pv
import json
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
import math
import metrics_NACA
from reorganize import reorganize
from dataset import NSResDataset
from copy import deepcopy
from tqdm import tqdm
import openfoamparser as Ofpp
from FVdataset import compute_normalOutward,get_cellCentres
from NSres import continuity_error
NU = np.array(1.56e-5)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def rsquared(predict, true):
    '''
    Args:
        predict (tensor): Predicted values, shape (N, *)
        true (tensor): True values, shape (N, *)

    Out:
        rsquared (tensor): Coefficient of determination of the prediction, shape (*,)
    '''
    mean = true.mean(dim = 0)
    return 1 - ((true - predict)**2).sum(dim = 0)/((true - mean)**2).sum(dim = 0)

def rel_err(a, b):
    return np.abs((a - b)/a)

def WallShearStress(Jacob_U, normals):
    S = .5*(Jacob_U + Jacob_U.transpose(0, 2, 1))
    S = S - S.trace(axis1 = 1, axis2 = 2).reshape(-1, 1, 1)*np.eye(2)[None]/3
    ShearStress = 2*NU.reshape(-1, 1, 1)*S
    ShearStress = (ShearStress*normals[:, :2].reshape(-1, 1, 2)).sum(axis = 2)

    return ShearStress

def WallShearStress_torch(Jacob_U, normals, device='cpu'):
    S = .5*(Jacob_U + Jacob_U.transpose(1,2))
    #print('torch: ',S.diagonal(offset=0, dim1=1, dim2=2).sum(0).shape,' ',S.diagonal(offset=0, dim1=1, dim2=2).sum(1).shape)
    traceS = S.diagonal(offset=0, dim1=1, dim2=2).sum(1)
    S_trace_S = S - traceS.reshape(-1,1,1)*torch.eye(2)[None].to(device)/3
    #S = S - S.trace(axis1 = 1, axis2 = 2).reshape(-1, 1, 1)*np.eye(2)[None]/3
    NU = torch.tensor([1.56e-5]).to(device)
    # ShearStress = 2*NU.reshape(-1, 1, 1)*S
    ShearStress = 2*NU.reshape(-1, 1, 1)*S_trace_S
    ShearStress = (ShearStress*normals[:, :2].reshape(-1, 1, 2)).sum(2)
    return ShearStress
    
@torch.no_grad()
def Infer_test(device, models, hparams, data, coef_norm = None):
    use_correct_loss = False 
    reg = 1
    # Inference procedure on new simulation
    outs = [torch.zeros_like(data.y)]*len(models)
    train_losses = torch.zeros(len(models),1)

    data_sampled = data.clone()
   
    tim = np.zeros(len(models))
    for n, model in enumerate(models):
        model.eval()
        data_sampled = data_sampled.to(device)
        start = time.time()
        o = model(data_sampled)
        tim[n] += time.time() - start
        # out[n][idx] = o.cpu()
        # outs[n] = outs[n] + out[n]
        outs[n] = o.cpu()
    # n_out[idx] = n_out[idx] + torch.ones_like(n_out[idx])

    # cond = (len(idx_points) > 0)
    #un_normalised_out = deepcopy(outs)
    for n, out in enumerate(outs):
        # outs[n] = out/n_out  
        if coef_norm is not None: # normalise outs[surf]
            outs[n][data.surf, :2] = -torch.tensor(coef_norm[2][None, :2])*torch.ones_like(out[data.surf, :2])/(torch.tensor(coef_norm[3][None, :2]) + 1e-8)
            outs[n][data.surf, 3] = -torch.tensor(coef_norm[2][3])*torch.ones_like(out[data.surf, 3])/(torch.tensor(coef_norm[3][3]) + 1e-8)
        else:
            outs[n][data.surf, :2] = torch.zeros_like(out[data.surf, :2])
            outs[n][data.surf, 3] = torch.zeros_like(out[data.surf, 3])
        if not use_correct_loss:
            criterion = nn.MSELoss(reduction = 'none')
            train_losses[n,0] = criterion(outs[n], data.y).mean(dim = 0).mean()
            train_losses_uv = criterion(outs[n][:2], data.y[:2]).mean(dim = 0).mean(0)
        else:
            loss_surf_var = criterion(outs[n][data.surf, :], data.y[data.surf, :]).mean(dim = 0)
            loss_vol_var = criterion(outs[n][~data.surf, :], data.y[~data.surf, :]).mean(dim = 0)
        # reg = 1
        # print('loss_vol_var.mean(): ',loss_vol_var.mean())
        # print('loss_surf_var.mean(): ',loss_surf_var.mean())
            train_losses[n,0] = (loss_vol_var.mean()  + reg*(loss_surf_var.mean()))
            train_losses_uv = (loss_vol_var[:,:2].mean(0) + reg* loss_surf_var[:,:2].mean(0)).mean(0)
        
    return outs, tim, train_losses, train_losses_uv

def Airfoil_test(internal, airfoil, outs, coef_norm, bool_surf):
    # Produce multiple copies of a simulation for different predictions.
    # stocker les internals, airfoils, calculer le wss, calculer le drag, le lift, plot pressure coef, plot skin friction coef, plot drag/drag, plot lift/lift
    # calcul spearsman coef, boundary layer
    internals = []
    airfoils = []
    for out in outs:
        intern = internal.copy()
        aerofoil = airfoil.copy()

        point_mesh = intern.points[bool_surf, :2]
        point_surf = aerofoil.points[:, :2]
        out = (out*(coef_norm[3] + 1e-8) + coef_norm[2]).numpy()
        out[bool_surf.numpy(), :2] = np.zeros_like(out[bool_surf.numpy(), :2])
        out[bool_surf.numpy(), 3] = np.zeros_like(out[bool_surf.numpy(), 3])
        intern.point_data['U'][:, :2] = out[:, :2]
        intern.point_data['p'] = out[:, 2]
        intern.point_data['nut'] = out[:, 3]

        surf_p = intern.point_data['p'][bool_surf]
        surf_p = reorganize(point_mesh, point_surf, surf_p)
        aerofoil.point_data['p'] = surf_p

        intern = intern.ptc(pass_point_data = True) 
        aerofoil = aerofoil.ptc(pass_point_data = True)       

        internals.append(intern)
        airfoils.append(aerofoil)
    
    return internals, airfoils

def Compute_coefficients(internals, airfoils, bool_surf, Uinf, angle, keep_vtk = False):
    # Compute force coefficients, if keet_vtk is True, also return the .vtu/.vtp with wall shear stress added over the airfoil and velocity gradient over the volume.
    verbose = False 
    coefs = []
    if keep_vtk:
        new_internals = []
        new_airfoils = []
    
    for internal, airfoil in zip(internals, airfoils):
        intern = internal.copy()
        aerofoil = airfoil.copy()

        point_mesh = intern.points[bool_surf, :2]
        point_surf = aerofoil.points[:, :2]
        if (verbose):
            print('before ddx: ', intern)
            print('type(intern): ',type(intern))
        intern = intern.compute_derivative(scalars = 'U', gradient = 'pred_grad')
        if verbose:
            print('after ddx: ', intern)
            print('after ddx: ', intern['pred_grad'].shape)
        
        # print("intern.point_data['pred_grad'] : ",intern.point_data['pred_grad'].shape)
        # print("aerofoil.point_data['Normals']: ",aerofoil.point_data['Normals'].shape)
        surf_grad = intern.point_data['pred_grad'].reshape(-1, 3, 3)[bool_surf, :2, :2]
        surf_p = intern.point_data['p'][bool_surf]

        surf_grad = reorganize(point_mesh, point_surf, surf_grad)
        surf_p = reorganize(point_mesh, point_surf, surf_p)
        if verbose:
            print("surf_grad.shape: ",surf_grad.shape, " ","Normals: ",aerofoil.point_data['Normals'].shape)
        Wss_pred = WallShearStress(surf_grad, -aerofoil.point_data['Normals'])
        aerofoil.point_data['wallShearStress'] = Wss_pred
        aerofoil.point_data['p'] = surf_p

        intern = intern.ptc(pass_point_data = True) 
        aerofoil = aerofoil.ptc(pass_point_data = True)
        if verbose:
            print("aerofoil.cell_data['p'][:, None]: ",aerofoil.cell_data['p'][:, None].shape)
        WP_int = -aerofoil.cell_data['p'][:, None]*aerofoil.cell_data['Normals'][:, :2]
        if (verbose):
            print("aerofoil.cell_data['Length'] : ",aerofoil.cell_data['Length'],' \n\r',aerofoil.cell_data['Length'].shape)
        
        #EDIT: Replace aerofoil.cell_data['Length'] with magSf[airfoil patch] (edge_attr[0])
        Wss_int = (aerofoil.cell_data['wallShearStress']*aerofoil.cell_data['Length'].reshape(-1, 1)).sum(axis = 0)
        WP_int = (WP_int*aerofoil.cell_data['Length'].reshape(-1, 1)).sum(axis = 0)
        force = Wss_int - WP_int

        alpha = angle*np.pi/180
        basis = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
        force_rot = basis@force
        coef = 2*force_rot/Uinf**2
        coefs.append(coef)
        if keep_vtk:
            new_internals.append(intern)
            new_airfoils.append(aerofoil)
        
    if keep_vtk:
        return coefs, new_internals, new_airfoils
    else:
        return coefs

def gradXY(batch, fxy=None):
    f = fxy if fxy is not None else batch['y'] #given values vs GT
    N, numFeat = batch.x.size(0),f.size(1); #ASSUME N=spatialDim=2 first. 
    x = batch.x[:,:2]
    
    owner_i = batch.edge_index[0,batch.owner_to_neighbor] #~Ex1
    neighbour_i = batch.edge_index[1,batch.owner_to_neighbor] #~Ex1

    #1: get surfacefield ssf via linear interpolation (Us)
    # e = batch.edge_index #size 2xE
    ssf_coor = batch.cf[:,0:2] #face centroid coordinate, size ~Ex2
    w = [(x[owner_i] - ssf_coor).norm(dim=1,p=2).to(dtype=f.dtype), \
       (x[neighbour_i] - ssf_coor).norm(dim=1,p=2).to(dtype=f.dtype)] #weights for interpolation
    w[0],w[1] = w[0].reshape(-1,1) , w[1].reshape(-1,1)
    ssf_i = (w[0]*f[owner_i] + w[1]*f[neighbour_i])/(w[0] + w[1]) #size ExnumFeat
    #print('ssf_i =>' , torch.any(ssf_i.isnan()))
    #2: get face area normal vector, Sf (Ns)
    Sf_i = batch.sf[:,:2].to(dtype=f.dtype) #size Ex2 or Ex(spatialDim)

    #3: loop over each face_i in internal surface
    Sfssf = Sf_i.unsqueeze(1) * ssf_i.unsqueeze(2) #~Ex2xnumfeat
    
    grad = torch.zeros((N,numFeat,2),dtype=Sfssf.dtype).to(Sfssf.device)
    grad = grad.index_add_(0,owner_i,Sfssf)
    #print('grad.index_add =>',torch.any(grad.isnan())) 
    grad = grad.index_add_(0,neighbour_i,Sfssf,alpha=-1)
    #print('grad.index_add =>',torch.any(grad.isnan())) 
    del Sfssf, ssf_i, Sf_i, ssf_coor
    
    # #4: Loop over boundary faces.
    # cellNext = face_data.cell_next; afSf = face_data.cell_next_normal
    #     Sf_i = afSf[i]; ssf_i = torch.tensor([0,0]) ??
    #     grad[nextCellid] += torch.outer(Sf_i,ssf_i) #spatialDim x numFeat???
        
    # 5: divide by volume, then surface points follow adjacent cell value?
    V = batch.node_attr.unsqueeze(1).unsqueeze(1).to(dtype=grad.dtype)
    #print(V[batch.cell_next])
    grad = torch.div(grad, V)
    grad[batch.surf,:] = grad[batch.cell_next]
    #print('grad[batch.cell_next] =>',torch.any(grad[batch.cell_next].isnan()))
    return grad #Ex2x2

def Compute_coefficients_forLoss(data_orig, Af, Uinf,angle,device = 'cuda:0'):
    #print(airfoil_data_pkl)
    # coefs = []
    verbose = False
    if verbose: print('New method:')
    #data = data_orig.clone() # unnormalized
    data = data_orig[0].to(device)
    #print('for loss: ',data.node_attr[data.cell_next])
    #Af = torch.load(airfoil_data_pkl[0]).to(device)
    
    #print(type(airfoil_data_pkl),' ',len(airfoil_data_pkl))
    incident_cells = Af.indices
    Sfn_out = data.x[data.surf,4:]
    
    surf_grad = gradXY(data,fxy = data.y[:,:2])[data.surf,:,:]
    #print('surf_grad nan => ',torch.any(torch.isnan(surf_grad)))
    if verbose: 
        print('surf_grad: ',torch.min(surf_grad),torch.max(surf_grad))
    surf_p = data.y[incident_cells,2]
    Sfn = -Sfn_out # inward
    cell_length = torch.sqrt(Af.val[:,4]**2 + Af.val[:,5]**2)
    # normal = data_orig.cf[:,:2]
    # cell_length = torch.sqrt(normal[:,0]**2+normal[:,1]**2)
    if verbose: print('cell_length: ',torch.min(cell_length),' ',torch.max(cell_length))
    if verbose: print('outward normal: ', torch.min(-Sfn),' ',torch.max(-Sfn)) 

    #Wss_pred = WallShearStress(surf_grad.numpy(), -Sfn.numpy())
    #print("wallshearstress_np: ",Wss_pred.min(),' ',Wss_pred.max())
    wallshearstress = WallShearStress_torch(surf_grad,-Sfn,device = device)
    #print("wallshearstress_torch: ",wallshearstress.shape, ' => ',torch.min(wallshearstress),' ',torch.max(wallshearstress))
    #wallshearstress = torch.from_numpy(Wss_pred)
    # if verbose: print('surf_p: ',np.sort(surf_p))
    if verbose: print('surfP: ',torch.min(surf_p),' ',torch.max(surf_p))
    WP_int = surf_p[:, None]*Sfn_out
    # if verbose: print(torch.min(Sfn_out),' ',torch.max(Sfn_out))
    Wss_int = (wallshearstress*cell_length.reshape(-1, 1)).sum(axis = 0)
    if verbose: print("wallshearstress: ",torch.min(wallshearstress),' ',torch.max(wallshearstress))
    if verbose:  print('Wss_int: ',torch.min(Wss_int),' ',torch.max(Wss_int))
    WP_int = (WP_int*cell_length.reshape(-1, 1)).sum(axis = 0)
    if verbose: print('WP_int: ', torch.min(WP_int),' ',torch.max(WP_int))
    force = Wss_int - WP_int
    if verbose: print('force: ', torch.min(force),' ',torch.max(force))
    alpha = (angle*torch.pi/180)
    basis = torch.tensor([[torch.cos(alpha), torch.sin(alpha)], [-torch.sin(alpha), torch.cos(alpha)]]).to(data.y.dtype).to(device)
    #print(basis.type(),' ',force.type())
    #print(basis.get_device(),' ',force.get_device())
    force = force.to(basis.dtype)
    force_rot = basis@force
    if verbose: print('force_rot: ',torch.min(force_rot),' ',torch.max(force_rot))
    if verbose: print('Uinf: ',Uinf)
    #print(force_rot.get_device(), ' ',Uinf.get_device())
    coef = 2*force_rot/Uinf**2
    # coefs.append(coef.numpy())
    return coef,wallshearstress, surf_grad, surf_p
    
def Compute_coefficients_new(data_orig, airfoil_data_pkl, Uinf,angle,device = 'cuda:0'):
    #print(airfoil_data_pkl)
    # coefs = []
    verbose = False
    if verbose: print('New method:')
    data = data_orig.clone() # unnormalized

    Af = torch.load(airfoil_data_pkl) # Data(val=[|afpts|, 6], indices=[|afpts|]) , indices = airfoil cell index in data.y, val = [C,Cf,outward normal] all unnorm
    incident_cells = Af.indices
    Sfn_out = data.x[data.surf,4:]
    
    
    surf_grad = gradXY(data,fxy = data.y[:,:2])[data_orig.surf,:,:]
    if verbose: 
        print('surf_grad: ',torch.min(surf_grad),torch.max(surf_grad))
    surf_p = data.y[incident_cells,2]
    Sfn = -Sfn_out # inward
    cell_length = torch.sqrt(Af.val[:,4]**2 + Af.val[:,5]**2)
    # normal = data_orig.cf[:,:2]
    # cell_length = torch.sqrt(normal[:,0]**2+normal[:,1]**2)
    if verbose: print('cell_length: ',torch.min(cell_length),' ',torch.max(cell_length))
    if verbose: print('outward normal: ', torch.min(-Sfn),' ',torch.max(-Sfn)) 

    #Wss_pred = WallShearStress(surf_grad.numpy(), -Sfn.numpy())
    #print("wallshearstress_np: ",Wss_pred.min(),' ',Wss_pred.max())
    wallshearstress = WallShearStress_torch(surf_grad,-Sfn)
    #print("wallshearstress_torch: ",wallshearstress.shape, ' => ',torch.min(wallshearstress),' ',torch.max(wallshearstress))
    #wallshearstress = torch.from_numpy(Wss_pred)
    # if verbose: print('surf_p: ',np.sort(surf_p))
    if verbose: print('surfP: ',torch.min(surf_p),' ',torch.max(surf_p))
    WP_int = surf_p[:, None]*Sfn_out
    # if verbose: print(torch.min(Sfn_out),' ',torch.max(Sfn_out))
    Wss_int = (wallshearstress*cell_length.reshape(-1, 1)).sum(axis = 0)
    if verbose: print("wallshearstress: ",torch.min(wallshearstress),' ',torch.max(wallshearstress))
    if verbose:  print('Wss_int: ',torch.min(Wss_int),' ',torch.max(Wss_int))
    WP_int = (WP_int*cell_length.reshape(-1, 1)).sum(axis = 0)
    if verbose: print('WP_int: ', torch.min(WP_int),' ',torch.max(WP_int))
    force = Wss_int - WP_int
    if verbose: print('force: ', torch.min(force),' ',torch.max(force))
    alpha = angle*np.pi/180
    basis = torch.from_numpy(np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]]))
    force_rot = basis@force
    if verbose: print('force_rot: ',torch.min(force_rot),' ',torch.max(force_rot))
    if verbose: print('Uinf: ',Uinf)
    coef = 2*force_rot/Uinf**2
    # coefs.append(coef.numpy())
    return coef.numpy(),wallshearstress.numpy(), surf_grad.numpy(), surf_p.numpy()

def Results_test(device, models, hparams, coef_norm, n_test = 1, path_in = 'Dataset/', criterion = 'MSE', x_bl = [.2, .4, .6, .8], s = 'full_test',\
    use_saf = False, use_dsdf = False, manifest_dsdf = None,manifest_saf = None, manifest_fv = None, use_fv_subsamp = False, res_est = None):
    '''
    Compute criterion scores for the fields over the volume and the surface, and for the force coefficients. Also compute Spearman's correlation scores
    for the force coefficients and the relative error for the wall shear stress and the pressure over the airfoil. Outputs the true, the mean predicted
    and the std predicted integrated force coefficients, the true and mean predicted local force coefficients (at the surface of airfoils) and the true
    mean predicted boundary layers at chord x_bl.

    Args:
        device (str): Device on which you do the prediction.
        models (torch_geometric.nn.Module): List of models to predict with. It is a list of a list of different training of the same model.
            For example, it can be [model_MLP, model_GraphSAGE] where model_MLP is itself a list of the form [MLP_1, MLP_2].
        hparams (dict): Dictionnary of hyperparameters of the models.
        coef_norm (tuple): Tuple of the form (mean_in, mean_out, std_in, std_out) for the denormalization of the data.
        n_test (int, optional): Number of airfoils on which you want to infer (they will be drawn randomly in the given set). Default: ``3``
        path_in (str, optional): Path to find the manifest.json file. Default: ``"Dataset/"``
        criterion(str, optional): Criterion for the fields scores. Choose between MSE and MAE. Default: ``"MSE"``
        x_bl (list, optional): List of chord where the extract boundary layer prediction will be extracted. Default: ``[.2, .4, .6, .8]``
        s (str, optional): Dataset in which the simulation names are going to be sampled. Default: ``"full_test"``
    '''
    # Compute scores and all metrics for a 
    l2norm_velErr = lambda uv_pred,uv: torch.mean(torch.norm(uv-uv_pred,dim = 1))
    #OF_DATA_PATH = '/ntuzfs/naheed/wp/submission/AirfRANS/OF_dataset/'
    # OF_DATA_PATH = '/home1/OF_dataset/'
    sns.set()
    with open(path_in + 'manifest.json', 'r') as f:
        manifest = json.load(f)
    test_dataset = manifest[s]#[:1]
    # print(test_dataset[0])
    verbose = False
    test_dataset_unnorm = [torch.load(os.path.join('nsresd/',s+'.pkl')) for s in test_dataset]
    test_dataset_vtk, coef_norm = NSResDataset(test_dataset, norm = coef_norm, res_est = res_est)
    # test_dataset_unnorm = [torch.load(manifest_fv[s]) for s in tqdm(test_dataset)]
    test_loader = DataLoader(test_dataset_vtk, shuffle = False)

    # if manifest_fv is None:
    #     idx = random.sample(range(len(test_dataset)), k = n_test)
    #     idx.sort()
    #     test_dataset_vtk = Dataset(test_dataset, sample = None, coef_norm = coef_norm,\
    #         use_saf = use_saf, use_dsdf = use_dsdf, manifest_dsdf = manifest_dsdf,manifest_saf = manifest_saf)   
    #     test_loader = DataLoader(test_dataset_vtk, shuffle = False)
    # else:
    #     if use_fv_subsamp:  # New gnet models
    #         test_dataset_vtk, coef_norm = FVDatasetreplaceSAF(test_dataset, norm = coef_norm, manifest_fv = manifest_fv)
    #         test_dataset_unnorm = [torch.load(manifest_fv[s]) for s in tqdm(test_dataset)]
    #         # test_dataset_unnorm = FVDatasetreplaceSAF(test_dataset, norm = False, manifest_fv = manifest_fv) # Un-normalised test data
           
    #     else: # Old FVGraphsage
    #         test_dataset_vtk = FVDataset(test_dataset, coef_norm = coef_norm,manifest_fv=manifest_fv)   
    #     test_loader = DataLoader(test_dataset_vtk, shuffle = False)

    if criterion == 'MSE':
        criterion = nn.MSELoss(reduction = 'none')
    elif criterion == 'MAE':
        criterion = nn.L1Loss(reduction = 'none')

    scores_vol = []
    scores_surf = []
    scores_force = []
    scores_p = []
    scores_wss = []
    internals = []
    # airfoils = []
    # true_internals = []
    # true_airfoils = []
    times = []
    true_coefs = [] #EMPTY
    pred_coefs = [] #EMPTY
    print('len(models): ',len(models))
    # print('len(models[0]): ',len(models[0]))
    #EDITED
    peraf_surfvar_loss = []
    peraf_volvar_loss = []
    peraf_coefficient_loss = []
    peraf_velocity_loss = []
    scores_velocity = []
    scores_trlosses = []
    scores_contE = []
    scores_contE_GT = []
    errs = []
    for i in range(len(models[0])):
        model = [models[n][i] for n in range(len(models))]
        avg_loss_per_var = np.zeros((len(model), 4))
        avg_loss = np.zeros(len(models))
        avg_loss_surf_var = np.zeros((len(model), 4))
        avg_loss_vol_var = np.zeros((len(model), 4))
        avg_loss_surf = np.zeros(len(models))
        avg_loss_vol = np.zeros(len(models))
        avg_rel_err_force = np.zeros((len(models), 2))
        avg_loss_p = np.zeros((len(models)))
        avg_loss_wss = np.zeros((len(models), 2))
        internal = []
        airfoil = []
        pred_coef = []
        # EDITED
        velocity_err = np.zeros((len(models),1))
        train_losses = np.zeros((len(models),1))
        cont_errors = []
        contGT_errors = []
        mse_errors = []
        for j, data in enumerate(tqdm(test_loader,desc = 'test progress: ')):
            if verbose: print('test_dataset[j]: ',test_dataset[j])
            if verbose: print('data: ',data)
            if verbose: print('GT data.y: ',data.y[-5:,:])
            Uinf, angle = float(test_dataset[j].split('_')[2]), float(test_dataset[j].split('_')[3])            
            outs, tim, train_loss, train_loss_uv = Infer_test(device, model, hparams, data, coef_norm = coef_norm)
            airfoil_data_pkl = "surf_data/"+test_dataset[j]+'.pkl'
            incident_cells = torch.load(airfoil_data_pkl).indices
            if (verbose):
                print('outs: ',len(outs), ' outs[0]: ',outs[0].shape,' tim: ',tim.shape)
            times.append(tim)
            # intern = pv.read('Dataset/' + test_dataset[j] + '/' + test_dataset[j] + '_internal.vtu')
            # aerofoil = pv.read('Dataset/' + test_dataset[j] + '/' + test_dataset[j] + '_aerofoil.vtp')
            # tc = Compute_coefficients(data, \
            #     os.path.join('/home1/OF_dataset',test_dataset[j]), data.surf, Uinf, angle,isGt=True) # baseline Cl,Cd
            # face_data_pkl = 'surfinfo/'+test_dataset[j]+'.pkl'
            # face_data_pkl = torch.load(face_data_pkl)
            # face_data_pkl = torch.load()

            tc,twallstress,tsurfGrad, tairfoilsurfP = Compute_coefficients_new(test_dataset_unnorm[j].clone(),\
                                    airfoil_data_pkl, Uinf,angle,device='cpu')

            # print('GT (wallshearstress): ',torch.sum(torch.linalg.norm(twallstress,dim=0)))
            # tc, true_intern, true_airfoil = Compute_coefficients([intern], [aerofoil], data.surf, Uinf, angle, keep_vtk = True) # GT coefficients
            # tc, true_intern, true_airfoil = tc[0], true_intern[0], true_airfoil[0]
            # intern, aerofoil = Airfoil_test(intern, aerofoil, outs, coef_norm, data.surf)
            # pc, intern, aerofoil = Compute_coefficients(intern, aerofoil, data.surf, Uinf, angle, keep_vtk = True)
            pcoef = []
            pwallstress = []
            psurfP = []
            cont = []
            contGT = []
            mses = []
            for m in range(len(models)):
                # Following Airfrans guys function Airfoil_test, we set on-boundary cell u=  0,v = 0 & nut = 0
                out = outs[m].clone().cpu() #use out for cl,cd wheras use outs[m] for error calculation
                out = out*(coef_norm[3] + 1e-8) + coef_norm[2] # unnormalise u,v,p,nut (only for Cl,Cd calculation)
                
                # Use prior knowledge for Cl,Cd calculation
                out[data.surf,2] = out[incident_cells,2] #comment
                out[data.surf, :2] = torch.zeros_like(out[data.surf, :2]) # setting velocity on NNout[surf] to 0
                out[data.surf, 3] = torch.zeros_like(out[data.surf, 3]) # setting nut on NNout[surf] to 0
                #cont. error
                tmp_data = data.clone()
                contGT.append(continuity_error(tmp_data).mean().numpy())
                tmp_data.y = outs[m]
                cont.append(continuity_error(tmp_data).mean().numpy())
                # tmp_data = Data(x = data.x, edge_index = data.edge_index, \
                #     edge_attr = data.edge_attr, node_attr = data.node_attr,\
                #          y = out, surf = data.surf) # y is already unnormalized
                # pc = Compute_coefficients(tmp_data, \
                #     os.path.join('/home1/OF_dataset',test_dataset[j]), data.surf, Uinf, angle, isGt = False) # NN output Cl,Cd
                if (verbose):
                  print('GT data.y[data.surf] sorted = ',torch.sort(data.y[data.surf]))
                  print('before CL CD: outs[m][data.surf] = ',outs[m][data.surf])
                
                # pcl,pcd,pwst,pairfoilsurfP = Compute_coefficients(tmp_data, airfoil_data_pkl,Uinf,angle,device='cpu')
                # tmp_data = Data(x = test_dataset_unnorm[j].x, edge_index = test_dataset_unnorm[j].edge_index, \
                #     edge_attr = test_dataset_unnorm[j].edge_attr, node_attr = test_dataset_unnorm[j].node_attr,\
                #          y = out, surf = test_dataset_unnorm[j].surf)
                # # tmp_data.y = out 
                unnormdata = test_dataset_unnorm[j].clone()
                unnormdata.y = out
               
                pc,pwst,psurfGrad,pairfoilsurfP =  Compute_coefficients_new(unnormdata,airfoil_data_pkl, Uinf,angle,device='cpu')
                
                pcoef.append(pc)
                #print('NN output (Cl,Cd): ',pc)
                pwallstress.append(pwst)
                #psurfP.append(pairfoilsurfP) # pairfoilsurfP is un-normalized
                psurfP.append(out[data.surf,2].numpy())
                
                # We do not use prior knowledge as the paper didn't
                outs[m][data.surf,2] = outs[m][incident_cells,2] # Use normalized P to update surfP #uncomment
                #outs[m][data.surf,2] = (outs[m][incident_cells,2] -torch.tensor(coef_norm[2][None, 2]))*torch.ones_like(outs[m][data.surf,2])/(torch.tensor(coef_norm[3][None, 2]) + 1e-8) # Use normalized P to update surfP
                #outs[m][data.surf,:2] = -torch.tensor(coef_norm[2][None, :2])*torch.ones_like(outs[m][data.surf,:2])/(torch.tensor(coef_norm[3][None, :2]) + 1e-8) # 0
                #outs[m][data.surf,3] = -torch.tensor(coef_norm[2][3])*torch.ones_like(outs[m][data.surf,3])/(torch.tensor(coef_norm[3][3]) + 1e-8) # 0
                if (verbose):
                  print('after CL CD: outs[m][data.surf] = ',outs[m][data.surf])
            
            if i == 0:
                true_coefs.append(tc)
            pred_coef.append(pcoef)
            cont_errors.append(cont)
            contGT_errors.append(contGT)
            
            # if j in idx:
            #     internal.append(intern)
            #     airfoil.append(aerofoil)
            #     if i == 0:
            #         true_internals.append(true_intern)
            #         true_airfoils.append(true_airfoil)
            # print('---')
            # print(tc)
            # print(pcoef[0])
            # print(twallstress.shape)
            # print(pwallstress[0].shape)
            
            for n, out in enumerate(outs):
                loss_per_var = criterion(out, data.y).mean(dim = 0)
                loss = loss_per_var.mean()
                loss_surf_var = criterion(out[data.surf, :], data.y[data.surf, :]).mean(dim = 0)
                loss_vol_var = criterion(out[~data.surf, :], data.y[~data.surf, :]).mean(dim = 0)
                loss_surf = loss_surf_var.mean()
                loss_vol = loss_vol_var.mean()

                avg_loss_per_var[n] += loss_per_var.cpu().numpy()
                avg_loss[n] += loss.cpu().numpy()
                avg_loss_surf_var[n] += loss_surf_var.cpu().numpy()
                avg_loss_vol_var[n] += loss_vol_var.cpu().numpy()
                avg_loss_surf[n] += loss_surf.cpu().numpy()
                avg_loss_vol[n] += loss_vol.cpu().numpy()
                avg_rel_err_force[n] += rel_err(tc, pcoef[n])
                # print('twallstress.shape: ',twallstress.shape)
                # print('pwallstress[n].shape: ',pwallstress[n].shape)
                # print("rel_err(twallstress, pwallstress[n]): ", rel_err(twallstress, pwallstress[n]).shape)
                # print('mean: ',rel_err(twallstress, pwallstress[n]).mean(axis=0).shape)
                avg_loss_wss[n] += rel_err(twallstress, pwallstress[n]).mean(axis = 0)
                avg_loss_p[n] += rel_err(tairfoilsurfP, psurfP[n]).mean(axis = 0)
                #print('average relative error WSS', avg_loss_wss[n])
                #print('average relative error p_surf',avg_loss_p[n])
                #print('average relative error Surf grad: ', rel_err(tsurfGrad,psurfGrad).mean(axis=0))
                #EDITED
                # mses.append(train_loss.numpy())
                mses.append(train_loss_uv.numpy())
                train_losses += train_loss.numpy()
                velocity_err[n] += l2norm_velErr(out[:,:2], data.y[:,:2]).numpy()
                peraf_surfvar_loss.append(loss_surf_var.cpu().numpy())
                peraf_volvar_loss.append(loss_vol_var.cpu().numpy())
                peraf_coefficient_loss.append(rel_err(tc, pcoef[n]))
                
                
            mse_errors.append(mses)
        # internals.append(internal)
        # airfoils.append(airfoil)
        pred_coefs.append(pred_coef)        
        scores_contE.append(cont_errors)
        scores_contE_GT.append(contGT_errors)
        score_var = np.array(avg_loss_per_var)/len(test_loader)
        score = np.array(avg_loss)/len(test_loader)
        
        score_surf_var = np.array(avg_loss_surf_var)/len(test_loader)
        score_vol_var = np.array(avg_loss_vol_var)/len(test_loader)
        score_surf = np.array(avg_loss_surf)/len(test_loader)
        score_vol = np.array(avg_loss_vol)/len(test_loader)
        score_force = np.array(avg_rel_err_force)/len(test_loader)
        score_p = np.array(avg_loss_p)/len(test_loader)
        score_wss = np.array(avg_loss_wss)/len(test_loader)

        score = score_surf + score_vol
        scores_vol.append(score_vol_var)
        scores_surf.append(score_surf_var)
        scores_force.append(score_force)
        scores_p.append(score_p)
        scores_wss.append(score_wss)
        #EDITED
        scores_velocity.append(velocity_err/len(test_loader))
        scores_trlosses.append(train_losses/len(test_loader))
        errs.append(mse_errors)

    scores_vol = np.array(scores_vol)
    scores_surf = np.array(scores_surf)
    scores_force = np.array(scores_force)
    scores_p = np.array(scores_p)
    scores_wss = np.array(scores_wss)
    times = np.array(times)
    true_coefs = np.array(true_coefs)
    pred_coefs = np.array(pred_coefs) #EMPTY
    scores_contE = np.array(scores_contE)
    scores_contE_GT = np.array(scores_contE_GT)
    print('true_coefs.shape: ', true_coefs.shape)
    print('pred_coefs.shape: ', pred_coefs.shape)
    print('continuity error shape: ',scores_contE.shape)
    print('Continuity error (FVF)= ', scores_contE.mean(axis=1),  scores_contE.std(axis=1))
    print('Continuity error (GT)= ', scores_contE_GT.mean(axis=1),  scores_contE_GT.std(axis=1))
    errs = np.array(errs)
    # print(scores_contE)
    # print(scores_contE_GT)
    # print(errs)
    errs = errs.flatten()
    scores_contE = scores_contE.flatten()
    scores_contE_GT = scores_contE_GT.flatten()
    # print(errs)
    # print(scores_contE)
    #print(errs.shape)
    #print(scores_contE.shape)
    #print(scores_contE_GT.shape)
    
    #res = sc.stats.spearmanr(errs,scores_contE)
    #pvalue, stat = res.pvalue , res.statistic
    #fig, ax = plt.subplots(figsize=(10, 5))
    #plt.scatter(errs,scores_contE)    
    #plt.xlabel('MSE')
    #plt.ylabel('cont. Error')
    # plt.title('corr = ', scores)
    #plt.tight_layout()
    #plt.savefig('Corr.pdf')
    #plt.close(fig)
    pred_coefs_mean = pred_coefs.mean(axis = 0) #EMPTY
    pred_coefs_std = pred_coefs.std(axis = 0)#EMPTY
    #EDITED
    peraf_surfvar_loss = np.array(peraf_surfvar_loss)
    peraf_volvar_loss = np.array(peraf_volvar_loss)
    peraf_coefficient_loss = np.array(peraf_coefficient_loss)
    scores_velocity = np.array(scores_velocity)
    scores_trlosses = np.array(scores_trlosses)
    # print(scores_trlosses)
    spear_coefs = []
    #EDITED
    spear_pvalues = []
    for j in range(pred_coefs.shape[0]):
        spear_coef = []
        spear_p = []
        for k in range(pred_coefs.shape[2]):
            sd = sc.stats.spearmanr(true_coefs[:, 0], pred_coefs[j, :, k, 0])
            sl = sc.stats.spearmanr(true_coefs[:, 1], pred_coefs[j, :, k, 1])
            spear_drag = sd[0]
            spear_lift = sl[0]
            spear_coef.append([spear_drag, spear_lift])
            #EDITED
            spear_p.append([sd[1],sl[1]])
        spear_coefs.append(spear_coef)
    spear_coefs = np.array(spear_coefs)
    spear_pvalues = np.array(spear_p)

    with open('score.json', 'w') as f:
        json.dump(
            {   
                'mean_time': times.mean(axis = 0),
                'std_time': times.std(axis = 0),
                'mean_score_vol': scores_vol.mean(axis = 0),
                'std_score_vol': scores_vol.std(axis = 0),
                'mean_score_surf': scores_surf.mean(axis = 0),
                'std_score_surf': scores_surf.std(axis = 0),
                'mean_rel_p': scores_p.mean(axis = 0),
                'std_rel_p': scores_p.std(axis = 0),
                'mean_rel_wss': scores_wss.mean(axis = 0),
                'std_rel_wss': scores_wss.std(axis = 0),
                'mean_score_force': scores_force.mean(axis = 0),
                'std_score_force': scores_force.std(axis = 0),
                'spearman_coef_mean': spear_coefs.mean(axis = 0),
                'spearman_coef_std': spear_coefs.std(axis = 0),
                #EDITED average over models
                'mean_training_loss': scores_trlosses.mean(axis = 0),
                'mean_score_velocityL2': scores_velocity.mean(axis=0),
                'std_score_velocityL2': scores_velocity.std(axis=0),
                'spearman_pval_mean': spear_pvalues.mean(axis = 0),
                'spearman_pval_std': spear_pvalues.std(axis = 0),
                #EDITED
                'mean_score_vol_af': peraf_volvar_loss.mean(axis = 0),
                'std_score_vol_af': peraf_volvar_loss.std(axis = 0),
                'mean_score_surf_af': peraf_surfvar_loss.mean(axis = 0),
                'std_score_surf_af': peraf_surfvar_loss.std(axis = 0),
                'mean_score_force_af': peraf_coefficient_loss.mean(axis = 0),
                'std_score_force_af': peraf_coefficient_loss.std(axis = 0)
            }, f, indent = 4, cls = NumpyEncoder
        )
    for i in range(len(models[0])):
        # EDITED
        import pandas as pd
        pc = pred_coefs[i].reshape(-1,2)
        df = pd.DataFrame(np.stack((true_coefs[:,0],pc[:,0])).T\
                    , columns = ['GT','Pred'])
        fig, ax = plt.subplots(figsize=(100, 4))
        df.plot(y=["GT", "Pred"], kind="bar", rot=0, ax = ax)
        # ax.bar(range(pc.shape[0]), true_coefs[:, 0] - pc[:,0]) #plot Cd error
        plt.xlabel('test Airfoils')
        # plt.xticks(range(pc.shape[0]))
        plt.title('Cd')
        plt.tight_layout()
        plt.savefig('Cd_model#'+str(i)+'.pdf')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(100, 4))
        df = pd.DataFrame(np.stack((true_coefs[:,1],pc[:,1])).T\
                    , columns = ['GT','Pred'])
        df.plot(y=["GT", "Pred"], kind="bar", rot=0, ax = ax)
        # ax.bar(range(pc.shape[0]), true_coefs[:,1] - pc[:,1]) # Plot Cl error
        plt.xlabel('test Airfoils')
        # plt.xticks(range(pc.shape[0]))
        plt.title('Cl')
        # plt.title('Cl(GT) - Cl(pred)')
        plt.tight_layout()
        plt.savefig('Cl_model#'+str(i)+'.pdf')
        plt.close(fig)

    # fig, ax = plt.subplots(1, 2, figsize = (20, 10))
    # # ax[2].scatter(true_coefs[:, 1], true_coefs[:, 0], label = 'True', color = 'black', marker = 's')
    # model_name = ['MLP', 'GraphSAGE', 'PointNet', 'GUNet']
    # for l, model in enumerate(model_name):
    #     ax[0].errorbar(true_coefs[:, 0], pred_coefs_mean[:, l, 0], yerr = pred_coefs_std[:, l, 0], fmt = 'x', capsize = 3, label = model)
    #     ax[1].errorbar(true_coefs[:, 1], pred_coefs_mean[:, l, 1], yerr = pred_coefs_std[:, l, 1], fmt = 'x', capsize = 3, label = model)
    #     # ax[2].errorbar(pred_coefs_mean[:, l, 1], pred_coefs_mean[:, l, 0], xerr = pred_coefs_std[:, l, 1], yerr = pred_coefs_std[:, l, 0], fmt = 'x', capsize = 3, label = model)
    # ax[0].set_xlabel('True ' + r'$C_D$')
    # ax[0].set_ylabel('Predicted ' + r'$C_D$')
    # ax[1].set_xlabel('True ' + r'$C_L$')
    # ax[1].set_ylabel('Predicted ' + r'$C_L$')
    # # ax[2].set_xlabel(r'$C_L$')
    # # ax[2].set_ylabel(r'$C_D$')
    # ax[0].legend(loc = 'best')
    # ax[1].legend(loc = 'best')
    # # ax[2].legend(loc = 'best')
    # fig.savefig('metrics/coefs.png', bbox_inches = 'tight', dpi = 150)

    surf_coefs = []
    true_surf_coefs = []
    bls = []
    true_bls = []
    #print(len(true_internals),' ',type(true_internals[0]))
    
    # for i in range(len(internals[0])):
    #     aero_name = test_dataset[idx[i]]
    #     true_internal = true_internals[i]
    #     true_airfoil = true_airfoils[i]
    #     surf_coef = []
    #     bl = []
    #     for j in range(len(internals[0][0])):
    #         internal_mean, airfoil_mean = Airfoil_mean([internals[k][i][j] for k in range(len(internals))], [airfoils[k][i][j] for k in range(len(airfoils))])
    #         internal_mean.cell_data['Error_ux'] = internal_mean.cell_data['U'][:,0] - true_internal.cell_data['U'][:,0]
    #         internal_mean.cell_data['Error_uy'] = internal_mean.cell_data['U'][:,1] - true_internal.cell_data['U'][:,1]
    #         internal_mean.cell_data['Error_p'] = internal_mean.cell_data['p'] - true_internal.cell_data['p']
    #         internal_mean.cell_data['ErrorR_ux'] = np.abs((internal_mean.cell_data['U'][:,0] - true_internal.cell_data['U'][:,0])/true_internal.cell_data['U'][:,0])
    #         internal_mean.cell_data['ErrorR_uy'] = np.abs((internal_mean.cell_data['U'][:,1] - true_internal.cell_data['U'][:,1])/ true_internal.cell_data['U'][:,1])
    #         internal_mean.cell_data['ErrorR_p'] = np.abs((internal_mean.cell_data['p'] - true_internal.cell_data['p'])/true_internal.cell_data['p'])

    #         internal_mean.cell_data['ErrorAb_ux'] = np.abs(internal_mean.cell_data['U'][:,0] - true_internal.cell_data['U'][:,0])
    #         internal_mean.cell_data['ErrorAb_uy'] = np.abs(internal_mean.cell_data['U'][:,1] - true_internal.cell_data['U'][:,1])
    #         internal_mean.cell_data['ErrorAb_p'] = np.abs(internal_mean.cell_data['p'] - true_internal.cell_data['p'])
    #         internal_mean.save(test_dataset[idx[i]] + '_' + str(j) + '.vtu')
    #         #print(type(internal_mean),' ',type(airfoil_mean))
    #         surf_coef.append(np.array(metrics_NACA.surface_coefficients(airfoil_mean, aero_name)))
    #         b = []
    #         for x in x_bl:
    #             b.append(np.array(metrics_NACA.boundary_layer(airfoil_mean, internal_mean, aero_name, x)))
    #         bl.append(np.array(b))
    #     true_surf_coefs.append(np.array(metrics_NACA.surface_coefficients(true_airfoil, aero_name)))
    #     true_bl = []
    #     for x in x_bl:
    #         true_bl.append(np.array(metrics_NACA.boundary_layer(true_airfoil, true_internal, aero_name, x)))
    #     true_bls.append(np.array(true_bl))
    #     surf_coefs.append(np.array(surf_coef))
    #     bls.append(np.array(bl))

    true_bls = np.array(true_bls)
    bls = np.array(bls)

    return true_coefs, pred_coefs_mean, pred_coefs_std, true_surf_coefs, surf_coefs, true_bls, bls
