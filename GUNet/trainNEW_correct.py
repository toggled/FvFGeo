import matplotlib as mpl
mpl.use('Agg')
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time, json

import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DataParallel
from torch_geometric.data import DataListLoader

from torch_geometric.utils import add_self_loops #ADDED: FV

import metrics

import os

from tqdm import tqdm

def get_nb_trainable_params(model):
   '''
   Return the number of trainable parameters
   '''
   model_parameters = filter(lambda p: p.requires_grad, model.parameters())
   return sum([np.prod(p.size()) for p in model_parameters])

def train(device, model, train_loader, optimizer, scheduler, scaler, criterion='MSE', reg=1, half_prec=True):
    #half_prec = False #False #
    model.train()
    avg_loss_per_var = torch.zeros(4, device = device)
    avg_loss = 0
    avg_loss_surf_var = torch.zeros(4, device = device)
    avg_loss_vol_var = torch.zeros(4, device = device)
    avg_loss_surf = 0
    avg_loss_vol = 0
    iter = 0
    
    for data in train_loader:          
        optimizer.zero_grad()
        data_clone = data.clone()
        data_clone = data_clone.to(device)
        if half_prec:
            with torch.cuda.amp.autocast(dtype=torch.float16): #ADDED: half pres
                out = model(data_clone)
                targets = data_clone.y

                if criterion == 'MSE' or criterion == 'MSE_weighted':
                    crit = nn.MSELoss(reduction = 'none')
                elif criterion == 'MAE':
                    crit = nn.L1Loss(reduction = 'none')
                loss_per_var = crit(out, targets).mean(dim = 0)
                total_loss = loss_per_var.mean()
                loss_surf_var = crit(out[data_clone.surf, :], targets[data_clone.surf, :]).mean(dim = 0)
                loss_vol_var = crit(out[~data_clone.surf, :], targets[~data_clone.surf, :]).mean(dim = 0)
                loss_surf = loss_surf_var.mean()
                loss_vol = loss_vol_var.mean() 
            
            if criterion == 'MSE_weighted':            
                #(loss_vol + reg*loss_surf).backward()     
                scaler.scale((loss_vol + reg*loss_surf)).backward() #ADDED: half pres
                
            else:
                #total_loss.backward()
                scaler.scale(total_loss).backward() #ADDED: half pres
            nn.utils.clip_grad_norm_(model.parameters(), 5) #if nan loss!
            # nn.utils.clip_grad_norm_(model.parameters(), 1) #for GUNetGCN_big
            scaler.step(optimizer) #ADDED: half pres
            scaler.update() #ADDED: half pres
        else:
            out = model(data_clone)
            targets = data_clone.y

            if criterion == 'MSE' or criterion == 'MSE_weighted':
                crit = nn.MSELoss(reduction = 'none')
            elif criterion == 'MAE':
                crit = nn.L1Loss(reduction = 'none')
            loss_per_var = crit(out, targets).mean(dim = 0)
            total_loss = loss_per_var.mean()
            loss_surf_var = crit(out[data_clone.surf, :], targets[data_clone.surf, :]).mean(dim = 0)
            loss_vol_var = crit(out[~data_clone.surf, :], targets[~data_clone.surf, :]).mean(dim = 0)
            loss_surf = loss_surf_var.mean()
            loss_vol = loss_vol_var.mean() 
            if criterion == 'MSE_weighted':            
                (loss_vol + reg*loss_surf).backward()     
                # scaler.scale((loss_vol + reg*loss_surf)).backward() #ADDED: half pres
                
            else:
                total_loss.backward()
                # scaler.scale(total_loss).backward()
            optimizer.step()

            
        scheduler.step()
        avg_loss_per_var += loss_per_var
        avg_loss += total_loss
        avg_loss_surf_var += loss_surf_var
        avg_loss_vol_var += loss_vol_var
        avg_loss_surf += loss_surf
        avg_loss_vol += loss_vol 
        iter += 1

    return avg_loss.cpu().data.numpy()/iter, avg_loss_per_var.cpu().data.numpy()/iter, avg_loss_surf_var.cpu().data.numpy()/iter, avg_loss_vol_var.cpu().data.numpy()/iter, \
            avg_loss_surf.cpu().data.numpy()/iter, avg_loss_vol.cpu().data.numpy()/iter

def DPtrain(model, train_loader, optimizer, scheduler, scaler, criterion = 'MSE', reg = 1):
    device = 'cuda:0'
    model.train()
    avg_loss_per_var = torch.zeros(4,device = device)
    avg_loss = 0
    avg_loss_surf_var = torch.zeros(4,device = device)
    avg_loss_vol_var = torch.zeros(4,device = device)
    avg_loss_surf = 0
    avg_loss_vol = 0
    iter = 0

    for batch_list in train_loader:
        # print('len(batch_list): ',len(batch_list))
        # data_clone = batch_list.clone()
        # data_clone = data_clone.to(device)          
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(dtype=torch.float16): #ADDED: half pres
            out = model(batch_list)
            # print(out.size())
            targets = torch.cat([data_clone.y for data_clone in batch_list],0).to(device)
            if criterion == 'MSE' or criterion == 'MSE_weighted':
                crit = nn.MSELoss(reduction = 'none')
            elif criterion == 'MAE':
                crit = nn.L1Loss(reduction = 'none')
            loss_per_var = crit(out, targets).mean(dim = 0)
            total_loss = loss_per_var.mean()

            surf_bools = torch.cat([data_clone.surf for data_clone in batch_list],0).to(device)
            # surf_out = torch.cat([out2[data_clone.surf,:] for data_clone, out2 in zip(batch_list,out.split(8))],0)
            # surf_target = torch.cat([data_clone.y[data_clone.surf, :] for data_clone in batch_list],0)
            loss_surf_var = crit(out[surf_bools, :], targets[surf_bools, :]).mean(dim = 0)

            # internal_out = torch.cat([out2[~data_clone.surf,:] for data_clone, out2 in zip(batch_list,out.split(8))],0)
            # internal_target = torch.cat([data_clone.y[~data_clone.surf, :] for data_clone in batch_list],0)
            loss_vol_var = crit(out[~surf_bools, :], targets[~surf_bools, :]).mean(dim = 0)
            # loss_surf_var = crit(surf_out,surf_target).mean(dim=0)
            # loss_vol_var = crit(internal_out,internal_target).mean(dim=0)

            loss_surf = loss_surf_var.mean()
            loss_vol = loss_vol_var.mean() 
        
        if criterion == 'MSE_weighted':            
            #(loss_vol + reg*loss_surf).backward() #EDITED: half pres
            scaler.scale((loss_vol + reg*loss_surf)).backward() #ADDED: half pres
        else:
            #total_loss.backward() #EDITED: half pres
            scaler.scale(total_loss).backward() #ADDED: half pres
        nn.utils.clip_grad_norm_(model.parameters(), 5) #if nan loss!
        scaler.step(optimizer) #ADDED: half pres
        scaler.update() #ADDED: half pres
        
        #optimizer.step() #EDITED: half pres
        scheduler.step()
        avg_loss_per_var += loss_per_var
        avg_loss += total_loss
        avg_loss_surf_var += loss_surf_var
        avg_loss_vol_var += loss_vol_var
        avg_loss_surf += loss_surf
        avg_loss_vol += loss_vol 
        iter += len(batch_list)
        del targets
    # print('iter: ',iter)
    return avg_loss.cpu().data.numpy()/iter, avg_loss_per_var.cpu().data.numpy()/iter, avg_loss_surf_var.cpu().data.numpy()/iter, avg_loss_vol_var.cpu().data.numpy()/iter, \
            avg_loss_surf.cpu().data.numpy()/iter, avg_loss_vol.cpu().data.numpy()/iter


@torch.no_grad()
def test(device, model, test_loader, criterion = 'MSE'):
    model.eval()
    avg_loss_per_var = np.zeros(4)
    avg_loss = 0
    avg_loss_surf_var = np.zeros(4)
    avg_loss_vol_var = np.zeros(4)
    avg_loss_surf = 0
    avg_loss_vol = 0
    iter = 0

    for data in test_loader:        
        data_clone = data.clone()
        #print('data: ',data_clone)
        #print('device: ',device)
        data_clone = data_clone.to(device)
        out = model(data_clone)       

        targets = data_clone.y
        if criterion == 'MSE' or 'MSE_weighted':
            crit = nn.MSELoss(reduction = 'none')
        elif criterion == 'MAE':
            crit = nn.L1Loss(reduction = 'none')

        loss_per_var = crit(out, targets).mean(dim = 0)
        loss = loss_per_var.mean()
        loss_surf_var = crit(out[data_clone.surf, :], targets[data_clone.surf, :]).mean(dim = 0)
        loss_vol_var = crit(out[~data_clone.surf, :], targets[~data_clone.surf, :]).mean(dim = 0)
        loss_surf = loss_surf_var.mean()
        loss_vol = loss_vol_var.mean()  

        avg_loss_per_var += loss_per_var.cpu().numpy()
        avg_loss += loss.cpu().numpy()
        avg_loss_surf_var += loss_surf_var.cpu().numpy()
        avg_loss_vol_var += loss_vol_var.cpu().numpy()
        avg_loss_surf += loss_surf.cpu().numpy()
        avg_loss_vol += loss_vol.cpu().numpy()  
        iter += 1
    
    return avg_loss/iter, avg_loss_per_var/iter, avg_loss_surf_var/iter, avg_loss_vol_var/iter, avg_loss_surf/iter, avg_loss_vol/iter

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main(device, train_dataset, val_dataset, Net, hparams, path, criterion = 'MSE', reg = 1, val_iter = 10, name_mod = 'GraphSAGE', val_sample = False, use_saf = False,use_dsdf = False,dp = False):
    '''
        Args:
        device (str): device on which you want to do the computation.
        train_dataset (list): list of the data in the training set.
        val_dataset (list): list of the data in the validation set.
        Net (class): network to train.
        hparams (dict): hyper parameters of the network.
        path (str): where to save the trained model and the figures.
        criterion (str, optional): chose between 'MSE', 'MAE', and 'MSE_weigthed'. The latter is the volumetric MSE plus the surface MSE computed independently. Default: 'MSE'.
        ref (float, optional): weigth for the surface loss when criterion is 'MSE_weighted'. Default: 1.
        val_iter (int, optional): number of epochs between each validation step. Default: 10.
    '''

    # model = Net.to(device)
    num_gpus = 8
    model = DataParallel(Net).to(device) if dp else Net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = hparams['lr'],
            total_steps = (len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'],
        )
    val_loader = DataLoader(val_dataset, batch_size = 1)
    start = time.time()

    train_loss_surf_list = []
    train_loss_vol_list = []
    loss_surf_var_list = []
    loss_vol_var_list = []
    val_surf_list = []
    val_vol_list = []
    val_surf_var_list = []
    val_vol_var_list = []

    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    scaler = torch.cuda.amp.GradScaler() #ADDED: half pres
    #early_stop = False #ADDED:early stopping
    #val_loss_list = [] #ADDED:early stopping
    for epoch in pbar_train:
        train_dataset_sampled = []
        for data in train_dataset:
            data_sampled = data.clone()
            if hparams['subsampling']>0: #if subsampling
                idx = random.sample(range(data_sampled.x.size(0)), hparams['subsampling'])
                idx = torch.tensor(idx)
                #print(data_sampled)
                data_sampled.pos = data_sampled.pos[idx]
                if hasattr(data_sampled, 'node_attr'):
                    data_sampled.node_attr = data_sampled.node_attr[idx]
                data_sampled.x = data_sampled.x[idx]
                data_sampled.y = data_sampled.y[idx]
                data_sampled.surf = data_sampled.surf[idx]
                if (use_saf):
                    data_sampled.saf = data_sampled.saf[idx]
                if (use_dsdf):
                    # for i in range(data_sampled.dsdf.size(0)):
                    # data_sampled.dsdf[i] = data_sampled.dsdf[i][idx]
                    data_sampled.dsdf = data_sampled.dsdf[:,idx]
                if name_mod != 'PointNet' and name_mod != 'MLP':
                    data_sampled.edge_index = nng.radius_graph(x = data_sampled.pos.to(device), r = hparams['r'], loop = True, max_num_neighbors = int(hparams['max_neighbors'])).cpu()
            else: #if no subsampling done, for FVGraphSAGE
                if name_mod != 'PointNet' and name_mod != 'MLP':
                    data_sampled.edge_index, data_sampled.edge_attr = add_self_loops(data_sampled.edge_index, data_sampled.edge_attr, fill_value='mean')
                    data_sampled.edge_indexA2, data_sampled.edge_attrA2 = add_self_loops(data_sampled.edge_index, data_sampled.edge_attr, fill_value='mean')
            
            train_dataset_sampled.append(data_sampled)
        # train_loader = DataLoader(train_dataset_sampled, batch_size = hparams['batch_size'], shuffle = True)
        if dp:
            train_loader = DataListLoader(train_dataset_sampled,batch_size = hparams['batch_size']*num_gpus, shuffle = True)
        else:
            train_loader = DataLoader(train_dataset_sampled, batch_size = hparams['batch_size'], shuffle = True)
        # for data in train_dataset_sampled:
        #     assert(data.x.size(0)==data.saf.size(0))
        del(train_dataset_sampled)
        if not dp:
            _, _, loss_surf_var, loss_vol_var, loss_surf, loss_vol = train(device, model, train_loader, optimizer, lr_scheduler, scaler, criterion, reg = reg, half_prec=hparams['half_prec'])
        else:
            _, _, loss_surf_var, loss_vol_var, loss_surf, loss_vol = DPtrain(model, train_loader, optimizer, lr_scheduler, scaler, criterion, reg = reg)
        del(train_loader)
        train_loss = loss_surf + loss_vol
        train_loss_surf_list.append(loss_surf)
        train_loss_vol_list.append(loss_vol)
        loss_surf_var_list.append(loss_surf_var)
        loss_vol_var_list.append(loss_vol_var)
  
        if val_iter is not None:
            if epoch%val_iter == val_iter - 1 or epoch == 0:
                if val_sample:
                    val_surf_vars, val_vol_vars, val_surfs, val_vols = [], [], [], []
                    for i in range(20):
                        val_dataset_sampled = []
                        for data in val_dataset:
                            data_sampled = data.clone()
                            
                            if hparams['subsampling']>0:
                                idx = random.sample(range(data_sampled.x.size(0)), hparams['subsampling'])
                                idx = torch.tensor(idx)

                                data_sampled.pos = data_sampled.pos[idx]
                                if hasattr(data_sampled, 'node_attr'):
                                    data_sampled.node_attr = data_sampled.node_attr[idx]
                                data_sampled.x = data_sampled.x[idx]
                                data_sampled.y = data_sampled.y[idx]
                                data_sampled.surf = data_sampled.surf[idx]
                                if (use_saf):
                                    data_sampled.saf = data_sampled.saf[idx]
                                if (use_dsdf):
                                    # for i in range(data_sampled.dsdf.size(0)):
                                    data_sampled.dsdf = data_sampled.dsdf[:,idx]
                                if name_mod != 'PointNet' and name_mod != 'MLP':
                                    data_sampled.edge_index = nng.radius_graph(x = data_sampled.pos.to(device), r = hparams['r'], loop = True, max_num_neighbors = int(hparams['max_neighbors'])).cpu()
                            else: #if no subsampling done, for FVGraphSAGE
                                if name_mod != 'PointNet' and name_mod != 'MLP':
                                    data_sampled.edge_index, data_sampled.edge_attr = add_self_loops(data_sampled.edge_index, data_sampled.edge_attr, fill_value='mean')
                                    data_sampled.edge_indexA2, data_sampled.edge_attrA2 = add_self_loops(data_sampled.edge_index, data_sampled.edge_attr, fill_value='mean')
                                
                            
                            val_dataset_sampled.append(data_sampled)
                        if dp:
                            val_loader = DataListLoader(val_dataset_sampled, batch_size = num_gpus, shuffle = True)
                        else:
                            val_loader = DataLoader(val_dataset_sampled, batch_size = 1, shuffle = True)
                        #val_loader = DataLoader(val_dataset_sampled, batch_size = num_gpus, shuffle = True)
                        del(val_dataset_sampled)
                        # print('[]: ',[d for d in val_loader])

                        #val_loss, _, val_surf_var, val_vol_var, val_surf, val_vol = test(device, model.module, val_loader, criterion)
                        val_loss, _, val_surf_var, val_vol_var, val_surf, val_vol = test(device, model, val_loader, criterion)
                        del(val_loader)
                        val_surf_vars.append(val_surf_var)
                        val_vol_vars.append(val_vol_var)
                        val_surfs.append(val_surf)
                        val_vols.append(val_vol)
                    val_surf_var = np.array(val_surf_vars).mean(axis = 0)
                    val_vol_var = np.array(val_vol_vars).mean(axis = 0)
                    val_surf = np.array(val_surfs).mean(axis = 0)
                    val_vol = np.array(val_vols).mean(axis = 0)
                else:
                    val_loss, _, val_surf_var, val_vol_var, val_surf, val_vol = test(device, model, val_loader, criterion)
                val_loss = val_surf + val_vol
                val_surf_list.append(val_surf)
                val_vol_list.append(val_vol)
                val_surf_var_list.append(val_surf_var)
                val_vol_var_list.append(val_vol_var)
                #val_loss_list.append(val_loss) #ADDED: early 
                #print('list length: ',len(val_loss_list))
                # if len(val_loss_list)>2 and val_loss_list[-1]>=val_loss_list[-2]:
                #     #print('it happened!')
                #     if val_loss_list[-1]>=val_loss_list[-3]:
                #         early_stop = True
                #     else:
                #         val_iter = max(val_iter//2,1) #ADDED####
                # else:
                # models = [model]
                # torch.save(model,args.model+"_"+str(epoch))
                        
                pbar_train.set_postfix(train_loss = train_loss, loss_surf = loss_surf, val_loss = val_loss, val_surf = val_surf)
            else:
                pbar_train.set_postfix(train_loss = train_loss, loss_surf = loss_surf, val_loss = val_loss, val_surf = val_surf)
        else:
            pbar_train.set_postfix(train_loss = train_loss, loss_surf = loss_surf)
        
        # if early_stop == True: #ADDED: early stopping
        #     break #ADDED####
        
        if epoch%val_iter == val_iter - 1:
            path_iter = path+str(epoch)+'/'
            os.system('mkdir -p '+path_iter)
            np_loss_surf_var_list = np.array(loss_surf_var_list)
            np_loss_vol_var_list = np.array(loss_vol_var_list)
            np_val_surf_var_list = np.array(val_surf_var_list)
            np_val_vol_var_list = np.array(val_vol_var_list)

            end = time.time()
            time_elapsed = end - start
            params_model = get_nb_trainable_params(model).astype('float')
            print('Number of parameters:', params_model)
            print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
            torch.save(model, path_iter + 'model')

            sns.set()
            fig_train_surf, ax_train_surf = plt.subplots(figsize = (20, 5))
            ax_train_surf.plot(train_loss_surf_list, label = 'Mean loss')
            ax_train_surf.plot(np_loss_surf_var_list[:, 0], label = r'$v_x$ loss'); ax_train_surf.plot(np_loss_surf_var_list[:, 1], label = r'$v_y$ loss')
            ax_train_surf.plot(np_loss_surf_var_list[:, 2], label = r'$p$ loss'); ax_train_surf.plot(np_loss_surf_var_list[:, 3], label = r'$\nu_t$ loss')
            ax_train_surf.set_xlabel('epochs')
            ax_train_surf.set_yscale('log')
            ax_train_surf.set_title('Train losses over the surface')
            ax_train_surf.legend(loc = 'best')
            fig_train_surf.savefig(path_iter + 'train_loss_surf.png', dpi = 150, bbox_inches = 'tight')

            fig_train_vol, ax_train_vol = plt.subplots(figsize = (20, 5))
            ax_train_vol.plot(train_loss_vol_list, label = 'Mean loss')
            ax_train_vol.plot(np_loss_vol_var_list[:, 0], label = r'$v_x$ loss'); ax_train_vol.plot(np_loss_vol_var_list[:, 1], label = r'$v_y$ loss')
            ax_train_vol.plot(np_loss_vol_var_list[:, 2], label = r'$p$ loss'); ax_train_vol.plot(np_loss_vol_var_list[:, 3], label = r'$\nu_t$ loss')
            ax_train_vol.set_xlabel('epochs')
            ax_train_vol.set_yscale('log')
            ax_train_vol.set_title('Train losses over the volume')
            ax_train_vol.legend(loc = 'best')
            fig_train_vol.savefig(path_iter + 'train_loss_vol.png', dpi = 150, bbox_inches = 'tight')

            if val_iter is not None:
                fig_val_surf, ax_val_surf = plt.subplots(figsize = (20, 5))
                ax_val_surf.plot(val_surf_list, label = 'Mean loss')
                ax_val_surf.plot(np_val_surf_var_list[:, 0], label = r'$v_x$ loss'); ax_val_surf.plot(np_val_surf_var_list[:, 1], label = r'$v_y$ loss')
                ax_val_surf.plot(np_val_surf_var_list[:, 2], label = r'$p$ loss'); ax_val_surf.plot(np_val_surf_var_list[:, 3], label = r'$\nu_t$ loss')
                ax_val_surf.set_xlabel('epochs')
                ax_val_surf.set_yscale('log')
                ax_val_surf.set_title('Validation losses over the surface')
                ax_val_surf.legend(loc = 'best')
                fig_val_surf.savefig(path_iter + 'val_loss_surf.png', dpi = 150, bbox_inches = 'tight')

                fig_val_vol, ax_val_vol = plt.subplots(figsize = (20, 5))
                ax_val_vol.plot(val_vol_list, label = 'Mean loss')
                ax_val_vol.plot(np_val_vol_var_list[:, 0], label = r'$v_x$ loss'); ax_val_vol.plot(np_val_vol_var_list[:, 1], label = r'$v_y$ loss')
                ax_val_vol.plot(np_val_vol_var_list[:, 2], label = r'$p$ loss'); ax_val_vol.plot(np_val_vol_var_list[:, 3], label = r'$\nu_t$ loss')
                ax_val_vol.set_xlabel('epochs')
                ax_val_vol.set_yscale('log')
                ax_val_vol.set_title('Validation losses over the volume')
                ax_val_vol.legend(loc = 'best')
                fig_val_vol.savefig(path_iter + 'val_loss_vol.png', dpi = 150, bbox_inches = 'tight');

                if val_iter is not None:
                    with open(path_iter + 'log.json', 'w') as f:
                        json.dump(
                            {
                                'regression': 'Total',
                                'loss': 'MSE',
                                'nb_parameters': params_model,
                                'time_elapsed': time_elapsed,
                                'hparams': hparams,
                                'train_loss_surf': train_loss_surf_list[-1],
                                'train_loss_surf_var': np_loss_surf_var_list[-1],
                                'train_loss_vol': train_loss_vol_list[-1],
                                'train_loss_vol_var': np_loss_vol_var_list[-1],
                                'val_loss_surf': val_surf_list[-1],
                                'val_loss_surf_var': np_val_surf_var_list[-1],
                                'val_loss_vol': val_vol_list[-1],
                                'val_loss_vol_var': np_val_vol_var_list[-1],
                            }, f, indent = 12, cls = NumpyEncoder
                        )

    return model
