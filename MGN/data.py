#@title EDITED DATA.py script to have saf, dSDF, FV attr, and generalise geom 

import os
import pickle
# from pathlib import Path
import json
import numpy as np
import math
import torch
from torch_geometric.data import Data, Batch, Dataset
import pyvista as pv
from tqdm import tqdm
# from process_vtk import compute_edge_indices, compute_markers_numpy
import torch_geometric.transforms as T

class MeshAirfoilDataset(Dataset): # '../cfd-gcn_geom/Fine_vtk' (police) # '/ntuzfs/data/cfdgcnRR/Fine_vtk/' (neptune)
    def __init__(self, root='/ntuzfs/data/cfdgcnRR/Fine_vtk/', data_type='scarce', mode='train', fvdata=False):
        self.fvdata = fvdata
        # with open('manifest2.json', 'r') as f:
        with open('manifest.json', 'r') as f:
            manifest = json.load(f)

        if data_type=='scarce':
            tr = manifest['scarce_train']
            te = manifest['full_test']
        else:
            tr = manifest[data_type+'_train']
            te = manifest[data_type+'_test']
            
            
        if mode == 'train' or mode == 'val':
            manifest_train = tr
            n = int(.1*len(manifest_train))
            train_dataset = manifest_train[:-n] #manifest['scarce_train'][:-n]#[0:10]
            val_dataset = manifest_train[-n:]#[0:10]
            print('train length: ',len(train_dataset))
            print('val length: ',len(val_dataset))
        if mode == 'test':
            test_dataset = te #manifest['full_test']#[0:10]
            
        
        self.mode = mode
        self.data_dir = root
        if mode=='train' or mode=='val':
            if mode=='train':
                self.file_list = train_dataset
            else:
                self.file_list = val_dataset
            if os.path.isfile('normfactor_train.pkl'):
                (self.mean_in, self.std_in, self.mean_out, self.std_out)  = pickle.load(open('normfactor_train.pkl','rb'))
                self.normalization_factors = (self.mean_in, self.std_in, self.mean_out, self.std_out)
            else:
                for k,s in tqdm(enumerate(self.file_list),desc='Computing mean coeff: '):
                    internal = pv.read(os.path.join(self.data_dir, s + '/' + s + '_internal.vtu'))
                    node_markers = np.load(os.path.join(self.data_dir,s+'/'+'markers.npy')).reshape(-1,1)
                    fields = internal.point_data['U'][:, :2]
                    aoa,mach = self.get_params_from_name(s)
                    aoa,mach = np.ones_like(node_markers)*aoa, np.ones_like(node_markers)*mach
                    init = np.hstack((node_markers,aoa,mach))
                    # print('init shape: ',init.shape)
                    if k == 0:
                        old_length = init.shape[0]
                        mean_in = init.mean(axis = 0, dtype = np.double)
                        mean_out = fields.mean(axis = 0, dtype = np.double)
                    else:
                        new_length = old_length + init.shape[0]
                        mean_in += (init.sum(axis = 0, dtype = np.double) - init.shape[0]*mean_in)/new_length
                        mean_out += (fields.sum(axis = 0, dtype = np.double) - init.shape[0]*mean_out)/new_length
                        old_length = new_length 

                mean_in = mean_in.astype(np.single)
                mean_out = mean_out.astype(np.single)
                # print('mean_in = ',mean_in.shape)
                for k,s in tqdm(enumerate(self.file_list),desc = 'Computing std.dev coeff: '):
                    internal = pv.read(os.path.join(self.data_dir, s + '/' + s + '_internal.vtu'))
                    node_markers = np.load(os.path.join(self.data_dir,s+'/'+'markers.npy')).reshape(-1,1)
                    fields = internal.point_data['U'][:, :2]
                    aoa,mach = self.get_params_from_name(s)
                    aoa,mach = np.ones_like(node_markers)*aoa, np.ones_like(node_markers)*mach
                    init = np.hstack((node_markers,aoa,mach))
                    if k == 0:
                        old_length = init.shape[0]
                        std_in = ((init - mean_in)**2).sum(axis = 0, dtype = np.double)/old_length
                        std_out = ((fields - mean_out)**2).sum(axis = 0, dtype = np.double)/old_length
                    else:
                        new_length = old_length + init.shape[0]
                        std_in += (((init - mean_in)**2).sum(axis = 0, dtype = np.double) -init.shape[0]*std_in)/new_length
                        std_out += (((fields - mean_out)**2).sum(axis = 0, dtype = np.double) - init.shape[0]*std_out)/new_length
                        old_length = new_length
                std_in = np.sqrt(std_in).astype(np.single)
                std_out = np.sqrt(std_out).astype(np.single)

                self.normalization_factors = (mean_in, std_in, mean_out, std_out) 
                self.mean_in, self.std_in, self.mean_out, self.std_out = self.normalization_factors
                with open('normfactor_train.pkl','wb') as f:
                    pickle.dump(self.normalization_factors,f)
        if mode=='test':
            self.file_list = test_dataset
            with open('normfactor_train.pkl','rb') as f:
                self.mean_in, self.std_in, self.mean_out, self.std_out  = pickle.load(open('normfactor_train.pkl','rb'))
                self.normalization_factors = (self.mean_in, self.std_in, self.mean_out, self.std_out)
        self.len = len(self.file_list)
        # print('norm coeff: ',self.normalization_factors)
        super().__init__(root)

    def indices(self):
        return range(self.len)
        
    def len(self):
        return self.len

    def get(self, idx):
        transformer = T.Compose([T.Cartesian(norm=False), T.Distance(norm=False)])
        prec = 16
        s = self.file_list[idx]
        internal = pv.read(os.path.join(self.data_dir, s + '/' + s + '_internal.vtu'))
        # aerofoil = pv.read(os.path.join(self.data_dir, s + '/' + s + '_aerofoil.vtp'))
        node_markers = np.load(os.path.join(self.data_dir,s+'/'+'markers.npy')).reshape(-1,1)
        E = np.load(os.path.join(self.data_dir,s+'/'+'edges.npy'))
        E_copy = np.vstack([E[:,1],E[:,0]]).transpose()
        edge_indices = np.concatenate((E,E_copy),axis = 0)
        geometric_features = torch.load(os.path.join(self.data_dir,s+'/'+'geom.ft')) # A Data() object

        
        fields = [internal.point_data['U'][:,0],internal.point_data['U'][:,1]]
        
        fields = self.preprocess(fields); # normalized target
        # print('fields.shape: ',fields.shape)
        aoa, mach = self.get_params_from_name(self.file_list[idx]) # TO DO
        aoa,mach = np.ones_like(node_markers)*aoa, np.ones_like(node_markers)*mach
        aoa = torch.tensor(aoa,dtype = float)
        mach_or_reynolds = torch.tensor(mach,dtype = float)
        # print(self.mean_in)
        aoa_mean, mach_mean= self.mean_in[1],self.mean_in[2]
        aoa_std, mach_std = self.std_in[1],self.std_in[2]
        norm_aoa = (aoa - aoa_mean) / aoa_std # norm to [-1,1] range
        norm_mach_or_reynolds = (mach_or_reynolds - mach_mean)/mach_std
        # print('pos.shape: ',pos.shape)
        # print( norm_aoa.unsqueeze(0).repeat(pos.shape[0], 1).shape)
        # print( norm_mach_or_reynolds.unsqueeze(0).repeat(pos.shape[0], 1).shape)
        # print(type(norm_aoa),' ',type(node_markers),' ',type(norm_mach_or_reynolds))
        nodes = torch.hstack([
            torch.from_numpy(node_markers),
            norm_aoa,
            norm_mach_or_reynolds
        ])
        if prec==16:
            dtype = torch.float16
        elif prec==32:
            dtype = torch.float32
        elif prec==64:
            dtype = torch.float64
        data = Data(x=nodes.to(torch.float32), y=fields.to(torch.float32), edge_index=torch.tensor(edge_indices).t())
        # data.aoa = aoa.to(torch.float32)
        # data.norm_aoa = norm_aoa.to(torch.float32)
        # data.mach_or_reynolds = mach_or_reynolds.to(torch.float32)
        # data.norm_mach_or_reynolds = norm_mach_or_reynolds.to(torch.float32)
        data.coarse_path = os.path.join('Coarse_vtk/',s,'data.pt')
        data.sdf = torch.from_numpy(geometric_features.sdf).to(dtype)
        data.saf = geometric_features.saf.to(dtype)
        data.dsdf = geometric_features.dsdf.to(dtype)
        data.pos = torch.tensor(internal.points[:, :2]).to(dtype)

        # if prec == 64:
        #     data = Data(x=nodes, y=fields.double(), edge_index=torch.tensor(edge_indices).t())
        #     # data.aoa = aoa
        #     # data.norm_aoa = norm_aoa
        #     # data.mach_or_reynolds = mach_or_reynolds
        #     # data.norm_mach_or_reynolds = norm_mach_or_reynolds
        #     data.coarse_path = os.path.join('Coarse_vtk/',s,'data.pt')
        #     data.sdf = torch.from_numpy(geometric_features.sdf)
        #     data.saf = geometric_features.saf
        #     data.dsdf = geometric_features.dsdf
        #     data.pos = torch.tensor(internal.points[:, :2])

        data = transformer(data)

        # # Computing edge_indexA2
        # A1 = torch.sparse_coo_tensor(data.edge_index,torch.ones(data.edge_index.size(1)),size=(data.x.size(0),data.x.size(0)))
        # A2 = torch.sparse.mm(A1,A1)
        # data.edge_indexA2 = A2.coalesce().indices()

        # if self.fvdata: # If fvdata (sdf,dsdf,A^2) are required load the precomputed things
        #     data.saf = self.saf.clone().detach()
        #     data.dsdf = self.dsdf.clone().detach()
        #     data.edge_indexA2 = self.edge_indexA2.clone().detach()
        #     data.edge_attr = self.edge_attr.clone().detach()
        #     data.edge_attrA2 = self.edge_attrA2.clone().detach()
        
        # data.y = data.y[:,:3] #remove nut field
        return data

    def preprocess(self, tensor_list, stack_output=True):
        """ Normalize [U,V,p,nut] to [-1,1] range """
        # data_max, data_min = self.normalization_factors
        # normalized_tensors = []
        # for i in range(len(tensor_list)):
        #     # tensor_list[i] = (tensor_list[i] - data_means[i]) / data_stds[i] / 10
        #     normalized = (torch.tensor(tensor_list[i]) - data_min[i]) / (data_max[i] - data_min[i]) * 2 - 1 # norm to [-1,1] range
        #     if type(normalized) is np.ndarray:
        #         normalized = torch.from_numpy(normalized)
        #     normalized_tensors.append(normalized)
        # if stack_output:
        #     normalized_tensors = torch.stack(normalized_tensors, dim=1)
        _, _, mean_out, std_out = self.normalization_factors
        normalized_tensors = []
        for i in range(len(tensor_list)):
            normalized = (np.array(tensor_list[i]) - mean_out[i]) / std_out[i]
            if type(normalized) is np.ndarray:
                normalized = torch.from_numpy(normalized)
            normalized_tensors.append(normalized)
        if stack_output:
            normalized_tensors = torch.stack(normalized_tensors, dim=1)
        return normalized_tensors
    
    def coarse_preprocess(self, tensor):
        """ Normalize [U,V,p,nut] to [-1,1] range """
        _, _, mean_out, std_out = self.normalization_factors
        mean_out, std_out = torch.from_numpy(mean_out), torch.from_numpy(std_out)
        normalized = (tensor - mean_out)/std_out
        return normalized

    def _download(self):
        pass

    def _process(self):
        pass
    #def index_select(self,idx):
    #    for d in self.get(idx):
    #        yield d
    @staticmethod
    def get_params_from_name(filename):
        # s = filename.rsplit('.', 1)[0].split('_')
        # aoa = np.array(s[s.index('aoa') + 1])[np.newaxis].astype(np.float32)
        # reynolds = s[s.index('re') + 1]
        # reynolds = np.array(reynolds)[np.newaxis].astype(np.float32) if reynolds != 'None' else None
        # mach = np.array(s[s.index('mach') + 1])[np.newaxis].astype(np.float32)
        sound_speed = math.sqrt(1.4*287*300)
        Uinf, alpha = float(filename.split('_')[2]), float(filename.split('_')[3])
        mach = Uinf/sound_speed
        return alpha, mach # alpha is in degrees

class ccMeshAirfoilNSDataset(Dataset):
    def __init__(self, root='/ntuzfs/data/cfdgcnRR/ccfine_ns/', data_type='scarce', mode='train', ns = False):
        # self.fvdata = fvdata
        #print('NSdataset')
        with open('manifest.json', 'r') as f:
            manifest = json.load(f)
        self.ns = ns
        if ns: #NOT USING
            if mode == 'train' or mode == 'val':
                full_without_scarce = list(set(manifest['full_train']).difference(set(manifest['scarce_train'])))
                scarce = manifest['scarce_train']
                n = int(.1*len(scarce))
                train_dataset = full_without_scarce+ scarce[:-n]
                val_dataset = scarce[-n:]
                norm_set = scarce[:-n] # normalisation is only respect to known gt (scarce[:-n])
            if mode == 'test':
                test_dataset = manifest['full_test']
        else:

            if data_type=='scarce':
                tr = manifest['scarce_train']
                te = manifest['full_test']
            else:
                tr = manifest[data_type+'_train']
                te = manifest[data_type+'_test']

            final_tr, final_te = tr,te

            if mode == 'train' or mode == 'val':
                manifest_train = final_tr #manifest['scarce_train']
                n = int(.1*len(manifest_train))
                train_dataset = manifest_train[:-n] #manifest['scarce_train'][:-n]#[0:10]
                val_dataset = manifest_train[-n:]#[0:10]
                print('train length: ',len(train_dataset))
                print('val length: ',len(val_dataset))
            if mode == 'test':
                test_dataset = final_te #manifest['full_test']#[0:10]

        self.full_without_scarce = set(manifest['full_train']).difference(set(manifest['scarce_train']))

        self.mode = mode
        self.data_dir = root
        if mode=='train' or mode=='val':
            if mode=='train':
                self.file_list = train_dataset #norm_set
            else:
                self.file_list = val_dataset
        if os.path.isfile('ccns_normfactor_train.pkl'):
            (self.mean_in, self.std_in, self.mean_out, self.std_out)  = pickle.load(open('ccns_normfactor_train.pkl','rb'))
            self.normalization_factors = (self.mean_in, self.std_in, self.mean_out, self.std_out)
        else:
            for k,s in tqdm(enumerate(self.file_list),desc='Computing mean coeff: '):
                intern = torch.load(os.path.join(self.data_dir, s +'.pkl'))
                node_markers = intern.markers.reshape(-1,1)
                fields = intern.y[:,:2].numpy()
                aoa,mach = self.get_params_from_name(s)
                aoa,mach = np.ones_like(node_markers)*aoa, np.ones_like(node_markers)*mach
                init = np.hstack((node_markers,aoa,mach))
                # print('init shape: ',init.shape)
                if k == 0:
                    old_length = init.shape[0]
                    mean_in = init.mean(axis = 0, dtype = np.double)
                    mean_out = fields.mean(axis = 0, dtype = np.double)
                else:
                    new_length = old_length + init.shape[0]
                    mean_in += (init.sum(axis = 0, dtype = np.double) - init.shape[0]*mean_in)/new_length
                    mean_out += (fields.sum(axis = 0, dtype = np.double) - init.shape[0]*mean_out)/new_length
                    old_length = new_length 

            mean_in = mean_in.astype(np.single)
            mean_out = mean_out.astype(np.single)
            # print('mean_in = ',mean_in.shape)
            for k,s in tqdm(enumerate(self.file_list),desc = 'Computing std.dev coeff: '):
                intern = torch.load(os.path.join(self.data_dir, s +'.pkl'))
                node_markers = intern.markers.reshape(-1,1)
                fields = intern.y[:,:2].numpy()
                aoa,mach = self.get_params_from_name(s)
                aoa,mach = np.ones_like(node_markers)*aoa, np.ones_like(node_markers)*mach
                init = np.hstack((node_markers,aoa,mach))
                if k == 0:
                    old_length = init.shape[0]
                    std_in = ((init - mean_in)**2).sum(axis = 0, dtype = np.double)/old_length
                    std_out = ((fields - mean_out)**2).sum(axis = 0, dtype = np.double)/old_length
                else:
                    new_length = old_length + init.shape[0]
                    std_in += (((init - mean_in)**2).sum(axis = 0, dtype = np.double) -init.shape[0]*std_in)/new_length
                    std_out += (((fields - mean_out)**2).sum(axis = 0, dtype = np.double) - init.shape[0]*std_out)/new_length
                    old_length = new_length
            std_in = np.sqrt(std_in).astype(np.single)
            std_out = np.sqrt(std_out).astype(np.single)

            self.normalization_factors = (mean_in, std_in, mean_out, std_out) 
            self.mean_in, self.std_in, self.mean_out, self.std_out = self.normalization_factors
            with open('ccns_normfactor_train.pkl','wb') as f:
                pickle.dump(self.normalization_factors,f)
        if mode=='test':
            self.file_list = test_dataset
            with open('ccns_normfactor_train.pkl','rb') as f:
                self.mean_in, self.std_in, self.mean_out, self.std_out  = pickle.load(f)
                self.normalization_factors = self.mean_in, self.std_in, self.mean_out, self.std_out
        self.len = len(self.file_list)
        # print('norm coeff: ',self.normalization_factors)
        super().__init__(root)

    def indices(self):
        return range(self.len)
        
    def len(self):
        return self.len

    def get(self, idx):
        transformer = T.Compose([T.Cartesian(norm=False), T.Distance(norm=False)])
        prec = 16
        s = self.file_list[idx]
        intern = torch.load(os.path.join(self.data_dir, s +'.pkl'))
        node_markers = intern.markers.reshape(-1,1)
        fields = [intern.y[:,0],intern.y[:,1]] # U,V only
        
        fields = self.preprocess(fields)
        # print('fields.shape: ',fields.shape)
        aoa, mach = self.get_params_from_name(self.file_list[idx]) # TO DO
        aoa,mach = np.ones_like(node_markers)*aoa, np.ones_like(node_markers)*mach
        aoa = torch.tensor(aoa,dtype = float)
        mach_or_reynolds = torch.tensor(mach,dtype = float)
        # print(self.mean_in)
        aoa_mean, mach_mean= self.mean_in[1],self.mean_in[2]
        aoa_std, mach_std = self.std_in[1],self.std_in[2]
        norm_aoa = (aoa - aoa_mean) / aoa_std # norm to [-1,1] range
        norm_mach_or_reynolds = (mach_or_reynolds - mach_mean)/mach_std
        nodes = torch.hstack([
            node_markers,
            norm_aoa,
            norm_mach_or_reynolds
        ])
        if prec==16:
            dtype = torch.float16
        elif prec==32:
            dtype = torch.float32
        elif prec==64:
            dtype = torch.float64
        data = Data(x=nodes.to(dtype), y=fields.to(dtype), edge_index=intern.edge_index)
        data.aoa = aoa.to(dtype)
        data.norm_aoa = norm_aoa.to(dtype)
        data.mach_or_reynolds = mach_or_reynolds.to(dtype)
        data.norm_mach_or_reynolds = norm_mach_or_reynolds.to(dtype)
        data.coarse_path = intern.coarse_path
        # data.sdf = intern.sdf.to(torch.float32)
        data.saf = intern.saf.to(dtype)
        data.dsdf = intern.dsdf.to(dtype)
        data.node_FVattr = intern.node_attr.unsqueeze(1).to(dtype=dtype)
        data.edge_FVattr = intern.edge_attr.to(dtype=dtype)
        data.sf = intern.sf.to(dtype=dtype)
        data.cf = intern.cf.to(dtype=dtype)
        data.cell_next = intern.cell_next 
        data.owner_to_neighbor = intern.owner_to_neighbor
        data.pos = intern.x[:,:2].to(dtype=dtype)

        # if prec == 64:
        #     data = Data(x=nodes, y=fields.double(), edge_index=intern.edge_index)
        #     data.aoa = aoa
        #     data.norm_aoa = norm_aoa
        #     data.mach_or_reynolds = mach_or_reynolds
        #     data.norm_mach_or_reynolds = norm_mach_or_reynolds
        #     data.coarse_path = intern.coarse_path
        #     # data.sdf = intern.sdf
        #     data.saf = intern.saf
        #     data.dsdf = intern.dsdf
        #     data.node_attr = intern.node_attr.unsqueeze(1).double()
        #     data.edge_attr_ours = intern.edge_attr.double()
        #     data.sf = intern.sf.double()
        #     data.cf = intern.cf.double()
        #     data.cell_next = intern.cell_next 
        #     data.owner_to_neighbor = intern.owner_to_neighbor
        #     data.pos = intern.x[:,:2]         
        data = transformer(data)

        del data.edge_attr_ours
        if self.ns:
            data.nsres = [False, True][s in self.full_without_scarce]
        if self.ns is False:
            del data.owner_to_neighbor
            del data.cf
            del data.sf
            del data.cell_next 
        del data.edge_indexA2
        # if self.fvdata:
        #     data.node_attr = intern.node_attr  #.clone().detach()
        #     data.edge_attr = intern.edge_attr  #.clone().detach()

        #     edge_attr = torch.cat((data.edge_attr[:,0:1],(data.edge_attr[:,1:]-pos[data.edge_index[0,:],:]),(data.edge_attr[:,1:]-pos[data.edge_index[1,:],:])), dim=1)
        #     node_attr = data.node_attr.unsqueeze(1) #torch.cat((pos,data.node_attr.unsqueeze(1)),dim=1)
            
        #     A1 = [torch.sparse_coo_tensor(data.edge_index,edge_attr[:,d],\
        #         size=(data.x.size(0),data.x.size(0))) for d in range(edge_attr.size(1))]
        #     A2 = [torch.sparse.mm(A1[d],A1[d]) for d in range(len(A1))]
        #     data.edge_indexA2 = A2[0].coalesce().indices() #
        #     edge_attrA2 = torch.stack([A2[d].coalesce().values() for d in range(len(A2))],dim=1)
            
        #     rel_Nattr = node_attr[data.edge_index[0,:],:] - node_attr[data.edge_index[1,:],:]
        #     data.edge_attr = torch.cat((rel_Nattr,edge_attr), dim=1).type(torch.float32) #
        #     rel_NattrA2 = node_attr[data.edge_indexA2[0,:],:] - node_attr[data.edge_indexA2[1,:],:]
        #     data.edge_attrA2 = torch.cat((rel_NattrA2,edge_attrA2), dim=1).type(torch.float32) #
        #     data.node_attr = None
        if idx==0:
            print('data from ccMeshAirfoilNS get(): ', data)
        return data

    def preprocess(self, tensor_list, stack_output=True):
        """ Normalize [U,V,p,nut] to [-1,1] range """
        _, _, mean_out, std_out = self.normalization_factors
        normalized_tensors = []
        for i in range(len(tensor_list)):
            normalized = (np.array(tensor_list[i]) - mean_out[i]) / std_out[i]
            if type(normalized) is np.ndarray:
                normalized = torch.from_numpy(normalized)
            normalized_tensors.append(normalized)
        if stack_output:
            normalized_tensors = torch.stack(normalized_tensors, dim=1)
        return normalized_tensors
    
    def coarse_preprocess(self, tensor):
        """ Normalize [U,V,p,nut] to [-1,1] range """
        _, _, mean_out, std_out = self.normalization_factors
        mean_out, std_out = torch.from_numpy(mean_out), torch.from_numpy(std_out)
        normalized = (tensor - mean_out)/std_out
        return normalized

    def _download(self):
        pass

    def _process(self):
        pass
    #def index_select(self,idx):
    #    for d in self.get(idx):
    #        yield d
    @staticmethod
    def get_params_from_name(filename):
        # s = filename.rsplit('.', 1)[0].split('_')
        # aoa = np.array(s[s.index('aoa') + 1])[np.newaxis].astype(np.float32)
        # reynolds = s[s.index('re') + 1]
        # reynolds = np.array(reynolds)[np.newaxis].astype(np.float32) if reynolds != 'None' else None
        # mach = np.array(s[s.index('mach') + 1])[np.newaxis].astype(np.float32)
        sound_speed = math.sqrt(1.4*287*300)
        Uinf, alpha = float(filename.split('_')[2]), float(filename.split('_')[3])
        mach = Uinf/sound_speed
        return alpha, mach # alpha is in degrees