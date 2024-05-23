#@title CFDGCN and CFDFVGCN to be geometry-general and not train SU2 coarse mesh, with bs>1
import os
import logging
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.unpool import knn_interpolate

import torch_geometric.nn as nng #ADDED
from torch_geometric.utils import add_self_loops #ADDED!

##ADDED for SAGE###
# import random
from typing import List, Optional, Tuple, Union
from torch import Tensor
from torch.nn import LSTM
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size ##ADDED for SAGE###

#from su2torch import SU2Module
from mesh_utils import write_graph_mesh, quad2tri, get_mesh_graph, signed_dist_graph, is_cw


class MeshGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=6, improved=False,
                 cached=False, bias=True, fine_marker_dict=None):
        super().__init__()
        #self.fine_marker_dict = torch.tensor(fine_marker_dict['airfoil']).unique()
        self.sdf = None
        in_channels += 1  # account for sdf

        channels = [in_channels]
        channels += [hidden_channels] * (num_layers - 1)
        channels.append(out_channels)

        convs = []
        for i in range(num_layers):
            convs.append(GCNConv(channels[i], channels[i+1], improved=improved,
                                 cached=cached, bias=bias))
        self.convs = nn.ModuleList(convs)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch_size = data.aoa.shape[0]

        # if self.sdf is None:
        #     with torch.no_grad():
        #         self.sdf = signed_dist_graph(x[data.batch == 0, :2],
        #                                      self.fine_marker_dict).unsqueeze(1)
        # x = torch.cat([data.x, self.sdf.repeat(batch_size, 1)], dim=1)
        x = torch.cat([data.x, data.saf, data.dsdf], dim=1)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)

        x = self.convs[-1](x, edge_index)
        return x


class CFDGCN(nn.Module):
    #def __init__(self, config_file, coarse_mesh, fine_marker_dict, process_sim=lambda x, y: x,
    def __init__(self, process_sim=lambda x, y: x, residual=False,
                 freeze_mesh=False, num_convs=6, num_end_convs=3, hidden_channels=512,
                 out_channels=3, saf = False, dsdf = False, device='cuda'):
        super().__init__()
        self.saf, self.dsdf = saf, dsdf
        self.residual = residual
        # meshes_temp_dir = 'temp_meshes'
        # os.makedirs(meshes_temp_dir, exist_ok=True)
        # self.mesh_file = meshes_temp_dir + '/' + str(os.getpid()) + '_mesh.su2'

        # if not coarse_mesh:
        #     raise ValueError('Need to provide a coarse mesh for CFD-GCN.')
        # nodes, edges, self.elems, self.marker_dict = get_mesh_graph(coarse_mesh)
        # self.nodes = torch.from_numpy(nodes).to(device)
        # if not freeze_mesh:
        #     self.nodes = nn.Parameter(self.nodes) # Treat x,y- coordinates as parameters of the NN.
        # self.elems, new_edges = quad2tri(sum(self.elems, [])) # sum() is combining triads and quads into a single list. Inside quad2tri(), Quads are broken down to triads whereas triads are kept as they are.
        # self.elems = [self.elems]
        # self.edges = torch.from_numpy(edges).to(device)
        # # print(self.edges.dtype, new_edges.dtype)
        # self.edges = torch.cat([self.edges, new_edges.to(self.edges.device)], dim=1)
        # self.marker_inds = torch.tensor(sum(self.marker_dict.values(), [])).unique()
        # assert is_cw(self.nodes, self.elems[0]).nonzero().shape[0] == 0, 'Mesh has flipped elems'

        self.process_sim = process_sim
        # self.su2 = SU2Module(config_file, mesh_file=self.mesh_file)
        # logging.info(f'Mesh filename: {self.mesh_file.format(batch_index="*")}')

        # self.fine_marker_dict = torch.tensor(fine_marker_dict['airfoil']).unique()
        self.sdf = None

        improved = False
        self.num_convs = num_end_convs
        self.convs = []
        if self.num_convs > 0:
            self.convs = nn.ModuleList()
            in_channels = out_channels + hidden_channels
            for i in range(self.num_convs - 1):
                self.convs.append(GCNConv(in_channels, hidden_channels, improved=improved))
                in_channels = hidden_channels
            self.convs.append(GCNConv(in_channels, out_channels, improved=improved))

        self.num_pre_convs = num_convs - num_end_convs
        self.pre_convs = []
        if self.num_pre_convs > 0:
            in_channels = 5  # one extra channel for sdf
            in_channels += 2 if (self.saf==1) else 1 #ADDED
            in_channels += 8 if (self.dsdf==1) else 0 #ADDED
            self.pre_convs = nn.ModuleList()
            for i in range(self.num_pre_convs - 1):
                self.pre_convs.append(GCNConv(in_channels, hidden_channels, improved=improved))
                in_channels = hidden_channels
            self.pre_convs.append(GCNConv(in_channels, hidden_channels, improved=improved))

        self.sim_info = {}  # store output of coarse simulation for logging / debugging

    def forward(self, batch):
        start = time.time()
        batch_size = 1 #batch.aoa.shape[0]
        
        fine_x = batch.x
        if self.saf==1:
            fine_x = torch.cat([fine_x, batch.saf], dim=1)
        else:
            # if self.sdf is None:
                # with torch.no_grad():
                    # self.sdf = signed_dist_graph(batch.x[batch.batch == 0, :2],
                    #                              self.fine_marker_dict).unsqueeze(1)
            # fine_x = torch.cat([fine_x, self.sdf.repeat(batch_size, 1)], dim=1)
            fine_x = torch.cat([fine_x, batch.sdf], dim=1)
        if self.dsdf==1:
            fine_x = torch.cat([fine_x, batch.dsdf], dim=1)

        for i, conv in enumerate(self.pre_convs):
            fine_x = F.relu(conv(fine_x, batch.edge_index))

        nodes = fine_x[:,:2] #self.get_nodes()
        num_nodes = nodes.shape[0]
        #self.write_mesh_file(nodes, self.elems, self.marker_dict, filename=self.mesh_file)

        params = torch.stack([batch.aoa, batch.mach_or_reynolds], dim=1)
        batch_aoa = params[:, 0].to('cpu', non_blocking=True)
        batch_mach_or_reynolds = params[:, 1].to('cpu', non_blocking=True)

        # batch_x = nodes.unsqueeze(0).expand(batch_size, -1, -1)
        # batch_x = batch_x.to('cpu', non_blocking=True)
        # batch_y = self.su2(batch_x[..., 0], batch_x[..., 1],
        #                    batch_aoa[..., None], batch_mach_or_reynolds[..., None])
        # batch_y = [y.to(batch.x.device) for y in batch_y]
        # batch_y = self.process_sim(batch_y, False)

        # coarse_y = torch.stack([y.flatten() for y in batch_y], dim=1)
        # coarse_x = nodes.repeat(batch_size, 1)[:, :2]
        # zeros = batch.batch.new_zeros(num_nodes)
        # coarse_batch = torch.cat([zeros + i for i in range(batch_size)])
        
        #coarse_x, coarse_y = batch.coarse_x, batch.coarse_y #ADDED
        #coarse_data = torch.load(batch.coarse_path[0]) #ONLY LOADING FIRST OF BATCH
        coarse_data = [torch.load(batch.coarse_path[b]) for b in range(batch_size)]
        coarse_data = torch_geometric.data.Batch.from_data_list(coarse_data)
        
        coarse_x, coarse_y = coarse_data.x, self.process_sim(coarse_data.y)
        coarse_x, coarse_y = coarse_x.to(dtype=torch.float32), coarse_y.to(dtype=torch.float32)[:,:3] #no nut
        #coarse_batch = torch.zeros(coarse_x.size(0)) #ASSUMING BS=1
        coarse_batch = coarse_data.batch

        fine_y_orig = self.upsample(coarse_y, coarse_x, coarse_batch, batch).to(fine_x.device)
        fine_y = torch.cat([fine_y_orig, fine_x], dim=1)

        for i, conv in enumerate(self.convs[:-1]):
            fine_y = F.relu(conv(fine_y, batch.edge_index))
        fine_y = self.convs[-1](fine_y, batch.edge_index)

        self.sim_info['nodes'] = coarse_x[:, :2]
        # self.sim_info['elems'] = [self.elems] * batch_size
        self.sim_info['batch'] = coarse_batch
        self.sim_info['output'] = coarse_y

        # return fine_y
        return (fine_y + fine_y_orig) if self.residual else fine_y

    def upsample(self, y, coarse_nodes, coarse_batch, fine):
        fine_nodes = fine.x[:, :2]
        y = knn_interpolate(y.cpu(), coarse_nodes[:, :2].cpu(), fine_nodes.cpu(),
                            coarse_batch.cpu(), fine.batch.cpu(), k=3).to(y.device)
        return y

    def get_nodes(self):
        # return torch.cat([self.marker_nodes, self.not_marker_nodes])
        return self.nodes

    @staticmethod
    def write_mesh_file(x, elems, marker_dict, filename='mesh.su2'):
        write_graph_mesh(filename, x[:, :2], elems, marker_dict)

    @staticmethod
    def contiguous_elems_list(elems, inds):
        # Hack to easily have compatibility with MeshEdgePool
        return elems


class UCM(CFDGCN):
    """Simply upsamples the coarse simulation without using any GCNs."""
    #def __init__(self, config_file, coarse_mesh, fine_marker_dict, process_sim=lambda x, y: x,
    def __init__(self, process_sim=lambda x, y: x,
                 freeze_mesh=False, device='cuda'):
        super().__init__(config_file, coarse_mesh, fine_marker_dict, process_sim=process_sim,
                         freeze_mesh=freeze_mesh, num_convs=0, num_end_convs=0, device=device)

    def forward(self, batch):
        batch_size = batch.aoa.shape[0]

        nodes = batch.x[:,:2] #self.get_nodes()
        num_nodes = nodes.shape[0]
        self.write_mesh_file(nodes, self.elems, self.marker_dict, filename=self.mesh_file)

        params = torch.stack([batch.aoa, batch.mach_or_reynolds], dim=1)
        batch_aoa = params[:, 0].to('cpu', non_blocking=True)
        batch_mach_or_reynolds = params[:, 1].to('cpu', non_blocking=True)

        # batch_x = nodes.unsqueeze(0).expand(batch_size, -1, -1)
        # batch_x = batch_x.to('cpu', non_blocking=True)
        # batch_y = self.su2(batch_x[..., 0], batch_x[..., 1],
        #                    batch_aoa[..., None], batch_mach_or_reynolds[..., None])
        # batch_y = [y.to(batch.x.device) for y in batch_y]
        # batch_y = self.process_sim(batch_y, False)

        # coarse_y = torch.stack([y.flatten() for y in batch_y], dim=1)
        # coarse_x = nodes.repeat(batch_size, 1)[:, :2]
        # zeros = batch.batch.new_zeros(num_nodes)
        # coarse_batch = torch.cat([zeros + i for i in range(batch_size)])
        
        #coarse_x, coarse_y = batch.coarse_x, batch.coarse_y #ADDED
        #coarse_data = torch.load(batch.coarse_path[0]) #ONLY LOADING FIRST OF BATCH
        coarse_data = [torch.load(batch.coarse_path[b]) for b in range(batch_size)]
        coarse_data = torch_geometric.data.Batch.from_data_list(coarse_data)
        
        coarse_x, coarse_y = coarse_data.x, self.process_sim(coarse_data.y)
        coarse_x, coarse_y = coarse_x.to(dtype=torch.float32), coarse_y.to(dtype=torch.float32)[:,:3] #no nut
        #coarse_batch = torch.zeros(coarse_x.size(0)) #ASSUMING BS=1
        coarse_batch = coarse_data.batch
        
        fine_y = self.upsample(coarse_y, coarse_x, coarse_batch, batch)

        self.sim_info['nodes'] = coarse_x[:, :2]
        # self.sim_info['elems'] = [self.elems] * batch_size
        self.sim_info['batch'] = coarse_batch
        self.sim_info['output'] = coarse_y

        return fine_y


class CFD(CFDGCN):
    """Simply outputs the results of the (fine) CFD simulation."""
    #def __init__(self, config_file, mesh, fine_marker_dict, process_sim=lambda x, y: x,
    def __init__(self, process_sim=lambda x, y: x,
                 freeze_mesh=False, device='cuda'):
        super().__init__(config_file, mesh, fine_marker_dict, process_sim=process_sim,
                         freeze_mesh=freeze_mesh, num_convs=0, num_end_convs=0, device=device)

    def forward(self, batch):
        batch_size = batch.aoa.shape[0]

        nodes = batch.x[:,:2] #self.get_nodes()
        num_nodes = nodes.shape[0]
        self.write_mesh_file(nodes, self.elems, self.marker_dict, filename=self.mesh_file)

        params = torch.stack([batch.aoa, batch.mach_or_reynolds], dim=1)
        batch_aoa = params[:, 0].to('cpu', non_blocking=True)
        batch_mach_or_reynolds = params[:, 1].to('cpu', non_blocking=True)

        # batch_x = nodes.unsqueeze(0).expand(batch_size, -1, -1)
        # batch_x = batch_x.to('cpu', non_blocking=True)
        # batch_y = self.su2(batch_x[..., 0], batch_x[..., 1],
        #                    batch_aoa[..., None], batch_mach_or_reynolds[..., None])
        # batch_y = [y.to(batch.x.device) for y in batch_y]
        # batch_y = self.process_sim(batch_y, False)

        # coarse_y = torch.stack([y.flatten() for y in batch_y], dim=1)
        # coarse_x = nodes.repeat(batch_size, 1)[:, :2]
        # zeros = batch.batch.new_zeros(num_nodes)
        # coarse_batch = torch.cat([zeros + i for i in range(batch_size)])

        #coarse_x, coarse_y = batch.coarse_x, batch.coarse_y #ADDED
        #coarse_data = torch.load(batch.coarse_path[0]) #ONLY LOADING FIRST OF BATCH
        coarse_data = [torch.load(batch.coarse_path[b]) for b in range(batch_size)]
        coarse_data = torch_geometric.data.Batch.from_data_list(coarse_data)
        
        coarse_x, coarse_y = coarse_data.x, self.process_sim(coarse_data.y)
        coarse_x, coarse_y = coarse_x.to(dtype=torch.float32), coarse_y.to(dtype=torch.float32)[:,:3] #no nut
        #coarse_batch = torch.zeros(coarse_x.size(0)) #ASSUMING BS=1
        coarse_batch = coarse_data.batch
        fine_y = coarse_y

        self.sim_info['nodes'] = coarse_x[:, :2]
        # self.sim_info['elems'] = [self.elems] * batch_size
        self.sim_info['batch'] = coarse_batch
        self.sim_info['output'] = coarse_y

        return fine_y

# ================================= Our code ========================================
class SpatialGraphConv(nng.MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_size, attr_num=0, dropout=0):
        """
        attrN - number of edge attributes (0 or other integer #attr) ##ADDED
        in_channels - number of the input channels (node features)
        out_channels - number of the output channels (node features)
        hidden_size - number of the inner convolutions
        dropout - dropout rate after the layer
        """
        super(SpatialGraphConv, self).__init__(aggr='add')
        self.dropout = dropout; self.attr_num = attr_num
        #self.lin_in = torch.nn.Linear(coors, hidden_size * in_channels)
        ##EDITED
        self.lin_in = torch.nn.Linear(attr_num, hidden_size * in_channels)
        self.lin_out = torch.nn.Linear(hidden_size * in_channels, out_channels)
        self.in_channels = in_channels

    def forward(self, x, edge_index, edge_attr=None):
        """
        x - feature matrix of the whole graph [num_nodes, label_dim]
        edge_index - graph connectivity [2, num_edges]
        edge_attr - edge attributes [num_edges, #attr] or none ##ADDED
        """
        #edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))  # num_edges = num_edges + num_nodes
        ##EDITED
        #edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=0, num_nodes=x.size(0))
        return self.propagate(edge_index=edge_index, edge_attr=edge_attr, x=x, aggr='add')  # [N, out_channels, label_dim]

    def message(self, x_j, edge_attr=None):
        """
        x_j [num_edges, label_dim]
        edge_attr [num_edges, #attr] or none ##ADDED
        """
        if edge_attr is not None:
          scaling = F.relu(self.lin_in(edge_attr))  # [n_edges, hidden_size * in_channels]
        else:
          scaling = torch.ones(x_j.size()).unsqueeze(-1).to(x_j.device)
        n_edges = x_j.size(0)
        # [n_edges, in_channels, ...] * [n_edges, in_channels, 1]
        result = scaling.reshape(n_edges, self.in_channels, -1) * x_j.unsqueeze(-1)
        return result.view(n_edges, -1)

    def update(self, aggr_out):
        """
        aggr_out [num_nodes, label_dim, out_channels]
        """
        aggr_out = self.lin_out(aggr_out)  # [num_nodes, label_dim, out_features]
        aggr_out = torch.tanh(aggr_out); #aggr_out = F.relu(aggr_out)
        aggr_out = F.dropout(aggr_out, p=self.dropout, training=self.training)
        return aggr_out

##############################
#### aGCN_conv CLASS DEF #####
##############################
#aGCN_conv CLASS codes, with use_A<0 if shared, adapted for dataparallel
class aGCN_conv(torch.nn.Module):
    '''#to use (e.g.):
    layer = aGCN_conv(use_A, conv_layer, **kwargs)'''
    def __init__(self, conv_layer, use_A=-2, **kwargs):
        super(aGCN_conv, self).__init__()
        self.layer_list = torch.nn.ModuleList(); self.use_A=use_A
        if use_A<=0: #shared parameters
          layer = conv_layer(**kwargs)
          self.layer_list.append(layer)
        else: #non-shared parameters
          for a in range(use_A):
            layer = conv_layer(**kwargs)
            self.layer_list.append(layer)

    def forward(self, x, edge_index_list, edge_attr_list=[None]): #data input for DataParallel listloader
      edge_index, edge_attr = edge_index_list[0], edge_attr_list[0]
    
      if edge_attr==None:
        x0 = self.layer_list[0](x, edge_index); n=x.size(0)
        for a in range(abs(self.use_A)-1):
          edge_index, = edge_index_list[a+1], 
          if self.use_A>0: #non-shared parameters
            xa = self.layer_list[a+1](x, edge_index)
          else: #shared parameters
            xa = self.layer_list[0](x, edge_index)
          x0 = torch.cat((x0,xa),dim=1)
        
      else: #if using edge_attr
        x0 = self.layer_list[0](x, edge_index, edge_attr=edge_attr); n=x.size(0)
        for a in range(abs(self.use_A)-1):
          edge_index, edge_attr = edge_index_list[a+1], edge_attr_list[a+1]
          if self.use_A>0: #non-shared parameters
            xa = self.layer_list[a+1](x, edge_index, edge_attr=edge_attr)
          else: #shared parameters
            xa = self.layer_list[0](x, edge_index, edge_attr=edge_attr)
          x0 = torch.cat((x0,xa),dim=1)
      return x0

class CFDFVGCN(nn.Module):
    #def __init__(self, config_file, coarse_mesh, fine_marker_dict, process_sim=lambda x, y: x,
    def __init__(self, process_sim=lambda x, y: x,
                 freeze_mesh=False, num_convs=6, num_end_convs=3, hidden_channels=512, out_channels=3,
                 saf=False, dsdf=False, A_pow=2, A_shared='False', FV='False', device='cuda'):
        super().__init__()
        # meshes_temp_dir = 'temp_meshes'
        # os.makedirs(meshes_temp_dir, exist_ok=True)
        # self.mesh_file = meshes_temp_dir + '/' + str(os.getpid()) + '_mesh.su2'

        # if not coarse_mesh:
        #     raise ValueError('Need to provide a coarse mesh for CFD-GCN.')
        # nodes, edges, self.elems, self.marker_dict = get_mesh_graph(coarse_mesh)
        # self.nodes = torch.from_numpy(nodes).to(device)
        # if not freeze_mesh:
        #     self.nodes = nn.Parameter(self.nodes)
        # self.elems, new_edges = quad2tri(sum(self.elems, []))
        # self.elems = [self.elems]
        # self.edges = torch.from_numpy(edges).to(device)
        # print(self.edges.dtype, new_edges.dtype)
        # self.edges = torch.cat([self.edges, new_edges.to(self.edges.device)], dim=1)
        # self.marker_inds = torch.tensor(sum(self.marker_dict.values(), [])).unique()
        # assert is_cw(self.nodes, self.elems[0]).nonzero().shape[0] == 0, 'Mesh has flipped elems'

        self.process_sim = process_sim
        # self.su2 = SU2Module(config_file, mesh_file=self.mesh_file)
        # logging.info(f'Mesh filename: {self.mesh_file.format(batch_index="*")}')

        # self.fine_marker_dict = torch.tensor(fine_marker_dict['airfoil']).unique()
        self.sdf = None

        #ADDED:
        self.saf, self.dsdf = saf, dsdf
        attr_num = 6 if FV else 0
        A_pow = A_pow*-1 if (A_shared) else A_pow #ADDED####

        improved = False
        self.num_convs = num_end_convs
        self.convs = []
        if self.num_convs > 0:
            self.convs = nn.ModuleList()
            in_channels = out_channels + hidden_channels*abs(A_pow)
            for i in range(self.num_convs - 1):
                #EDITED:
                #self.convs.append(GCNConv(in_channels, hidden_channels, improved=improved))
                self.convs.append(aGCN_conv(SpatialGraphConv,
                                      in_channels = in_channels,
                                      out_channels = hidden_channels,
                                      attr_num = attr_num,
                                      hidden_size = 3, 
                                      dropout = 0,
                                      use_A = A_pow))
                in_channels = hidden_channels*abs(A_pow)
            #EDITED:
            #self.convs.append(GCNConv(in_channels, out_channels, improved=improved))
            self.convs.append(aGCN_conv(SpatialGraphConv,
                                      in_channels = in_channels,
                                      out_channels = out_channels,
                                      attr_num = attr_num,
                                      hidden_size = 3, 
                                      dropout = 0,
                                      use_A = 1)) #EDITED#### #last one just A1

        self.num_pre_convs = num_convs - num_end_convs
        self.pre_convs = []
        if self.num_pre_convs > 0:
            in_channels = 5 # one extra channel for sdf
            in_channels += 2 if (self.saf==1) else 1 #ADDED
            in_channels += 8 if (self.dsdf==1) else 0#ADDED
            self.pre_convs = nn.ModuleList()
            for i in range(self.num_pre_convs - 1):
                #EDITED:
                #self.pre_convs.append(GCNConv(in_channels, hidden_channels, improved=improved))
                self.pre_convs.append(aGCN_conv(SpatialGraphConv,
                                      in_channels = in_channels,
                                      out_channels = hidden_channels,
                                      attr_num = attr_num,
                                      hidden_size = 3, 
                                      dropout = 0,
                                      use_A = A_pow)) #EDITED####
                in_channels = hidden_channels*abs(A_pow)
            #EDITED:
            #self.pre_convs.append(GCNConv(in_channels, hidden_channels, improved=improved))
            self.pre_convs.append(aGCN_conv(SpatialGraphConv,
                                      in_channels = in_channels,
                                      out_channels = hidden_channels,
                                      attr_num = attr_num,
                                      hidden_size = 3, 
                                      dropout = 0,
                                      use_A = A_pow))

        self.sim_info = {}  # store output of coarse simulation for logging / debugging

    def forward(self, batch):
        start = time.time()
        batch_size = batch.aoa.shape[0]

        fine_x = batch.x
        if self.saf==1:
            fine_x = torch.cat([fine_x, batch.saf], dim=1)
        else:
            # if self.sdf is None:
            #     with torch.no_grad():
            #         self.sdf = signed_dist_graph(batch.x[batch.batch == 0, :2],
            #                                      self.fine_marker_dict).unsqueeze(1)
            # fine_x = torch.cat([fine_x, self.sdf.repeat(batch_size, 1)], dim=1)
            fine_x = torch.cat([fine_x, batch.sdf], dim=1)
        if self.dsdf==1:
            fine_x = torch.cat([fine_x, batch.dsdf], dim=1)
        
        edge_index_list = [batch.edge_index, batch.edge_indexA2]
        edge_attr_list = [batch.edge_attr, batch.edge_attrA2]

        for i, conv in enumerate(self.pre_convs):
            #fine_x = F.relu(conv(fine_x, batch.edge_index))
            fine_x = F.relu(conv(fine_x, edge_index_list, edge_attr_list=edge_attr_list)) #EDITED

        nodes = batch.x[:,:2] #self.get_nodes()
        num_nodes = nodes.shape[0]
        #self.write_mesh_file(nodes, self.elems, self.marker_dict, filename=self.mesh_file)

        params = torch.stack([batch.aoa, batch.mach_or_reynolds], dim=1)
        batch_aoa = params[:, 0].to('cpu', non_blocking=True)
        batch_mach_or_reynolds = params[:, 1].to('cpu', non_blocking=True)

        # batch_x = nodes.unsqueeze(0).expand(batch_size, -1, -1)
        # batch_x = batch_x.to('cpu', non_blocking=True)
        # batch_y = self.su2(batch_x[..., 0], batch_x[..., 1],
        #                    batch_aoa[..., None], batch_mach_or_reynolds[..., None])
        # batch_y = [y.to(batch.x.device) for y in batch_y]
        # batch_y = self.process_sim(batch_y, False)

        # coarse_y = torch.stack([y.flatten() for y in batch_y], dim=1)
        # coarse_x = nodes.repeat(batch_size, 1)[:, :2]
        # zeros = batch.batch.new_zeros(num_nodes)
        # coarse_batch = torch.cat([zeros + i for i in range(batch_size)])

        #coarse_x, coarse_y = batch.coarse_x, batch.coarse_y #ADDED
        #coarse_data = torch.load(batch.coarse_path[0]) #ONLY LOADING FIRST OF BATCH
        coarse_data = [torch.load(batch.coarse_path[b]) for b in range(batch_size)]
        coarse_data = torch_geometric.data.Batch.from_data_list(coarse_data)
        
        coarse_x, coarse_y = coarse_data.x, self.process_sim(coarse_data.y)
        coarse_x, coarse_y = coarse_x.to(dtype=torch.float32), coarse_y.to(dtype=torch.float32)[:,:3] #no nut
        #coarse_batch = torch.zeros(coarse_x.size(0)) #ASSUMING BS=1
        coarse_batch = coarse_data.batch
        
        fine_y = self.upsample(coarse_y, coarse_x, coarse_batch, batch).to(fine_x.device)
        fine_y = torch.cat([fine_y, fine_x], dim=1)

        for i, conv in enumerate(self.convs[:-1]):
            #fine_y = F.relu(conv(fine_y, batch.edge_index))
            fine_y = F.relu(conv(fine_y, edge_index_list, edge_attr_list=edge_attr_list)) #EDITED
        #fine_y = self.convs[-1](fine_y, batch.edge_index)
        fine_y = self.convs[-1](fine_y, edge_index_list, edge_attr_list=edge_attr_list) #EDITED

        self.sim_info['nodes'] = coarse_x[:, :2]
        # self.sim_info['elems'] = [self.elems] * batch_size
        self.sim_info['batch'] = coarse_batch
        self.sim_info['output'] = coarse_y

        return fine_y

    def upsample(self, y, coarse_nodes, coarse_batch, fine):
        fine_nodes = fine.x[:, :2]
        y = knn_interpolate(y.cpu(), coarse_nodes[:, :2].cpu(), fine_nodes.cpu(),
                            coarse_batch.cpu(), fine.batch.cpu(), k=3).to(y.device)
        return y

    def get_nodes(self):
        # return torch.cat([self.marker_nodes, self.not_marker_nodes])
        return self.nodes

    @staticmethod
    def write_mesh_file(x, elems, marker_dict, filename='mesh.su2'):
        write_graph_mesh(filename, x[:, :2], elems, marker_dict)

    @staticmethod
    def contiguous_elems_list(elems, inds):
        # Hack to easily have compatibility with MeshEdgePool
        return elems
    
# ================================= Our code ========================================

##### NEW CLASS ####

class CFDAGCN(nn.Module):
    #def __init__(self, config_file, coarse_mesh, fine_marker_dict, process_sim=lambda x, y: x,
    def __init__(self, process_sim=lambda x, y: x,
                 freeze_mesh=False, num_convs=6, num_end_convs=3, hidden_channels=512, out_channels=3,
                 saf=False, dsdf=False, A_pow=2, A_shared='False', FV='False', device='cuda'):
        super().__init__()
        
#         meshes_temp_dir = 'temp_meshes'
#         os.makedirs(meshes_temp_dir, exist_ok=True)
#         self.mesh_file = meshes_temp_dir + '/' + str(os.getpid()) + '_mesh.su2'

#         if not coarse_mesh:
#             raise ValueError('Need to provide a coarse mesh for CFD-GCN.')
#         nodes, edges, self.elems, self.marker_dict = get_mesh_graph(coarse_mesh)
#         self.nodes = torch.from_numpy(nodes).to(device)
#         if not freeze_mesh:
#             self.nodes = nn.Parameter(self.nodes) # Treat x,y- coordinates as parameters of the NN.
#         self.elems, new_edges = quad2tri(sum(self.elems, [])) # sum() is combining triads and quads into a single list. Inside quad2tri(), Quads are broken down to triads whereas triads are kept as they are.
#         self.elems = [self.elems]
#         self.edges = torch.from_numpy(edges).to(device)
#         # print(self.edges.dtype, new_edges.dtype)
#         self.edges = torch.cat([self.edges, new_edges.to(self.edges.device)], dim=1)
#         self.marker_inds = torch.tensor(sum(self.marker_dict.values(), [])).unique()
#         assert is_cw(self.nodes, self.elems[0]).nonzero().shape[0] == 0, 'Mesh has flipped elems'

        self.process_sim = process_sim
#         self.su2 = SU2Module(config_file, mesh_file=self.mesh_file)
#         logging.info(f'Mesh filename: {self.mesh_file.format(batch_index="*")}')

#         self.fine_marker_dict = torch.tensor(fine_marker_dict['airfoil']).unique()
        self.sdf = None
        
        self.saf, self.dsdf = saf, dsdf
        A_pow = A_pow*-1 if (A_shared) else A_pow #ADDED####

        improved = False
        self.num_convs = num_end_convs
        self.convs = []
        if self.num_convs > 0:
            self.convs = nn.ModuleList()
            in_channels = out_channels + hidden_channels*abs(A_pow)
            for i in range(self.num_convs - 1):
                #self.convs.append(GCNConv(in_channels, hidden_channels, improved=improved))
                self.convs.append(aGCN_conv(GCNConv,
                                      in_channels = in_channels,
                                      out_channels = hidden_channels,
                                      improved=improved,
                                      use_A = A_pow))
                in_channels = hidden_channels*abs(A_pow)
            #self.convs.append(GCNConv(in_channels, out_channels, improved=improved))
            self.convs.append(aGCN_conv(GCNConv,
                                      in_channels = in_channels,
                                      out_channels = out_channels,
                                      improved=improved,
                                      use_A = 1))

        self.num_pre_convs = num_convs - num_end_convs
        self.pre_convs = []
        if self.num_pre_convs > 0:
            in_channels = 5  # one extra channel for sdf
            in_channels += 2 if (self.saf==1) else 1 #ADDED
            in_channels += 8 if (self.dsdf==1) else 0 #ADDED
            self.pre_convs = nn.ModuleList()
            for i in range(self.num_pre_convs - 1):
                #self.pre_convs.append(GCNConv(in_channels, hidden_channels, improved=improved))
                self.pre_convs.append(aGCN_conv(GCNConv,
                                      in_channels = in_channels,
                                      out_channels = hidden_channels,
                                      improved=improved,
                                      use_A = A_pow))
                in_channels = hidden_channels*abs(A_pow)
            #self.pre_convs.append(GCNConv(in_channels, hidden_channels, improved=improved))
            self.pre_convs.append(aGCN_conv(GCNConv,
                                      in_channels = in_channels,
                                      out_channels = hidden_channels,
                                      improved=improved,
                                      use_A = A_pow))

        self.sim_info = {}  # store output of coarse simulation for logging / debugging

    def forward(self, batch):
        start = time.time()
        batch_size = batch.aoa.shape[0]
        
        fine_x = batch.x
        if self.saf==1:
            fine_x = torch.cat([fine_x, batch.saf], dim=1)
        else:
            # if self.sdf is None:
            #     with torch.no_grad():
            #         self.sdf = signed_dist_graph(batch.x[batch.batch == 0, :2],
            #                                      self.fine_marker_dict).unsqueeze(1)
            # fine_x = torch.cat([fine_x, self.sdf.repeat(batch_size, 1)], dim=1)
            fine_x = torch.cat([fine_x, batch.sdf], dim=1)
        if self.dsdf==1:
            fine_x = torch.cat([fine_x, batch.dsdf], dim=1)
            
        edge_index_list = [batch.edge_index, batch.edge_indexA2]

        for i, conv in enumerate(self.pre_convs):
            #fine_x = F.relu(conv(fine_x, batch.edge_index))
            fine_x = F.relu(conv(fine_x, edge_index_list, edge_attr_list=[None])) #EDITED


        nodes = batch.x[:,:2] #self.get_nodes()
        num_nodes = nodes.shape[0]
#         self.write_mesh_file(nodes, self.elems, self.marker_dict, filename=self.mesh_file)

#         params = torch.stack([batch.aoa, batch.mach_or_reynolds], dim=1)
#         batch_aoa = params[:, 0].to('cpu', non_blocking=True)
#         batch_mach_or_reynolds = params[:, 1].to('cpu', non_blocking=True)

#         batch_x = nodes.unsqueeze(0).expand(batch_size, -1, -1)
#         batch_x = batch_x.to('cpu', non_blocking=True)
#         batch_y = self.su2(batch_x[..., 0], batch_x[..., 1],
#                            batch_aoa[..., None], batch_mach_or_reynolds[..., None])
#         batch_y = [y.to(batch.x.device) for y in batch_y]
#         batch_y = self.process_sim(batch_y, False)

#         coarse_y = torch.stack([y.flatten() for y in batch_y], dim=1)
#         coarse_x = nodes.repeat(batch_size, 1)[:, :2]
#         zeros = batch.batch.new_zeros(num_nodes)
#         coarse_batch = torch.cat([zeros + i for i in range(batch_size)])

        #coarse_x, coarse_y = batch.coarse_x, batch.coarse_y #ADDED
        #coarse_data = torch.load(batch.coarse_path[0]) #ONLY LOADING FIRST OF BATCH
        coarse_data = [torch.load(batch.coarse_path[b]) for b in range(batch_size)]
        coarse_data = torch_geometric.data.Batch.from_data_list(coarse_data)
        
        coarse_x, coarse_y = coarse_data.x, self.process_sim(coarse_data.y)
        coarse_x, coarse_y = coarse_x.to(dtype=torch.float32), coarse_y.to(dtype=torch.float32)[:,:3] #no nut
        #coarse_batch = torch.zeros(coarse_x.size(0)) #ASSUMING BS=1
        coarse_batch = coarse_data.batch
        
        fine_y = self.upsample(coarse_y, coarse_x, coarse_batch, batch).to(fine_x.device)
        fine_y = torch.cat([fine_y, fine_x], dim=1)

        for i, conv in enumerate(self.convs[:-1]):
            #fine_y = F.relu(conv(fine_y, batch.edge_index))
            fine_y = F.relu(conv(fine_y, edge_index_list, edge_attr_list=[None])) #EDITED
        #fine_y = self.convs[-1](fine_y, batch.edge_index)
        fine_y = self.convs[-1](fine_y, edge_index_list, edge_attr_list=[None]) #EDITED

        self.sim_info['nodes'] = coarse_x[:, :2]
        #self.sim_info['elems'] = [self.elems] * batch_size
        self.sim_info['batch'] = coarse_batch
        self.sim_info['output'] = coarse_y

        return fine_y

    def upsample(self, y, coarse_nodes, coarse_batch, fine):
        fine_nodes = fine.x[:, :2]
        y = knn_interpolate(y.cpu(), coarse_nodes[:, :2].cpu(), fine_nodes.cpu(),
                            coarse_batch.cpu(), fine.batch.cpu(), k=3).to(y.device)
        return y

    def get_nodes(self):
        # return torch.cat([self.marker_nodes, self.not_marker_nodes])
        return self.nodes

    @staticmethod
    def write_mesh_file(x, elems, marker_dict, filename='mesh.su2'):
        write_graph_mesh(filename, x[:, :2], elems, marker_dict)

    @staticmethod
    def contiguous_elems_list(elems, inds):
        # Hack to easily have compatibility with MeshEdgePool
        return elems
    
#### FVnew Class ###


##############################
#### aGCN_conv CLASS DEF #####
##############################
#aGCN_conv CLASS codes, with use_A<0 if shared, adapted for dataparallel
class aGCN_conv(torch.nn.Module):
    '''#to use (e.g.):
    layer = aGCN_conv(use_A, conv_layer, **kwargs)'''
    def __init__(self, conv_layer, use_A=-2, **kwargs):
        super(aGCN_conv, self).__init__()
        self.layer_list = torch.nn.ModuleList(); self.use_A=use_A
        if use_A<=0: #shared parameters
          layer = conv_layer(**kwargs)
          self.layer_list.append(layer)
        else: #non-shared parameters
          for a in range(use_A):
            layer = conv_layer(**kwargs)
            self.layer_list.append(layer)

    def forward(self, x, edge_index_list, edge_attr_list=[None], node_attr=None): #data input for DataParallel listloader
      edge_index, edge_attr = edge_index_list[0], edge_attr_list[0]
    
      if edge_attr==None:
        x0 = self.layer_list[0](x, edge_index); #n=x.size(0)
        for a in range(abs(self.use_A)-1):
          edge_index, = edge_index_list[a+1], 
          if self.use_A>0: #non-shared parameters
            xa = self.layer_list[a+1](x, edge_index)
          else: #shared parameters
            xa = self.layer_list[0](x, edge_index)
          x0 = torch.cat((x0,xa),dim=1)
        
      else: #if using edge_attr
        x0 = self.layer_list[0](x,edge_index,edge_attr=edge_attr,node_attr=node_attr);
        for a in range(abs(self.use_A)-1):
          edge_index, edge_attr = edge_index_list[a+1], edge_attr_list[a+1]
          if self.use_A>0: #non-shared parameters
            xa = self.layer_list[a+1](x,edge_index,edge_attr=edge_attr,node_attr=node_attr)
          else: #shared parameters
            xa = self.layer_list[0](x,edge_index,edge_attr=edge_attr,node_attr=node_attr)
          x0 = torch.cat((x0,xa),dim=1)
      return x0


class FVnewConv(nng.MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_size, node_attr_num=0, edge_attr_num=0, dropout=0):
        """
        node_attr_num - number of node attributes
        edge_attr_num - number of edge attributes (0 or other integer #attr) ##ADDED
        in_channels - number of the input channels (node features)
        out_channels - number of the output channels (node features)
        hidden_size - number of the inner convolutions
        dropout - dropout rate after the layer
        """
        super(FVnewConv, self).__init__(aggr='add')
        self.dropout = dropout; self.attr_num = edge_attr_num
        #self.lin_in = torch.nn.Linear(coors, hidden_size * in_channels)
        ##EDITED
        in_channels += node_attr_num #For FVnew
        self.lin_in = torch.nn.Linear(self.attr_num, hidden_size * in_channels)
        self.lin_out = torch.nn.Linear(hidden_size * in_channels, out_channels)
        self.in_channels = in_channels

    def forward(self, x, edge_index, edge_attr=None, node_attr=None,):
        """
        x - feature matrix of the whole graph [num_nodes, label_dim]
        edge_index - graph connectivity [2, num_edges]
        edge_attr - edge attributes [num_edges, #attr] or none ##ADDED
        """
        if (node_attr is not None):
            x = torch.cat((x,node_attr),dim=1) #persistent node_attr.
        return self.propagate(edge_index=edge_index, edge_attr=edge_attr, x=x, aggr='add')  # [N, out_channels, label_dim]

    def message(self, x_j, edge_attr=None):
        """
        x_j [num_edges, label_dim]
        edge_attr [num_edges, #attr] or none ##ADDED
        """
        if edge_attr is not None:
          scaling = F.relu(self.lin_in(edge_attr))  # [n_edges, hidden_size * in_channels]
        else:
          size = (x_j.size(0),self.lin_in.out_features)
          scaling = torch.ones(size).unsqueeze(-1).to(x_j.device)
          # scaling = torch.ones(x_j.size()).unsqueeze(-1).to(x_j.device)
        n_edges = x_j.size(0)
        # [n_edges, in_channels, ...] * [n_edges, in_channels, 1]
        result = scaling.reshape(n_edges, self.in_channels, -1) * x_j.unsqueeze(-1)
        return result.view(n_edges, -1)

    def update(self, aggr_out):
        """
        aggr_out [num_nodes, label_dim, out_channels]
        """
        aggr_out = self.lin_out(aggr_out)  # [num_nodes, label_dim, out_features]
        aggr_out = torch.tanh(aggr_out); #aggr_out = F.relu(aggr_out)
        aggr_out = F.dropout(aggr_out, p=self.dropout, training=self.training)
        return aggr_out


class CFDFVnewGCN(nn.Module):
    def __init__(self, process_sim=lambda x, y: x, residual=False, #ADDED: add coarse mesh before output
                 freeze_mesh=False, num_convs=6, num_end_convs=3, hidden_channels=512, out_channels=3,
                 saf=False, dsdf=False, A_pow=2, A_shared='False', FV='False', device='cuda'):
        super().__init__()

        self.process_sim = process_sim
        self.sdf = None

        #ADDED:
        self.saf, self.dsdf = saf, dsdf
        self.FV = FV; self.residual = residual
        attr_num = 6 if FV else 0
        node_attr_num = 1 if FV else 0
        A_pow = A_pow*-1 if (A_shared) else A_pow #ADDED####

        improved = False
        self.num_convs = num_end_convs
        self.convs = []
        if self.num_convs > 0:
            self.convs = nn.ModuleList()
            in_channels = out_channels + hidden_channels*abs(A_pow)
            for i in range(self.num_convs - 1):
                #EDITED:
                self.convs.append(aGCN_conv(FVnewConv,
                                      in_channels = in_channels,
                                      out_channels = hidden_channels,
                                      node_attr_num = node_attr_num,
                                      edge_attr_num = attr_num,
                                      hidden_size = 3, 
                                      dropout = 0,
                                      use_A = A_pow))
                in_channels = hidden_channels*abs(A_pow)
            #EDITED:
            self.convs.append(aGCN_conv(FVnewConv,
                                      in_channels = in_channels,
                                      out_channels = out_channels,
                                      node_attr_num = node_attr_num,
                                      edge_attr_num = attr_num,
                                      hidden_size = 3, 
                                      dropout = 0,
                                      use_A = 1)) #EDITED#### #last one just A1

        self.num_pre_convs = num_convs - num_end_convs
        self.pre_convs = []
        if self.num_pre_convs > 0:
            in_channels = 5 # one extra channel for sdf
            in_channels += 2 if (self.saf==1) else 1 #ADDED
            in_channels += 8 if (self.dsdf==1) else 0#ADDED
            self.pre_convs = nn.ModuleList()
            for i in range(self.num_pre_convs - 1):
                #EDITED:
                self.pre_convs.append(aGCN_conv(FVnewConv,
                                      in_channels = in_channels,
                                      out_channels = hidden_channels,
                                      node_attr_num = node_attr_num,
                                      edge_attr_num = attr_num,
                                      hidden_size = 3, 
                                      dropout = 0,
                                      use_A = A_pow)) #EDITED####
                in_channels = hidden_channels*abs(A_pow)
            #EDITED:
            self.pre_convs.append(aGCN_conv(FVnewConv,
                                      in_channels = in_channels,
                                      out_channels = hidden_channels,
                                      node_attr_num = node_attr_num,
                                      edge_attr_num = attr_num,
                                      hidden_size = 3, 
                                      dropout = 0,
                                      use_A = A_pow))

        self.sim_info = {}  # store output of coarse simulation for logging / debugging

    def forward(self, batch):
        start = time.time()
        batch_size = batch.aoa.shape[0]

        fine_x = batch.x
        if self.saf==1:
            fine_x = torch.cat([fine_x, batch.saf], dim=1)
        else:
            fine_x = torch.cat([fine_x, batch.sdf], dim=1)
        if self.dsdf==1:
            fine_x = torch.cat([fine_x, batch.dsdf], dim=1)
        
        edge_index_list = [batch.edge_index] #, batch.edge_indexA2]
        edge_attr_list = [batch.edge_attr] if self.FV else [None] #, batch.edge_attrA2]
        node_attr = batch.node_attr if self.FV else [None] #ADDED for FVnew

        for i, conv in enumerate(self.pre_convs):
            #fine_x = F.relu(conv(fine_x, edge_index_list, edge_attr_list=edge_attr_list)) #EDITED
            fine_x = F.relu(conv(fine_x, edge_index_list, edge_attr_list=edge_attr_list, node_attr=node_attr))

        nodes = batch.x[:,:2] #self.get_nodes()
        num_nodes = nodes.shape[0]

        params = torch.stack([batch.aoa, batch.mach_or_reynolds], dim=1)
        batch_aoa = params[:, 0].to('cpu', non_blocking=True)
        batch_mach_or_reynolds = params[:, 1].to('cpu', non_blocking=True)

        coarse_data = [torch.load(batch.coarse_path[b]) for b in range(batch_size)]
        coarse_data = torch_geometric.data.Batch.from_data_list(coarse_data)
        
        coarse_x, coarse_y = coarse_data.x, self.process_sim(coarse_data.y)
        coarse_x, coarse_y = coarse_x.to(dtype=torch.float32), coarse_y.to(dtype=torch.float32)[:,:3] #no nut
        coarse_batch = coarse_data.batch
        
        fine_y_orig = self.upsample(coarse_y, coarse_x, coarse_batch, batch).to(fine_x.device)
        fine_y = torch.cat([fine_y_orig, fine_x], dim=1)

        for i, conv in enumerate(self.convs[:-1]):
            # fine_y = F.relu(conv(fine_y, edge_index_list, edge_attr_list=edge_attr_list)) #EDITED
            fine_y = F.relu(conv(fine_y, edge_index_list, edge_attr_list=edge_attr_list, node_attr=node_attr))
        # fine_y = self.convs[-1](fine_y, edge_index_list, edge_attr_list=edge_attr_list) #EDITED
        fine_y = self.convs[-1](fine_y, edge_index_list, edge_attr_list=edge_attr_list, node_attr=node_attr)

        self.sim_info['nodes'] = coarse_x[:, :2]
        self.sim_info['batch'] = coarse_batch
        self.sim_info['output'] = coarse_y

        return (fine_y + fine_y_orig) if self.residual else fine_y

    def upsample(self, y, coarse_nodes, coarse_batch, fine):
        fine_nodes = fine.x[:, :2]
        y = knn_interpolate(y.cpu(), coarse_nodes[:, :2].cpu(), fine_nodes.cpu(),
                            coarse_batch.cpu(), fine.batch.cpu(), k=3).to(y.device)
        return y

    def get_nodes(self):
        # return torch.cat([self.marker_nodes, self.not_marker_nodes])
        return self.nodes

    @staticmethod
    def write_mesh_file(x, elems, marker_dict, filename='mesh.su2'):
        write_graph_mesh(filename, x[:, :2], elems, marker_dict)

    @staticmethod
    def contiguous_elems_list(elems, inds):
        # Hack to easily have compatibility with MeshEdgePool
        return elems
    

######## NEW CLASS #######

##############################
## SAGE_attr_Conv CLASS DEF ##
##############################
#SAGE_attr_Conv codes, with node_attr_num and edge_attr_num
class SAGE_attr_Conv(nng.SAGEConv):
    """Convolution class built on SAGEConv, with SGCN and FVnew adaptations"""
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        node_attr_num: int = 0, #FVnew: number of persistent node_attr
        edge_attr_num: int = 0, #FVnew: number of edge_attr + rel_node_attr
        hidden_size: int = 1, #sGCN: convolution internal hidden size
        dropout: float = 0.3, #sGCN: dropout
        **kwargs,
    ):
        in_channels += node_attr_num
        self.in_channels = (out_channels,in_channels)
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        super().__init__(self.in_channels,self.out_channels,aggr,self.normalize,\
                         self.root_weight,self.project,bias,**kwargs)
        
        #FVnew or sGCN####
        self.lin_in = torch.nn.Linear(edge_attr_num, hidden_size * in_channels)
        self.lin_out = torch.nn.Linear(hidden_size * in_channels, out_channels)
        self.in_channels = in_channels
        self.dropout = dropout ####

    def forward(self, x, edge_index, edge_attr=None, node_attr=None, size=None):
        """"""
        if (node_attr is not None):
            x = torch.cat((x,node_attr),dim=1) #persistent node_attr.

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # #propagate_type: (x: OptPairTensor)
        # out = self.propagate(edge_index, x=x, size=size) #sGCN: propagate.
        out = self.propagate(edge_index=edge_index, edge_attr=edge_attr, x=x, aggr='add')
        # [N, out_channels, label_dim]
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j, edge_attr=None):
        """
        x_j [num_edges, label_dim]
        edge_attr [num_edges, #attr] or none ##ADDED
        """
        if edge_attr is not None:
          scaling = F.relu(self.lin_in(edge_attr))  # [n_edges, hidden_size * in_channels]
        else:
          size = (x_j.size(0),self.lin_in.out_features)
          scaling = torch.ones(size).unsqueeze(-1).to(x_j.device)
        n_edges = x_j.size(0)
        # [n_edges, in_channels, ...] * [n_edges, in_channels, 1]
        result = scaling.reshape(n_edges, self.in_channels, -1) * x_j.unsqueeze(-1)
        return result.view(n_edges, -1)
    
    def update(self, aggr_out):
        """
        aggr_out [num_nodes, label_dim, out_channels]
        """
        aggr_out = self.lin_out(aggr_out)  # [num_nodes, label_dim, out_features]
        aggr_out = torch.tanh(aggr_out); #aggr_out = F.relu(aggr_out)
        aggr_out = F.dropout(aggr_out, p=self.dropout, training=self.training)
        return aggr_out
    

class CFDFVnewGSAGE(nn.Module):
    def __init__(self, process_sim=lambda x, y: x, residual=False, #ADDED: add coarse mesh before output
                 freeze_mesh=False, num_convs=6, num_end_convs=3, hidden_channels=512, out_channels=3,
                 saf=False, dsdf=False, A_pow=2, A_shared='False', FV='False', device='cuda'):
        super().__init__()

        self.process_sim = process_sim
        self.sdf = None

        #ADDED:
        self.saf, self.dsdf = saf, dsdf
        self.FV = FV; self.residual = residual
        attr_num = 6 if FV else 0
        node_attr_num = 1 if FV else 0
        A_pow = A_pow*-1 if (A_shared) else A_pow #ADDED####

        improved = False
        self.num_convs = num_end_convs
        self.convs = []
        if self.num_convs > 0:
            self.convs = nn.ModuleList()
            in_channels = out_channels + hidden_channels*abs(A_pow)
            for i in range(self.num_convs - 1):
                #EDITED:
                if FV:
                    self.convs.append(aGCN_conv(SAGE_attr_Conv,
                                          in_channels = in_channels,
                                          out_channels = hidden_channels,
                                          node_attr_num = node_attr_num,
                                          edge_attr_num = attr_num,
                                          hidden_size = 3, 
                                          dropout = 0,
                                          use_A = A_pow))
                else:
                    self.convs.append(aGCN_conv(nng.SAGEConv,
                                          in_channels = in_channels,
                                          out_channels = hidden_channels,
                                          dropout = 0,
                                          use_A = A_pow))
                in_channels = hidden_channels*abs(A_pow)
            #EDITED:
            if FV:
                self.convs.append(aGCN_conv(SAGE_attr_Conv,
                                          in_channels = in_channels,
                                          out_channels = out_channels,
                                          node_attr_num = node_attr_num,
                                          edge_attr_num = attr_num,
                                          hidden_size = 3, 
                                          dropout = 0,
                                          use_A = 1)) #EDITED#### #last one just A1
            else:
                self.convs.append(aGCN_conv(nng.SAGEConv,
                                          in_channels = in_channels,
                                          out_channels = out_channels,
                                          dropout = 0,
                                          use_A = 1)) #EDITED#### #last one just A1

        self.num_pre_convs = num_convs - num_end_convs
        self.pre_convs = []
        if self.num_pre_convs > 0:
            in_channels = 5 # one extra channel for sdf
            in_channels += 2 if (self.saf==1) else 1 #ADDED
            in_channels += 8 if (self.dsdf==1) else 0#ADDED
            self.pre_convs = nn.ModuleList()
            for i in range(self.num_pre_convs - 1):
                #EDITED:
                if FV:
                    self.pre_convs.append(aGCN_conv(SAGE_attr_Conv,
                                          in_channels = in_channels,
                                          out_channels = hidden_channels,
                                          node_attr_num = node_attr_num,
                                          edge_attr_num = attr_num,
                                          hidden_size = 3, 
                                          dropout = 0,
                                          use_A = A_pow)) #EDITED####
                else:
                    self.pre_convs.append(aGCN_conv(nng.SAGEConv,
                                          in_channels = in_channels,
                                          out_channels = hidden_channels,
                                          dropout = 0,
                                          use_A = A_pow))
                in_channels = hidden_channels*abs(A_pow)
            #EDITED:
            if FV:
                self.pre_convs.append(aGCN_conv(SAGE_attr_Conv,
                                          in_channels = in_channels,
                                          out_channels = hidden_channels,
                                          node_attr_num = node_attr_num,
                                          edge_attr_num = attr_num,
                                          hidden_size = 3, 
                                          dropout = 0,
                                          use_A = A_pow))
            else:
                self.pre_convs.append(aGCN_conv(nng.SAGEConv,
                                          in_channels = in_channels,
                                          out_channels = hidden_channels,
                                          dropout = 0,
                                          use_A = A_pow))

        self.sim_info = {}  # store output of coarse simulation for logging / debugging

    def forward(self, batch):
        start = time.time()
        batch_size = batch.aoa.shape[0]

        fine_x = batch.x
        if self.saf==1:
            fine_x = torch.cat([fine_x, batch.saf], dim=1)
        else:
            fine_x = torch.cat([fine_x, batch.sdf], dim=1)
        if self.dsdf==1:
            fine_x = torch.cat([fine_x, batch.dsdf], dim=1)
        
        edge_index_list = [batch.edge_index] #, batch.edge_indexA2]
        edge_attr_list = [batch.edge_attr] if self.FV else [None] #, batch.edge_attrA2]
        node_attr = batch.node_attr if self.FV else [None] #ADDED for FVnew

        for i, conv in enumerate(self.pre_convs):
            #fine_x = F.relu(conv(fine_x, edge_index_list, edge_attr_list=edge_attr_list)) #EDITED
            fine_x = F.relu(conv(fine_x, edge_index_list, edge_attr_list=edge_attr_list, node_attr=node_attr))

        nodes = batch.x[:,:2] #self.get_nodes()
        num_nodes = nodes.shape[0]

        params = torch.stack([batch.aoa, batch.mach_or_reynolds], dim=1)
        batch_aoa = params[:, 0].to('cpu', non_blocking=True)
        batch_mach_or_reynolds = params[:, 1].to('cpu', non_blocking=True)

        coarse_data = [torch.load(batch.coarse_path[b]) for b in range(batch_size)]
        coarse_data = torch_geometric.data.Batch.from_data_list(coarse_data)
        
        coarse_x, coarse_y = coarse_data.x, self.process_sim(coarse_data.y)
        coarse_x, coarse_y = coarse_x.to(dtype=torch.float32), coarse_y.to(dtype=torch.float32)[:,:3] #no nut
        coarse_batch = coarse_data.batch
        
        fine_y_orig = self.upsample(coarse_y, coarse_x, coarse_batch, batch).to(fine_x.device)
        fine_y = torch.cat([fine_y_orig, fine_x], dim=1)

        for i, conv in enumerate(self.convs[:-1]):
            # fine_y = F.relu(conv(fine_y, edge_index_list, edge_attr_list=edge_attr_list)) #EDITED
            fine_y = F.relu(conv(fine_y, edge_index_list, edge_attr_list=edge_attr_list, node_attr=node_attr))
        # fine_y = self.convs[-1](fine_y, edge_index_list, edge_attr_list=edge_attr_list) #EDITED
        fine_y = self.convs[-1](fine_y, edge_index_list, edge_attr_list=edge_attr_list, node_attr=node_attr)

        self.sim_info['nodes'] = coarse_x[:, :2]
        self.sim_info['batch'] = coarse_batch
        self.sim_info['output'] = coarse_y

        return (fine_y + fine_y_orig) if self.residual else fine_y

    def upsample(self, y, coarse_nodes, coarse_batch, fine):
        fine_nodes = fine.x[:, :2]
        y = knn_interpolate(y.cpu(), coarse_nodes[:, :2].cpu(), fine_nodes.cpu(),
                            coarse_batch.cpu(), fine.batch.cpu(), k=3).to(y.device)
        return y

    def get_nodes(self):
        # return torch.cat([self.marker_nodes, self.not_marker_nodes])
        return self.nodes

    @staticmethod
    def write_mesh_file(x, elems, marker_dict, filename='mesh.su2'):
        write_graph_mesh(filename, x[:, :2], elems, marker_dict)

    @staticmethod
    def contiguous_elems_list(elems, inds):
        # Hack to easily have compatibility with MeshEdgePool
        return elems