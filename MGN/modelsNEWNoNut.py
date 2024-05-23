#@title CFDGCN and CFDFVGCN to be geometry-general and not train SU2 coarse mesh, with bs>1
import os
import logging
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn

from mgn import *
from mgnNEWRebut import *

class MGN(nn.Module):
    # def __init__(self, in_channels, hidden_channels, out_channels, num_layers=15, \
    #              saf = False, dsdf = False, device='cuda'):
    def __init__(self, node_channels, edge_channels, hidden_channels, \
                 saf = False, dsdf = False):
        super().__init__()
        self.saf = saf; self.dsdf = dsdf
        if saf:
            node_channels+=2
        if dsdf:
            node_channels+=8
        # print('node channels => ',node_channels)
        self.gnn = MeshGraphNets(node_input_size=node_channels, \
                edge_input_size=edge_channels,hidden_size=hidden_channels)

    def forward(self, data):
        if self.saf:
            data.x = torch.cat((data.x,data.saf),dim=1)
        if self.dsdf:
            data.x = torch.cat((data.x,data.dsdf),dim=1)
        # print('mgn forward() => ', data)
        return self.gnn(data)
    
    
class FVMGN(nn.Module):
    def __init__(self, node_channels, edge_channels, hidden_channels, \
                 saf = False, dsdf = False, use_FV = True):
        super().__init__()

        self.saf = saf; self.dsdf = dsdf
        if saf:
            node_channels+=2
        if dsdf:
            node_channels+=8
        
        self.gnn = FVMeshGraphNets(node_input_size=node_channels, \
                edge_input_size=edge_channels,hidden_size=hidden_channels, use_FV=use_FV)

    def forward(self, data):
        if self.saf:
            data.x = torch.cat((data.x,data.saf),dim=1)
        if self.dsdf:
            data.x = torch.cat((data.x,data.dsdf),dim=1)
        return self.gnn(data)
    
    
class FVMGN_residual(nn.Module):
    def __init__(self, node_channels, edge_channels, hidden_channels, \
                 saf = False, dsdf = False, use_FV = True, use_res = True):
        super().__init__()

        self.saf = saf; self.dsdf = dsdf; self.res = use_res
        if saf:
            node_channels+=2
        if dsdf:
            node_channels+=8
        if use_res:
            node_channels+=2
            
        self.gnn = FVMeshGraphNets(node_input_size=node_channels, \
                edge_input_size=edge_channels,hidden_size=hidden_channels, use_FV=use_FV)

    def forward(self, data):
        if self.saf:
            data.x = torch.cat((data.x,data.saf),dim=1)
        if self.dsdf:
            data.x = torch.cat((data.x,data.dsdf),dim=1)
        
        if self.res:
            # coarse_data = torch.load(data.coarse_path[0])
            # coarse_x, coarse_y = coarse_data.x, test_data.coarse_preprocess(coarse_data.y)
            # coarse_x, coarse_y = coarse_x.to(dtype=torch.float32), coarse_y.to(dtype=torch.float32)[:,:3] #no nut
            # coarse_batch = coarse_data.batch
            # fine_y_orig = knn_interpolate(coarse_y, coarse_x[:, :2], data.x[:, :2], k=3)
            # estimate = fine_y_orig
            
            data.x = torch.cat((data.x,data.estimate),dim=1)
            
        output = self.gnn(data)
        if self.res:
            output+= data.estimate
            
        return output
    