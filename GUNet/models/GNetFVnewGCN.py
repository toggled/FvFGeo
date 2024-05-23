
##@title GNetFVnewGraphSAGE: with directional normal as edge_attr, and persistent node_attr.

import torch.nn as nn
import torch_geometric.nn as nng
import torch
import random
#from torch_geometric.utils import add_self_loops #ADDED!
import torch.nn.functional as F #ADDED

from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size

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

    
#########################
## FVnewConv CLASS DEF ##
#########################
#FVnewConv codes, with node_attr_num and edge_attr_num
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

##############################
## GNetFVnewGCN CLASS ##
##############################
#GNetFVnewGCN, with persistent node_attr and edge_attr
class GNetFVnewGCN(nn.Module):
    def __init__(self, hparams, encoder, decoder):
        super(GNetFVnewGCN, self).__init__()

        self.L = hparams['nb_scale']
        self.layer = hparams['layer']
        # self.pool_type = hparams['pool']
        # self.pool_ratio = hparams['pool_ratio']
        # self.list_r = hparams['list_r']
        self.size_hidden_layers = hparams['size_hidden_layers']
        self.size_hidden_layers_init = hparams['size_hidden_layers']
        self.max_neighbors = hparams['max_neighbors']
        self.dim_enc = hparams['encoder'][-1]
        self.bn_bool = hparams['batchnorm']
        self.res = hparams['res']
        self.head = 2
        self.activation = nn.ReLU()

        self.SAF = hparams['SAF']
        self.dSDF = hparams['dSDF']
        self.FV = hparams['FV']
        hidd_FV_size = hparams['hidd_FV_size'] #3 is good.
        attr_num = 6 if hparams['FV'] else 0 #dir-normal, rel-pos of nodes to face centre
        node_attr_num = 1 if hparams['FV'] else 0 #cell volume
        A_pow = hparams['A_pow']
        A_pow = A_pow*-1 if hparams['A_shared'] else A_pow #ADDED####

        self.encoder = encoder
        self.decoder = decoder

        self.down_layers = nn.ModuleList()

        # if self.pool_type != 'random':
        #     self.pool = nn.ModuleList()
        # else:
        #     self.pool = None

        if self.layer == 'SAGE':
            # self.down_layers.append(nng.SAGEConv(
            #     in_channels = self.dim_enc,
            #     out_channels = self.size_hidden_layers
            # ))
            self.down_layers.append(aGCN_conv(nng.SAGEConv,
                                  in_channels = self.dim_enc,
                                  out_channels = self.size_hidden_layers//abs(A_pow),
                                  use_A = A_pow))
            bn_in = self.size_hidden_layers
        
        elif self.layer == 'GCN':
            self.down_layers.append(nng.GCNConv(
                in_channels = self.dim_enc,
                out_channels = self.size_hidden_layers,
                add_self_loops = False
            ))
            bn_in = self.size_hidden_layers

        elif self.layer == 'GAT':
            self.down_layers.append(nng.GATConv(
                in_channels = self.dim_enc,
                out_channels = self.size_hidden_layers,
                heads = self.head,
                add_self_loops = False,
                concat = True
            ))
            bn_in = self.head*self.size_hidden_layers
        
        elif self.layer == 'fvGCN': #ADDED
              self.down_layers.append(aGCN_conv(FVnewConv,
                                      in_channels = self.dim_enc,
                                      out_channels = self.size_hidden_layers//abs(A_pow),
                                      node_attr_num = node_attr_num,
                                      edge_attr_num = attr_num,
                                      hidden_size = hidd_FV_size, #3,
                                      dropout = 0,
                                      use_A = A_pow))
              bn_in = self.size_hidden_layers #ADDED
        
        else: #ADDED #'fvSAGE'
              self.down_layers.append(aGCN_conv(SAGE_attr_Conv,
                                      in_channels = self.dim_enc,
                                      out_channels = self.size_hidden_layers//abs(A_pow),
                                      node_attr_num = node_attr_num,
                                      edge_attr_num = attr_num,
                                      hidden_size = hidd_FV_size, #3,
                                      dropout = 0,
                                      use_A = A_pow))
              bn_in = self.size_hidden_layers #ADDED

        if self.bn_bool == True:
            self.bn = nn.ModuleList()
            self.bn.append(nng.BatchNorm(
                in_channels = bn_in,
                track_running_stats = False
            ))
        else:
            self.bn = None


        for n in range(1, self.L):
        #     if self.pool_type != 'random':
        #         self.pool.append(nng.TopKPooling(
        #             in_channels = self.size_hidden_layers,
        #             ratio = self.pool_ratio[n - 1],
        #             nonlinearity = torch.sigmoid
        #         ))

            if self.layer == 'SAGE':
                # self.down_layers.append(nng.SAGEConv(
                #     in_channels = self.size_hidden_layers,
                #     out_channels = 2*self.size_hidden_layers,
                # ))
                self.down_layers.append(aGCN_conv(nng.SAGEConv,
                                  in_channels = self.size_hidden_layers,
                                  out_channels = 2*self.size_hidden_layers//abs(A_pow),
                                  use_A = A_pow))
                self.size_hidden_layers = 2*self.size_hidden_layers
                bn_in = self.size_hidden_layers

            elif self.layer == 'GAT':
                self.down_layers.append(nng.GATConv(
                    in_channels = self.head*self.size_hidden_layers,
                    out_channels = self.size_hidden_layers,
                    heads = 2,
                    add_self_loops = False,
                    concat = True
                ))
            
            elif self.layer == 'fvGCN': #ADDED
                self.down_layers.append(aGCN_conv(FVnewConv,
                                      in_channels = self.size_hidden_layers,
                                      out_channels = 2*self.size_hidden_layers//abs(A_pow),
                                      node_attr_num = node_attr_num,
                                      edge_attr_num = attr_num,
                                      hidden_size = hidd_FV_size, #3,
                                      dropout = 0,
                                      use_A = A_pow))
                self.size_hidden_layers = 2*self.size_hidden_layers
                bn_in = self.size_hidden_layers #ADDED
            
            else: #ADDED
                self.down_layers.append(aGCN_conv(SAGE_attr_Conv,
                                      in_channels = self.size_hidden_layers,
                                      out_channels = 2*self.size_hidden_layers//abs(A_pow),
                                      node_attr_num = node_attr_num,
                                      edge_attr_num = attr_num,
                                      hidden_size = hidd_FV_size, #3,
                                      dropout = 0,
                                      use_A = A_pow))
                self.size_hidden_layers = 2*self.size_hidden_layers
                bn_in = self.size_hidden_layers #ADDED

            if self.bn_bool == True:
                self.bn.append(nng.BatchNorm(
                    in_channels = bn_in,
                    track_running_stats = False
                ))

        self.up_layers = nn.ModuleList()

        if self.layer == 'SAGE':
            # self.up_layers.append(nng.SAGEConv(
            #     in_channels = 3*self.size_hidden_layers_init,
            #     out_channels = self.dim_enc
            # ))
            self.up_layers.append(aGCN_conv(nng.SAGEConv,
                                  in_channels = 3*self.size_hidden_layers_init,
                                  out_channels = self.dim_enc//abs(A_pow),
                                  use_A = A_pow))
            self.size_hidden_layers_init = 2*self.size_hidden_layers_init

        elif self.layer == 'GAT':
            self.up_layers.append(nng.GATConv(
                in_channels = 2*self.head*self.size_hidden_layers,
                out_channels = self.dim_enc,
                heads = 2,
                add_self_loops = False,
                concat = False
            ))

        elif self.layer == 'fvGCN': #ADDED
            self.up_layers.append(aGCN_conv(FVnewConv,
                                  in_channels = 3*self.size_hidden_layers_init,
                                  out_channels = self.dim_enc, #//abs(A_pow),
                                  node_attr_num = node_attr_num,
                                  edge_attr_num = attr_num,
                                  hidden_size = hidd_FV_size, #3,
                                  dropout = 0,
                                  use_A = 1))
            self.size_hidden_layers_init = 2*self.size_hidden_layers_init #ADDED
        
        else: #ADDED
            self.up_layers.append(aGCN_conv(SAGE_attr_Conv,
                                    in_channels = 3*self.size_hidden_layers_init,
                                    out_channels = self.dim_enc, #//abs(A_pow),
                                    node_attr_num = node_attr_num,
                                    edge_attr_num = attr_num,
                                    #attr_num = attr_num, #or 2????
                                    hidden_size = hidd_FV_size, #3,
                                    dropout = 0,
                                    use_A = 1)) #A_pow)) #final layer only uses A1.
            self.size_hidden_layers_init = 2*self.size_hidden_layers_init #ADDED

        if self.bn_bool == True:
                self.bn.append(nng.BatchNorm(
                    in_channels = self.dim_enc,
                    track_running_stats = False
                ))

        for n in range(1, self.L - 1):
            if self.layer == 'SAGE':
                # self.up_layers.append(nng.SAGEConv(
                #     in_channels = 3*self.size_hidden_layers_init,
                #     out_channels = self.size_hidden_layers_init,
                # ))
                self.up_layers.append(aGCN_conv(nng.SAGEConv,
                                  in_channels = 3*self.size_hidden_layers_init,
                                  out_channels = self.size_hidden_layers_init//abs(A_pow),
                                  use_A = A_pow))
                bn_in = self.size_hidden_layers_init
                self.size_hidden_layers_init = 2*self.size_hidden_layers_init                

            elif self.layer == 'GAT':
                self.up_layers.append(nng.GATConv(
                    in_channels = 2*self.head*self.size_hidden_layers,
                    out_channels = self.size_hidden_layers,
                    heads = 2,
                    add_self_loops = False,
                    concat = True
                ))

            elif self.layer == 'fvGCN': #ADDED
                self.up_layers.append(aGCN_conv(FVnewConv,
                                      in_channels = 3*self.size_hidden_layers_init,
                                      out_channels = self.size_hidden_layers_init//abs(A_pow),
                                      node_attr_num = node_attr_num,
                                      edge_attr_num = attr_num,
                                      hidden_size = hidd_FV_size, #3,
                                      dropout = 0,
                                      use_A = A_pow))
                bn_in = self.size_hidden_layers_init
                self.size_hidden_layers_init = 2*self.size_hidden_layers_init #ADDED
            
            else: #ADDED
                self.up_layers.append(aGCN_conv(SAGE_attr_Conv,
                                      in_channels = 3*self.size_hidden_layers_init,
                                      out_channels = self.size_hidden_layers_init//abs(A_pow),
                                      node_attr_num = node_attr_num,
                                      edge_attr_num = attr_num,
                                      hidden_size = hidd_FV_size, #3,
                                      dropout = 0,
                                      use_A = A_pow))
                bn_in = self.size_hidden_layers_init
                self.size_hidden_layers_init = 2*self.size_hidden_layers_init #ADDED

            if self.bn_bool == True:
                self.bn.append(nng.BatchNorm(
                    in_channels = bn_in,
                    track_running_stats = False
                ))

    def forward(self, data):
        #x, edge_index = data.x, data.edge_index
        x = data.x.clone()
        edge_index_list = [data.edge_index.clone()] #, data.edge_indexA2.clone()]
        if self.FV:
            edge_attr_list = [data.edge_attr.clone()] #,  data.edge_attrA2.clone()]
            node_attr = data.node_attr #ADDED: for persistent node_attr
        else:
            edge_attr_list = [None]; node_attr = None

        id = []
        # edge_index_list = [edge_index.clone()]
        # pos_x_list = []
        z = self.encoder(x)
        if self.res:
            z_res = z.clone()

        #EDITED:
        #z = self.down_layers[0](z, edge_index)
        if (self.layer == 'SAGE') or (self.layer == 'GAT'):
          #z = self.down_layers[0](z, edge_index)
          z = self.down_layers[0](z, edge_index_list)
        else:
          #z = self.down_layers[0](z, edge_index_list, edge_attr_list=edge_attr_list)
          z = self.down_layers[0](z, edge_index_list, edge_attr_list=edge_attr_list, node_attr=node_attr)
        #EDITED####

        if self.bn_bool == True:
            z = self.bn[0](z)

        z = self.activation(z)
        z_list = [z.clone()]
        for n in range(self.L - 1):
            # pos_x = x[:, :2] if n == 0 else pos_x[id[n - 1]]
            # pos_x_list.append(pos_x.clone())
            # if self.pool_type != 'random':
            #     z, edge_index, node_attr = DownSample(id, z, edge_index, pos_x, self.pool[n], self.pool_ratio[n], self.list_r[n],self.max_neighbors, node_attr)
            # else:
            #     z, edge_index, node_attr = DownSample(id, z, edge_index, pos_x, None, self.pool_ratio[n], self.list_r[n], self.max_neighbors, node_attr)
            # edge_index_list.append(edge_index.clone())

            #EDITED:
            #z = self.down_layers[n + 1](z, edge_index)
            if (self.layer == 'SAGE') or (self.layer == 'GAT'):
              #z = self.down_layers[n + 1](z, edge_index)
              z = self.down_layers[n + 1](z, edge_index_list)
            else:
              #z = self.down_layers[n + 1](z, edge_index_list, edge_attr_list=edge_attr_list)
              z = self.down_layers[n + 1](z, edge_index_list, edge_attr_list=edge_attr_list, node_attr=node_attr)
            #EDITED####

            if self.bn_bool == True:
                z = self.bn[n + 1](z)

            z = self.activation(z)
            z_list.append(z.clone())
        #pos_x_list.append(pos_x[id[-1]].clone())
        
        for n in range(self.L - 1, 0, -1):
            # z = UpSample(z, pos_x_list[n - 1], pos_x_list[n])
            z = torch.cat([z, z_list[n - 1]], dim = 1)
            #EDITED:
            #z = self.up_layers[n - 1](z, edge_index_list[n - 1])
            if (self.layer == 'SAGE') or (self.layer == 'GAT'):
              #z = self.up_layers[n - 1](z, edge_index_list[n - 1])
              z = self.up_layers[n - 1](z, edge_index_list)
            else:
              #z = self.up_layers[n - 1](z, edge_index_list, edge_attr_list=edge_attr_list)
              z = self.up_layers[n - 1](z, edge_index_list, edge_attr_list=edge_attr_list, node_attr=node_attr)
            #EDITED####

            if self.bn_bool == True:
                z = self.bn[self.L + n - 1](z)

            z = self.activation(z) if n != 1 else z

        # del(z_list, pos_x_list)

        if self.res:
            z = z + z_res

        z = self.decoder(z)

        return z


