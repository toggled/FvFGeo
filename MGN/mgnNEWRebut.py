import torch
from torch import nn
import copy
from torch_geometric.data import Data
from mlp import MLP
from mgn_convNEWRebut import MeshGraphNetsConv



class Encoder(nn.Module):
    """MeshGraphNets Encoder.
    The encoder must take a PyG graph object `data` and output the same `data`
    with additional fields `h_node` and `h_edge` that correspond to the node and edge embedding.
    """
    def __init__(self, node_dim, edge_dim, hidden_size): # 3,3,128
        super().__init__()
        self.hidden_size = hidden_size
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.embed_node = nn.Sequential(
            MLP([node_dim, hidden_size, hidden_size], act=nn.ReLU()),
            nn.LayerNorm(hidden_size),
        )
        self.embed_edge = nn.Sequential(
            MLP([edge_dim, hidden_size, hidden_size], act=nn.ReLU()),
            nn.LayerNorm(hidden_size),
        )
    
    def forward(self, data):
        # Embed nodes
        data.h_node = self.embed_node(data.x) # node embedding

        # Embed edges
        e_ij = data.edge_attr
        # h_edge = torch.cat([e_ij, e_ij.norm(dim=-1, keepdim=True)], dim=-1)
        data.h_edge = self.embed_edge(e_ij) # edge embedding
        
        return data


class Processor(nn.Module):
    """MeshGraphNets Processor.
    The processor updates both node and edge embeddings `data.h_node`, `data.h_edge`.
    """
    def __init__(self, node_dim, edge_dim, num_convs=15, use_FV=True): # 15 (from paper)
        super().__init__()
        self.num_convs = num_convs
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        self.use_FV = use_FV
        self.convs = nn.ModuleList(
            [copy.deepcopy(MeshGraphNetsConv(node_dim, edge_dim, use_FV=use_FV)) for _ in range(num_convs)]
        )

    def forward(self, data):
        for conv in self.convs:
            
            if self.use_FV:
                data.h_node = torch.cat((data.h_node,data.node_FVattr),dim=1)
                data.h_edge = torch.cat((data.h_edge,data.edge_FVattr),dim=1)
                
            data.h_node, data.h_edge = conv(data.h_node, data.edge_index, data.h_edge)
        return data


class Decoder(nn.Module):
    """MeshGraphNets Decoder.
    This decoder only operates on the node embedding `data.h_node`.
    """
    def __init__(self, node_dim, out_dim):
        super().__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.decoder = MLP([node_dim, node_dim, out_dim], act=nn.ReLU())
    
    def forward(self, data):
        return self.decoder(data.h_node)


class FVMeshGraphNets(nn.Module):
    """MeshGraphNets.
    """
    def __init__(self, node_input_size=3, edge_input_size=3,\
                  hidden_size=128, use_FV=True):
        super().__init__()
        
        self.use_FV = use_FV
        if use_FV:
            self.encoder   = Encoder(node_dim=node_input_size + 1,\
                                    edge_dim=edge_input_size + 6,
                                    hidden_size=hidden_size)
            self.processor = Processor(node_dim=hidden_size,edge_dim=hidden_size, use_FV=use_FV)
            self.decoder   = Decoder(node_dim=hidden_size + 1,out_dim=2) # out_size = u,v
        else:
            self.encoder   = Encoder(node_dim=node_input_size,\
                                edge_dim=edge_input_size,
                                hidden_size=hidden_size)
            self.processor = Processor(node_dim=hidden_size,edge_dim=hidden_size, use_FV=use_FV)
            self.decoder   = Decoder(node_dim=hidden_size,out_dim=2) # out_size = u,v
    
    def forward(self, data):
        if self.use_FV:
            data.x = torch.cat((data.x,data.node_FVattr),dim=1)
            data.edge_attr = torch.cat((data.edge_attr,data.edge_FVattr),dim=1)
        data = self.encoder(data)
        
        data = self.processor(data)
        
        if self.use_FV:
                data.h_node = torch.cat((data.h_node,data.node_FVattr),dim=1)
        return self.decoder(data)

