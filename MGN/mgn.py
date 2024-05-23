import torch
from torch import nn
import copy
from torch_geometric.data import Data
from mlp import MLP
from mgn_conv import MeshGraphNetsConv


# class Encoder(nn.Module):
#     """MeshGraphNets Encoder.
#     The encoder must take a PyG graph object `data` and output the same `data`
#     with additional fields `h_node` and `h_edge` that correspond to the node and edge embedding.
#     """
#     def __init__(self,
#                 edge_input_size=3, # data.y.shape[1]
#                 node_input_size=3, # data.x.shape[1]
#                 hidden_size=128,
#                 use_FV=True):
#         super(Encoder, self).__init__()

#         self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
#         self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)
    
#     def forward(self, data):
#         x, edge_attr = data.x,data.edge_attr
#         data.h_node = self.nb_encoder(x)
#         data.h_edge = self.eb_encoder(edge_attr)
#         return data


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
    def __init__(self, node_dim, edge_dim, num_convs=15): # 15 (from paper)
        super().__init__()
        self.num_convs = num_convs
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.convs = nn.ModuleList(
            [copy.deepcopy(MeshGraphNetsConv(node_dim, edge_dim)) for _ in range(num_convs)]
        )

    def forward(self, data):
        for conv in self.convs:
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


class MeshGraphNets(nn.Module):
    """MeshGraphNets.
    """
    def __init__(self, node_input_size=3, edge_input_size=3,\
                  hidden_size=64):
        super().__init__()
        self.encoder   = Encoder(node_dim=node_input_size,\
                            edge_dim=edge_input_size,
                            hidden_size=hidden_size)
        self.processor = Processor(node_dim=hidden_size,edge_dim=hidden_size)
        self.decoder   = Decoder(node_dim=hidden_size,out_dim=2) # out_size = u,v
    
    def forward(self, data):
        data = self.encoder(data)
        data = self.processor(data)
        return self.decoder(data)

