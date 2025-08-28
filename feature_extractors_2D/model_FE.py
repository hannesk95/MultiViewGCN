import torch
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm, GCNConv
from weighted_SAGEConv import WeightedSAGEConv
from typing import Literal
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import inspect

from torch_geometric.nn import SAGEConv
from torch_geometric.nn.conv import MessagePassing
import torch
from torch import nn

# class WeightedSAGEConv(MessagePassing):
#     def __init__(self, in_channels, out_channels, **kwargs):
#         super().__init__(**kwargs)
#         self.lin = nn.Linear(in_channels, out_channels)
#         self.lin_update = nn.Linear(in_channels + out_channels, out_channels)

#     def forward(self, x, edge_index, edge_weight=None):
#         if edge_weight is None:
#             edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
#         return self.propagate(edge_index, x=x, edge_weight=edge_weight)

#     def message(self, x_j, edge_weight):
#         return edge_weight.view(-1, 1) * x_j  # apply weights to neighbor messages

#     def update(self, aggr_out, x):
#         return self.lin_update(torch.cat([x, aggr_out], dim=1))


class MLP(torch.nn.Module):
    def __init__(self, 
                 input_dim: int = None, 
                 hidden_dim: int = 32, 
                 num_layers: int = 1, 
                 num_classes: int = 2, 
                 readout: str = "mean",
                 hierarchical_readout: bool = True):       
        super(MLP, self).__init__()

        assert input_dim is not None, "Input dimension must be specified."

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.readout = readout
        self.hierarchical_readout = hierarchical_readout

        self.linear_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        

        for i in range(num_layers):
            if i == 0:
                linear = Linear(self.input_dim, self.hidden_dim)
            else:
                linear = Linear(self.hidden_dim, self.hidden_dim)
            
            self.linear_layers.append(linear)
            self.bn_layers.append(BatchNorm1d(self.hidden_dim))
            
        
        self.linear = torch.nn.Linear(hidden_dim, num_classes) if not self.hierarchical_readout else torch.nn.Linear(hidden_dim*num_layers, num_classes)    
        
    
    def forward(self, x):            

        features = []
        for linear, batch_norm in zip(self.linear_layers, self.bn_layers):

            x = linear(x)
            x = F.relu(batch_norm(x))      
            features.append(x)                        

        x = torch.cat(features, dim=-1) if self.hierarchical_readout else features[-1]  
        x = self.linear(x)

        return x
    
class GNN(torch.nn.Module):
    # https://github.com/pyg-team/pytorch_geometric/discussions/2891
    def __init__(self, 
                 architecture: Literal["SAGE"] = "SAGE",
                 aggregate: Literal["mean", "sum", "max"] = ["mean", "max"],                 
                #   aggregate: Literal["mean", "sum", "max"] = "sum",                 
                 input_dim: int = 384, 
                 hidden_dim: int = 16, 
                 num_classes: int = 2,
                 num_layers:int = 1,
                 readout: Literal["mean", "sum", "max"] = "mean",
                 hierarchical_readout: bool = True,
                 include_edge_attr: bool = False,
                 edge_attr_dim: int = 9, 
                 flow: Literal["source_to_target", "target_to_source"] = "target_to_source",):
           
        super(GNN, self).__init__() 

        self.aggregate = aggregate
        self.hierarchical_readout = hierarchical_readout

        match architecture:                
            case "SAGE":
                # self.conv = SAGEConv
                self.conv = WeightedSAGEConv
            case "WeightedSAGE":
                self.conv = WeightedSAGEConv
            case "GCN":
                self.conv = GCNConv
            case "GraphConv":
                from torch_geometric.nn import GraphConv
                self.conv = GraphConv
            case _:                
                raise ValueError("Given GNN architecture is not available!")                  

        match readout:
            case "mean":
                self.pool = global_mean_pool
            case "sum":
                self.pool = global_add_pool
            case "max":
                self.pool = global_max_pool
            case _:
                raise ValueError(f"Invalid readout method: {readout}. Choose from 'mean', 'sum', or 'max'.")    
        
        self.edge_attr_is_available = True if "edge_attr" in inspect.signature(self.conv.forward).parameters.keys() else False
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # input_dim = input_dim * 2

        for i in range(num_layers):
            if i == 0:
                if self.edge_attr_is_available:
                    conv = self.conv(input_dim, hidden_dim, aggr=aggregate, flow=flow, edge_dim=edge_attr_dim)
                else:
                    conv = self.conv(input_dim, hidden_dim, aggr=aggregate, flow=flow)
            else:
                if self.edge_attr_is_available:
                    conv = self.conv(hidden_dim, hidden_dim, aggr=aggregate, flow=flow, edge_dim=edge_attr_dim)
                else:    
                    conv = self.conv(hidden_dim, hidden_dim, aggr=aggregate, flow=flow)
            
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        self.linear = torch.nn.Linear(hidden_dim, num_classes) if not hierarchical_readout else torch.nn.Linear(hidden_dim*num_layers, num_classes)
        
        


    def forward(self, data):

        x = data.x.to(torch.float32)
        edge_index = data.edge_index.to(torch.long)
        edge_attr = data.edge_attr.to(torch.float32)
        batch = data.batch

        if data.edge_weight is not None:
            edge_weight = data.edge_weight
        else:
            edge_weight = None

        features = []
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            if self.edge_attr_is_available:
                x = F.relu(batch_norm(conv(x, edge_index, edge_attr, edge_weight=edge_weight)))
            else:
                x = F.relu(batch_norm(conv(x, edge_index, edge_weight=edge_weight)))
            features.append(x)
        
        x = torch.cat(features, dim=-1) if self.hierarchical_readout else features[-1]

        x = self.pool(x, batch)
        x = self.linear(x)       

        return x