import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, BatchNorm, GraphNorm
from torch_geometric.nn.models import GCN, MLP
import torch.nn.functional as F
from torch.nn import Linear, Dropout, BatchNorm1d
from torch_geometric.nn.inits import uniform
from _delete.model_paper import SimpleTanhAttn
from torch_geometric.utils import to_undirected
import torch
import torch.nn as nn
import os
import glob
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn.inits import uniform
from typing import Literal, Union


class PyGModel(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, pool = "mean", architecture = "GCN"):
        super(PyGModel, self).__init__()

        self.architecture = architecture
        self.pool = pool

        match architecture:
            case "MLP":
                self.model = MLP(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=hidden_dim//2, num_layers=2)
            
            case "GCN":
                self.model = GCN(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=hidden_dim//2, num_layers=2)

        self.linear = torch.nn.Linear(in_features=hidden_dim//2, out_features=2)

    def forward(self, x, edge_index, batch, batch_size, n_views):

        if self.architecture == "MLP":
            x = self.model(x=x, batch=batch, batch_size=batch_size)
        else:
            x = self.model(x=x, edge_index=edge_index, batch=batch, batch_size=batch_size)

        x = x.reshape(-1, n_views, x.shape[-1])
        
        if self.pool == "mean":
            x = torch.mean(x, dim=1)
        elif self.pool == "sum":
            x = torch.sum(x, dim=1)
        elif self.pool == "max":
            x = torch.max(x, dim=1).values

        out = self.linear(x)
        
        return out

class ViewGNN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, pool = "mean", conv = "GCN"):
        super(ViewGNN, self).__init__()
        self.pool = pool
        
        match conv:
            case "GCN":           
                self.local_conv1 = GCNConv(input_dim, hidden_dim)
                self.local_conv2 = GCNConv(hidden_dim, hidden_dim)

                self.global_conv1 = GCNConv(hidden_dim, hidden_dim)
                self.global_conv2 = GCNConv(hidden_dim, hidden_dim)

                self.pool1 = SAGPooling(in_channels=hidden_dim, ratio=0.5)
                self.pool2 = SAGPooling(in_channels=hidden_dim, ratio=0.5)
            
            case "SAGE":
                self.conv1 = SAGEConv(input_dim, hidden_dim)
                self.conv2 = SAGEConv(hidden_dim, 8)
            
            case "GAT":
                self.conv1 = GATConv(input_dim, hidden_dim)
                self.conv2 = GATConv(hidden_dim, 8)
            
            case "GIN":
                self.conv1 = GINConv(input_dim, hidden_dim)
                self.conv2 = GINConv(hidden_dim, 8)

        self.lin1 = Linear(32, 8)
        self.lin2 = Linear(8, 2)

        self.local_lin1 = Linear(2*32, 8)
        self.local_lin2 = Linear(8, 2)

        self.global_lin1 = Linear(2*32, 8)
        self.global_lin2 = Linear(8, 2)    

    def forward(self, x, edge_index, batch):

        batch_size = torch.max(batch).detach().cpu().item() + 1

        # Local GCN
        x = F.leaky_relu(self.local_conv1(x, edge_index), negative_slope=0.2)
        x_local1 = torch.max(x.reshape(batch_size, -1, x.shape[-1]), dim=1)[0]

        # Global GCN
        data = Data(x=x, edge_index=edge_index, batch=batch)
        data = self.add_virtual_node(data=data, batch_size=batch_size)
        x_global1 = F.leaky_relu(self.global_conv1(data.x, data.edge_index), negative_slope=0.2)
        x_global1 = x_global1.reshape(batch_size, -1, x_global1.shape[-1])
        x_global1 = x_global1[:, -1, :]

        # Node Pooling
        pooled_data = self.pool1(x=x, edge_index=edge_index) # (x, connect_out.edge_index, connect_out.edge_attr, connect_out.batch, perm, score)
        x, edge_index, batch = pooled_data[0], pooled_data[1], pooled_data[3]
        
        # Local GCN
        x = F.leaky_relu(self.local_conv2(x, edge_index), negative_slope=0.2) 
        x_local2 = torch.max(x.reshape(batch_size, -1, x.shape[-1]), dim=1)[0]

        # Global GCN
        data = Data(x=x, edge_index=edge_index)
        data = self.add_virtual_node(data=data, batch_size=batch_size)
        x_global2 = F.leaky_relu(self.global_conv2(data.x, data.edge_index), negative_slope=0.2)
        x_global2 = x_global2.reshape(batch_size, -1, x_global2.shape[-1])
        x_global2 = x_global2[:, -1, :]

        # Node Pooling
        pooled_data = self.pool2(x=x, edge_index=edge_index) # (x, connect_out.edge_index, connect_out.edge_attr, connect_out.batch, perm, score)
        x, edge_index, batch = pooled_data[0], pooled_data[1], pooled_data[3]

        if self.pool == "mean":
            # x = global_mean_pool(x, batch)
            x = torch.mean(x.reshape(batch_size, -1, x.shape[-1]), dim=1)
        elif self.pool == "sum":
            # x = global_add_pool(x, batch)
            x = torch.sum(x.reshape(batch_size, -1, x.shape[-1]), dim=1)
        elif self.pool == "max":
            # x = global_max_pool(x, batch)
            x = torch.max(x.reshape(batch_size, -1, x.shape[-1]), dim=1)[0]
        
        x = self.lin1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)        

        x_local = torch.cat((x_local1, x_local2), dim=1)
        x_local = self.local_lin1(x_local)
        x_local = F.leaky_relu(x_local, negative_slope=0.2)
        x_local = F.dropout(x_local, p=0.5, training=self.training)
        x_local = self.local_lin2(x_local)

        x_global = torch.cat((x_global1, x_global2), dim=1)
        x_global = self.global_lin1(x_global)
        x_global = F.leaky_relu(x_global, negative_slope=0.2)
        x_global = F.dropout(x_global, p=0.5, training=self.training)
        x_global = self.global_lin2(x_global)
        
        selected_nodes = torch.ones(10)
        
        return x, x_local, x_global, selected_nodes
    

    def add_virtual_node(self, data: Data, batch_size: int) -> Data:
        """
        Adds a virtual node to the graph which is connected to all other nodes.
        Information flow is allowed only from the nodes to the virtual node.
        
        Args:
            data (Data): PyTorch Geometric Data object representing the graph.
        
        Returns:
            Data: Updated Data object with the virtual node added.
        """       

        # Batch size
        # batch_size = torch.max(data.batch).detach().cpu().item() + 1

        # Number of nodes in the graph
        num_nodes = data.num_nodes // batch_size

        # Feature dimension
        feature_dim = data.x.size(1)

        # Original node features
        original_node_features = data.x.reshape(batch_size, num_nodes, feature_dim)
        
        # Create a new node feature for the virtual node (e.g., all zeros)
        virtual_node_feature = torch.zeros((batch_size, 1, feature_dim), device=data.x.device)
        
        # Update node features to include the virtual node
        new_node_features = torch.cat([original_node_features, virtual_node_feature], dim=1).reshape(-1, feature_dim)
        data.x = new_node_features

        # original_edge_indices = data.edge_index.reshape(2, -1, batch_size)
        
        # Add edges from all original nodes to the virtual node
        virtual_node_edges = torch.tensor([[i, num_nodes] for i in range(num_nodes)], device=data.edge_index.device).t()

        virtual_node_edges = torch.unsqueeze(virtual_node_edges, dim=-1).repeat(1, 1, batch_size)
        
        # Concatenate the new edges with the existing edge index
        # new_edge_index = torch.cat([original_edge_indices, virtual_node_edges], dim=1).reshape(2, -1)

        data.edge_index = virtual_node_edges.reshape(2, -1)
        
        return data

class GNN(torch.nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 32, readout: str = "mean", ratio: float = 1.0):       
        super(GNN, self).__init__()
        self.readout = readout
        self.ratio = ratio

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 8)

        self.pool = SAGPooling(in_channels=8, ratio=self.ratio)
        # self.pool = SimpleTanhAttn(feat_dim=8, retention_ratio=self.ratio)

        if isinstance(self.pool, SimpleTanhAttn):
            self.pool.reset_parameters()

        elif isinstance(self.pool, SAGPooling):
            self.pool.reset_parameters()

        self.lin1 = Linear(8, 8)
        self.lin2 = Linear(8, 2)        

    def forward(self, x, edge_index, batch):
        batch_size = torch.max(batch).detach().cpu().item() + 1
        num_nodes = x.reshape(batch_size, -1, x.shape[-1]).shape[1]

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2)
        
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2)


        if self.ratio < 1.0:
        
            if isinstance(self.pool, SimpleTanhAttn):
                x = self.pool(x, batch)["x"]

            elif isinstance(self.pool, SAGPooling):
                x = self.pool(x=x, edge_index=edge_index, batch=batch)[0]
                num_nodes = int(num_nodes*self.ratio)
                
                if self.readout == "mean":
                    x = torch.mean(x.reshape(batch_size, num_nodes, x.shape[-1]), dim=1)
                elif self.readout == "sum":
                    x = torch.sum(x.reshape(batch_size, num_nodes, x.shape[-1]), dim=1)
                elif self.readout == "max":
                    x = torch.max(x.reshape(batch_size, num_nodes, x.shape[-1]), dim=1)[0]
        else:
            if self.readout == "mean":
                x = torch.mean(x.reshape(batch_size, num_nodes, x.shape[-1]), dim=1)
            elif self.readout == "sum":
                x = torch.sum(x.reshape(batch_size, num_nodes, x.shape[-1]), dim=1)
            elif self.readout == "max":
                x = torch.max(x.reshape(batch_size, num_nodes, x.shape[-1]), dim=1)[0]

        
        x = self.lin1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x

class MLP(torch.nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 32, readout: str = "mean"):       
        super(MLP, self).__init__()
        self.readout = readout

        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, 8)

        self.lin1 = Linear(8, 8)
        self.lin2 = Linear(8, 2)        

    def forward(self, x, batch_size):
        
        n_nodes = x.shape[0] // batch_size

        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        
        x = self.fc2(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = x.view(batch_size, n_nodes, -1)     

        if self.readout == "mean":
            x = torch.mean(x, dim=1)
        elif self.readout == "sum":
            x = torch.sum(x, dim=1)
        elif self.readout == "max":
            x = torch.max(x, dim=1)[0]
        
        x = self.lin1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x