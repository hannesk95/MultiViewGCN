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
            x = torch.max(x, dim=1)

        out = self.linear(x)
        
        return out


class GNN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, pool = "mean", conv = "GCN"):
        super(GNN, self).__init__()
        self.pool = pool
        
        match conv:
            case "GCN":           
                self.conv1 = GCNConv(input_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, 8)
            
            case "SAGE":
                self.conv1 = SAGEConv(input_dim, hidden_dim)
                self.conv2 = SAGEConv(hidden_dim, 8)
            
            case "GAT":
                self.conv1 = GATConv(input_dim, hidden_dim)
                self.conv2 = GATConv(hidden_dim, 8)
            
            case "GIN":
                self.conv1 = GINConv(input_dim, hidden_dim)
                self.conv2 = GINConv(hidden_dim, 8)

        self.lin1 = Linear(8, 8)
        self.lin2 = Linear(8, 2)

        self.cls = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 2)
        )        

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2)
        pooled_view1 = global_max_pool(x, batch)

        x = self.conv2(x, edge_index) 
        x = F.leaky_relu(x, negative_slope=0.2)
        pooled_view2 = global_max_pool(x, batch)

        if self.pool == "mean":
            x = global_mean_pool(x, batch)
        elif self.pool == "sum":
            x = global_add_pool(x, batch)
        elif self.pool == "max":
            x = global_max_pool(x, batch)
        
        x = self.lin1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        selected_nodes = torch.ones(10)

        pooled_view = torch.cat((pooled_view1, pooled_view2), 1)
        pooled_view = self.cls(pooled_view)

        
        return x, selected_nodes

# class MLP(torch.nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int = 32, pool= "mean"):       
#         super(MLP, self).__init__()
#         self.pool = pool

#         self.fc1 = Linear(input_dim, hidden_dim)
#         self.fc2 = Linear(hidden_dim, 8)

#         self.lin1 = Linear(8, 8)
#         self.lin2 = Linear(8, 2)        

#     def forward(self, x, batch_size):
        
#         n_nodes = x.shape[0] // batch_size

#         x = self.fc1(x)
#         x = F.leaky_relu(x, negative_slope=0.2)
        
#         x = self.fc2(x)
#         x = F.leaky_relu(x, negative_slope=0.2)

#         x = x.view(batch_size, n_nodes, -1)     

#         if self.pool == "mean":
#             x = torch.mean(x, dim=1)
#         elif self.pool == "sum":
#             x = torch.sum(x, dim=1)
#         elif self.pool == "max":
#             x = torch.max(x, dim=1)
        
#         x = self.lin1(x)
#         x = F.leaky_relu(x, negative_slope=0.2)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin2(x)

#         return x
    