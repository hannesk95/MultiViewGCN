import torch
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, BatchNorm, GraphNorm
import torch.nn.functional as F
from torch.nn import Linear, Dropout, BatchNorm1d
from torch_geometric.nn.inits import uniform
from _delete.model_paper import SimpleTanhAttn
from torch_geometric.utils import to_undirected


class GCN(torch.nn.Module):
    def __init__(self, input_dim, pool="mean", prune=False, retention_ratio=1.0, fully_connected_graph=False):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)

        self.pool = pool
        self.prune = prune
        self.fully_connected_graph = fully_connected_graph
        
        self.conv1 = GCNConv(input_dim, 32)
        self.conv2 = GCNConv(32, 8)

        self.pooling = SimpleTanhAttn(feat_dim=8, retention_ratio=retention_ratio, aggregation=pool)

        self.lin1 = Linear(8, 8)
        self.lin2 = Linear(8, 2)        

    def forward(self, x, edge_index, batch):

        if self.fully_connected_graph:
            edges = torch.tensor([[0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4, 5,5,5,5,5],
                                [1,2,3,4,5, 0,2,3,4,5, 0,1,3,4,5, 0,1,2,4,5, 0,1,2,3,5, 0,1,2,3,4]])
            
            edge_index = to_undirected(edges)
            edge_index = edge_index.cuda()

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = self.conv2(x, edge_index) 
        x = F.leaky_relu(x, negative_slope=0.2)

        if self.prune:
            x = self.pooling(x, batch)
            selected_nodes = x.get('selected_nodes')
            x = x.get('x')
            
        else:
            x = global_add_pool(x, batch)
        
        x = self.lin1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return x, selected_nodes
    
class SAGE(torch.nn.Module):
    def __init__(self, input_dim, pool="mean", prune=False, retention_ratio=1.0, fully_connected_graph=False):
        super(SAGE, self).__init__()
        # torch.manual_seed(12345)

        self.pool = pool
        self.prune = prune
        self.fully_connected_graph = fully_connected_graph
        
        self.conv1 = SAGEConv(input_dim, 32)
        self.conv2 = SAGEConv(32, 8)

        self.pooling = SimpleTanhAttn(feat_dim=8, retention_ratio=retention_ratio, aggregation=pool)

        self.lin1 = Linear(8, 8)
        self.lin2 = Linear(8, 2)        

    def forward(self, x, edge_index, batch):

        if self.fully_connected_graph:
            edges = torch.tensor([[0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4, 5,5,5,5,5],
                                [1,2,3,4,5, 0,2,3,4,5, 0,1,3,4,5, 0,1,2,4,5, 0,1,2,3,5, 0,1,2,3,4]])
            
            edge_index = to_undirected(edges)
            edge_index = edge_index.cuda()

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = self.conv2(x, edge_index) 
        x = F.leaky_relu(x, negative_slope=0.2)

        if self.prune:
            x = self.pooling(x, batch)
            selected_nodes = x.get('selected_nodes')
            x = x.get('x')
            
        else:
            x = global_add_pool(x, batch)
        
        x = self.lin1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return x, selected_nodes
    
class GAT(torch.nn.Module):
    def __init__(self, input_dim, pool="mean", prune=False, retention_ratio=1.0, fully_connected_graph=False):
        super(GAT, self).__init__()
        # torch.manual_seed(12345)

        self.pool = pool
        self.prune = prune
        self.fully_connected_graph = fully_connected_graph
        
        self.conv1 = GATConv(input_dim, 32)
        self.conv2 = GATConv(32, 8)

        self.pooling = SimpleTanhAttn(feat_dim=8, retention_ratio=retention_ratio, aggregation=pool)

        self.lin1 = Linear(8, 8)
        self.lin2 = Linear(8, 2)        

    def forward(self, x, edge_index, batch):

        if self.fully_connected_graph:
            edges = torch.tensor([[0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4, 5,5,5,5,5],
                                [1,2,3,4,5, 0,2,3,4,5, 0,1,3,4,5, 0,1,2,4,5, 0,1,2,3,5, 0,1,2,3,4]])
            
            edge_index = to_undirected(edges)
            edge_index = edge_index.cuda()

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = self.conv2(x, edge_index) 
        x = F.leaky_relu(x, negative_slope=0.2)

        if self.prune:
            x = self.pooling(x, batch)
            selected_nodes = x.get('selected_nodes')
            x = x.get('x')
            
        else:
            x = global_add_pool(x, batch)
        
        x = self.lin1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return x, selected_nodes
    
class MLP(torch.nn.Module):
    def __init__(self, input_dim, pool="mean"):       
        super(MLP, self).__init__()
        # torch.manual_seed(12345)

        self.pool = pool

        self.fc1 = Linear(input_dim, 32)
        self.fc2 = Linear(32, 8)

        self.lin1 = Linear(8, 8)
        self.lin2 = Linear(8, 2)        

    def forward(self, x, batch_size):
        
        n_nodes = x.shape[0] // batch_size

        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        
        x = self.fc2(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = x.view(batch_size, n_nodes, -1)     

        if self.pool == "mean":
            x = torch.mean(x, dim=1)
        elif self.pool == "sum":
            x = torch.sum(x, dim=1)
        
        x = self.lin1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x
