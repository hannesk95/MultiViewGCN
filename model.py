import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, GMMConv, SplineConv, NNConv, CGConv, BatchNorm, GraphNorm
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
import torchvision
import copy
from i3res import I3ResNet
from i3dense import I3DenseNet
import inspect
from torch_geometric.nn import GINConv, GINEConv
from torch.nn import Linear, ReLU, Sequential, LeakyReLU


# class PyGModel(torch.nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int = 32, pool = "mean", architecture = "GCN"):
#         super(PyGModel, self).__init__()

#         self.architecture = architecture
#         self.pool = pool

#         match architecture:
#             case "MLP":
#                 self.model = MLP(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=hidden_dim//2, num_layers=2)
            
#             case "GCN":
#                 self.model = GCN(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=hidden_dim//2, num_layers=2)

#         self.linear = torch.nn.Linear(in_features=hidden_dim//2, out_features=2)

#     def forward(self, x, edge_index, batch, batch_size, n_views):

#         if self.architecture == "MLP":
#             x = self.model(x=x, batch=batch, batch_size=batch_size)
#         else:
#             x = self.model(x=x, edge_index=edge_index, batch=batch, batch_size=batch_size)

#         x = x.reshape(-1, n_views, x.shape[-1])
        
#         if self.pool == "mean":
#             x = torch.mean(x, dim=1)
#         elif self.pool == "sum":
#             x = torch.sum(x, dim=1)
#         elif self.pool == "max":
#             x = torch.max(x, dim=1).values

#         out = self.linear(x)
        
#         return out

# class ViewGNN(torch.nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int = 32, pool = "mean", conv = "GCN"):
#         super(ViewGNN, self).__init__()
#         self.pool = pool
        
#         match conv:
#             case "GCN":           
#                 self.local_conv1 = GCNConv(input_dim, hidden_dim)
#                 self.local_conv2 = GCNConv(hidden_dim, hidden_dim)

#                 self.global_conv1 = GCNConv(hidden_dim, hidden_dim)
#                 self.global_conv2 = GCNConv(hidden_dim, hidden_dim)

#                 self.pool1 = SAGPooling(in_channels=hidden_dim, ratio=0.5)
#                 self.pool2 = SAGPooling(in_channels=hidden_dim, ratio=0.5)
            
#             case "SAGE":
#                 self.conv1 = SAGEConv(input_dim, hidden_dim)
#                 self.conv2 = SAGEConv(hidden_dim, 8)
            
#             case "GAT":
#                 self.conv1 = GATConv(input_dim, hidden_dim)
#                 self.conv2 = GATConv(hidden_dim, 8)
            
#             case "GIN":
#                 self.conv1 = GINConv(input_dim, hidden_dim)
#                 self.conv2 = GINConv(hidden_dim, 8)

#         self.lin1 = Linear(32, 8)
#         self.lin2 = Linear(8, 2)

#         self.local_lin1 = Linear(2*32, 8)
#         self.local_lin2 = Linear(8, 2)

#         self.global_lin1 = Linear(2*32, 8)
#         self.global_lin2 = Linear(8, 2)    

#     def forward(self, x, edge_index, batch):

#         batch_size = torch.max(batch).detach().cpu().item() + 1

#         # Local GCN
#         x = F.leaky_relu(self.local_conv1(x, edge_index), negative_slope=0.2)
#         x_local1 = torch.max(x.reshape(batch_size, -1, x.shape[-1]), dim=1)[0]

#         # Global GCN
#         data = Data(x=x, edge_index=edge_index, batch=batch)
#         data = self.add_virtual_node(data=data, batch_size=batch_size)
#         x_global1 = F.leaky_relu(self.global_conv1(data.x, data.edge_index), negative_slope=0.2)
#         x_global1 = x_global1.reshape(batch_size, -1, x_global1.shape[-1])
#         x_global1 = x_global1[:, -1, :]

#         # Node Pooling
#         pooled_data = self.pool1(x=x, edge_index=edge_index) # (x, connect_out.edge_index, connect_out.edge_attr, connect_out.batch, perm, score)
#         x, edge_index, batch = pooled_data[0], pooled_data[1], pooled_data[3]
        
#         # Local GCN
#         x = F.leaky_relu(self.local_conv2(x, edge_index), negative_slope=0.2) 
#         x_local2 = torch.max(x.reshape(batch_size, -1, x.shape[-1]), dim=1)[0]

#         # Global GCN
#         data = Data(x=x, edge_index=edge_index)
#         data = self.add_virtual_node(data=data, batch_size=batch_size)
#         x_global2 = F.leaky_relu(self.global_conv2(data.x, data.edge_index), negative_slope=0.2)
#         x_global2 = x_global2.reshape(batch_size, -1, x_global2.shape[-1])
#         x_global2 = x_global2[:, -1, :]

#         # Node Pooling
#         pooled_data = self.pool2(x=x, edge_index=edge_index) # (x, connect_out.edge_index, connect_out.edge_attr, connect_out.batch, perm, score)
#         x, edge_index, batch = pooled_data[0], pooled_data[1], pooled_data[3]

#         if self.pool == "mean":
#             # x = global_mean_pool(x, batch)
#             x = torch.mean(x.reshape(batch_size, -1, x.shape[-1]), dim=1)
#         elif self.pool == "sum":
#             # x = global_add_pool(x, batch)
#             x = torch.sum(x.reshape(batch_size, -1, x.shape[-1]), dim=1)
#         elif self.pool == "max":
#             # x = global_max_pool(x, batch)
#             x = torch.max(x.reshape(batch_size, -1, x.shape[-1]), dim=1)[0]
        
#         x = self.lin1(x)
#         x = F.leaky_relu(x, negative_slope=0.2)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin2(x)        

#         x_local = torch.cat((x_local1, x_local2), dim=1)
#         x_local = self.local_lin1(x_local)
#         x_local = F.leaky_relu(x_local, negative_slope=0.2)
#         x_local = F.dropout(x_local, p=0.5, training=self.training)
#         x_local = self.local_lin2(x_local)

#         x_global = torch.cat((x_global1, x_global2), dim=1)
#         x_global = self.global_lin1(x_global)
#         x_global = F.leaky_relu(x_global, negative_slope=0.2)
#         x_global = F.dropout(x_global, p=0.5, training=self.training)
#         x_global = self.global_lin2(x_global)
        
#         selected_nodes = torch.ones(10)
        
#         return x, x_local, x_global, selected_nodes
    

#     def add_virtual_node(self, data: Data, batch_size: int) -> Data:
#         """
#         Adds a virtual node to the graph which is connected to all other nodes.
#         Information flow is allowed only from the nodes to the virtual node.
        
#         Args:
#             data (Data): PyTorch Geometric Data object representing the graph.
        
#         Returns:
#             Data: Updated Data object with the virtual node added.
#         """       

#         # Batch size
#         # batch_size = torch.max(data.batch).detach().cpu().item() + 1

#         # Number of nodes in the graph
#         num_nodes = data.num_nodes // batch_size

#         # Feature dimension
#         feature_dim = data.x.size(1)

#         # Original node features
#         original_node_features = data.x.reshape(batch_size, num_nodes, feature_dim)
        
#         # Create a new node feature for the virtual node (e.g., all zeros)
#         virtual_node_feature = torch.zeros((batch_size, 1, feature_dim), device=data.x.device)
        
#         # Update node features to include the virtual node
#         new_node_features = torch.cat([original_node_features, virtual_node_feature], dim=1).reshape(-1, feature_dim)
#         data.x = new_node_features

#         # original_edge_indices = data.edge_index.reshape(2, -1, batch_size)
        
#         # Add edges from all original nodes to the virtual node
#         virtual_node_edges = torch.tensor([[i, num_nodes] for i in range(num_nodes)], device=data.edge_index.device).t()

#         virtual_node_edges = torch.unsqueeze(virtual_node_edges, dim=-1).repeat(1, 1, batch_size)
        
#         # Concatenate the new edges with the existing edge index
#         # new_edge_index = torch.cat([original_edge_indices, virtual_node_edges], dim=1).reshape(2, -1)

#         data.edge_index = virtual_node_edges.reshape(2, -1)
        
#         return data

# class GNN(torch.nn.Module):
#     def __init__(self, input_dim: int = 384, hidden_dim: int = 32, readout: str = "mean", ratio: float = 1.0):       
#         super(GNN, self).__init__()
#         self.readout = readout
#         self.ratio = ratio

#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, 8)

#         self.pool = SAGPooling(in_channels=8, ratio=self.ratio)
#         # self.pool = SimpleTanhAttn(feat_dim=8, retention_ratio=self.ratio)

#         if isinstance(self.pool, SimpleTanhAttn):
#             self.pool.reset_parameters()

#         elif isinstance(self.pool, SAGPooling):
#             self.pool.reset_parameters()

#         self.lin1 = Linear(8, 8)
#         self.lin2 = Linear(8, 2)        

#     def forward(self, x, edge_index, batch):
#         batch_size = torch.max(batch).detach().cpu().item() + 1
#         num_nodes = x.reshape(batch_size, -1, x.shape[-1]).shape[1]

#         x = self.conv1(x, edge_index)
#         x = F.leaky_relu(x, negative_slope=0.2)
        
#         x = self.conv2(x, edge_index)
#         x = F.leaky_relu(x, negative_slope=0.2)


#         if self.ratio < 1.0:
        
#             if isinstance(self.pool, SimpleTanhAttn):
#                 x = self.pool(x, batch)["x"]

#             elif isinstance(self.pool, SAGPooling):
#                 x = self.pool(x=x, edge_index=edge_index, batch=batch)[0]
#                 num_nodes = int(num_nodes*self.ratio)
                
#                 if self.readout == "mean":
#                     x = torch.mean(x.reshape(batch_size, num_nodes, x.shape[-1]), dim=1)
#                 elif self.readout == "sum":
#                     x = torch.sum(x.reshape(batch_size, num_nodes, x.shape[-1]), dim=1)
#                 elif self.readout == "max":
#                     x = torch.max(x.reshape(batch_size, num_nodes, x.shape[-1]), dim=1)[0]
#         else:
#             if self.readout == "mean":
#                 x = torch.mean(x.reshape(batch_size, num_nodes, x.shape[-1]), dim=1)
#             elif self.readout == "sum":
#                 x = torch.sum(x.reshape(batch_size, num_nodes, x.shape[-1]), dim=1)
#             elif self.readout == "max":
#                 x = torch.max(x.reshape(batch_size, num_nodes, x.shape[-1]), dim=1)[0]

        
#         x = self.lin1(x)
#         x = F.leaky_relu(x, negative_slope=0.2)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin2(x)

#         return x

# class MLP(torch.nn.Module):
#     def __init__(self, input_dim: int = 384, hidden_dim: int = 32, readout: str = "mean"):       
#         super(MLP, self).__init__()
#         self.readout = readout

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

#         if self.readout == "mean":
#             x = torch.mean(x, dim=1)
#         elif self.readout == "sum":
#             x = torch.sum(x, dim=1)
#         elif self.readout == "max":
#             x = torch.max(x, dim=1)[0]
        
#         x = self.lin1(x)
#         x = F.leaky_relu(x, negative_slope=0.2)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin2(x)

#         return x




# class GINE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers, readout):
#         super().__init__()

#         self.readout = readout

#         self.convs = torch.nn.ModuleList()
#         self.batch_norms = torch.nn.ModuleList()

#         for i in range(num_layers):
#             mlp = Sequential(
#                 Linear(in_channels, 2 * hidden_channels),
#                 BatchNorm(2 * hidden_channels),
#                 ReLU(),
#                 Linear(2 * hidden_channels, hidden_channels),
#             )
#             # conv = GINConv(mlp, train_eps=False)
#             conv = GINEConv(mlp, train_eps=False, edge_dim=9)

#             self.convs.append(conv)
#             self.batch_norms.append(BatchNorm(hidden_channels))

#             in_channels = hidden_channels

#         self.lin1 = Linear(hidden_channels, hidden_channels)
#         self.batch_norm1 = BatchNorm(hidden_channels)
#         self.lin2 = Linear(hidden_channels, out_channels)

#     def forward(self, x, edge_index, batch, edge_attr=None):
#         for conv, batch_norm in zip(self.convs, self.batch_norms):
#             x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
#             # x = F.relu(batch_norm(conv(x, edge_index)))

#         if self.readout == "mean":
#             x = global_mean_pool(x, batch)
#         elif self.readout == "sum":
#             x = global_add_pool(x, batch)        
        
#         x = F.relu(self.batch_norm1(self.lin1(x)))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin2(x)
#         # return F.log_softmax(x, dim=-1)
#         return x


class CNN(torch.nn.Module):
    def __init__(self, 
                 architecture: str = "M3D-ResNet50", 
                 pretrained: bool = True,
                 num_classes: int = 2):       
        super(CNN, self).__init__()

        self.architecture = architecture

        match architecture:
            case "M3D-ResNet50":
                from monai.networks.nets import ResNetFeatures
                self.model = ResNetFeatures(model_name=f"resnet50", pretrained=pretrained, spatial_dims=3, in_channels=1)
                self.pool = self.model.avgpool
                self.linear = torch.nn.Linear(self.get_last_conv_out_channels(self.model), num_classes)           
            case "M3D-ResNet34":
                from monai.networks.nets import ResNetFeatures
                self.model = ResNetFeatures(model_name=f"resnet34", pretrained=pretrained, spatial_dims=3, in_channels=1)
                self.pool = self.model.avgpool
                self.linear = torch.nn.Linear(self.get_last_conv_out_channels(self.model), num_classes)           
            case "M3D-ResNet18":
                from monai.networks.nets import ResNetFeatures
                self.model = ResNetFeatures(model_name=f"resnet18", pretrained=pretrained, spatial_dims=3, in_channels=1)
                self.pool = self.model.avgpool
                self.linear = torch.nn.Linear(self.get_last_conv_out_channels(self.model), num_classes)           
            case "M3D-ResNet10":
                from monai.networks.nets import ResNetFeatures
                self.model = ResNetFeatures(model_name=f"resnet10", pretrained=pretrained, spatial_dims=3, in_channels=1)
                self.pool = self.model.avgpool
                self.linear = torch.nn.Linear(self.get_last_conv_out_channels(self.model), num_classes)           
            
            case "ModelsGenesis":

                import unet3d

                # prepare the 3D model
                class TargetNet(nn.Module):
                    def __init__(self, base_model,n_class=num_classes):
                        super(TargetNet, self).__init__()

                        self.base_model = base_model
                        self.dense_1 = nn.Linear(512, 1024, bias=True)
                        self.dense_2 = nn.Linear(1024, n_class, bias=True)

                    def forward(self, x):
                        self.base_model(x)
                        self.base_out = self.base_model.out512
                        # This global average polling is for shape (N,C,H,W) not for (N, H, W, C)
                        # where N = batch_size, C = channels, H = height, and W = Width
                        self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0],-1)
                        self.linear_out = self.dense_1(self.out_glb_avg_pool)
                        final_out = self.dense_2( F.relu(self.linear_out))
                        return final_out
                        
                base_model = unet3d.UNet3D()

                #Load pre-trained weights
                weight_dir = 'pretrained_weights/Genesis_Chest_CT.pt'
                checkpoint = torch.load(weight_dir)
                state_dict = checkpoint['state_dict']
                unParalled_state_dict = {}
                for key in state_dict.keys():
                    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
                base_model.load_state_dict(unParalled_state_dict, strict=False)
                self.model = TargetNet(base_model)
                # target_model = TargetNet(base_model)
                # target_model = nn.DataParallel(target_model, device_ids = [i for i in range(torch.cuda.device_count())])

                # self.model = target_model
            
            case "I3D-DenseNet121":
                densenet = torchvision.models.densenet121(pretrained=True)
                self.model = I3DenseNet(copy.deepcopy(densenet), frame_nb=8, num_classes=num_classes)

            case "I3D-ResNet50":
                resnet = torchvision.models.resnet50(pretrained=True)
                self.model = I3ResNet(copy.deepcopy(resnet), frame_nb=16, class_nb=num_classes)
    
    def get_last_conv_out_channels(self, model):
        last_conv = None
        for layer in model.modules():
            if isinstance(layer, nn.Conv3d):
                last_conv = layer
        if last_conv is None:
            raise ValueError("No Conv3d layers found in model.")
        return last_conv.out_channels        

    def forward(self, x):

        if self.architecture == "ModelsGenesis":
            x = self.model(x)
        elif self.architecture == "I3D-DenseNet121":
            x = self.model(x)
        elif self.architecture == "I3D-ResNet50":
            x = self.model(x)            
        else:
            x = self.model(x)[-1]
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.linear(x)

        return x

class GNN(torch.nn.Module):
    # https://github.com/pyg-team/pytorch_geometric/discussions/2891
    def __init__(self, 
                 architecture: Literal["GCN", "GAT", "SAGE", "GIN", "GINE", "GMM", "Spline", "NN", "CG"] = "GCN",
                 aggregate: Literal["mean", "sum", "max"] = "mean",                 
                 input_dim: int = 384, 
                 hidden_dim: int = 32, 
                 num_classes: int = 2,
                 num_layers:int = 3,
                 readout: Literal["mean", "sum", "max"] = "mean",
                 hierarchical_readout: bool = False,
                 include_edge_attr: bool = False,
                 edge_attr_dim: int = 9, 
                 flow: Literal["source_to_target", "target_to_source"] = "target_to_source",):
           
        super(GNN, self).__init__()       

        self.hierarchical_readout = hierarchical_readout              
        
        match architecture:                    
            case "GAT":
                self.conv = GATConv
            case "GINE":
                self.conv = GINEConv
            case "GMM":
                self.conv = GMMConv
            case "Spline":
                self.conv = SplineConv
            case "NN":
                self.conv = NNConv
            case "CG":
                self.conv = CGConv
            case "GCN":
                self.conv = GCNConv
            case "SAGE":
                self.conv = SAGEConv
            case "GIN":
                self.conv = GINConv  
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

        x, edge_index, edge_attr, batch = data.x.to(torch.float32), data.edge_index.to(torch.long), data.edge_attr.to(torch.float32), data.batch

        features = []
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            if self.edge_attr_is_available:
                x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
            else:
                x = F.relu(batch_norm(conv(x, edge_index)))
            features.append(x)
        
        # x = torch.cat(features, dim=0)
        if self.hierarchical_readout:
            x = torch.cat(features, dim=-1)

        else:
            x = features[-1]

        x = self.pool(x, batch)
        x = self.linear(x)

        return x

class MLP(torch.nn.Module):
    def __init__(self, input_dim: int = 384, num_classes: int = 2, aggregation: str = "mean", n_views = None):       
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.aggregation = aggregation
        self.n_views = n_views
        assert self.n_views is not None, "Number of views must be provided."

        if self.aggregation == "MLP":             
            self.aggregate = nn.Sequential(
                    nn.Linear(self.n_views, 1),
                    nn.ReLU())                

        self.linear = Linear(self.input_dim, num_classes)

    def forward(self, x):

        # x = x.reshape(-1, self.n_views, self.input_dim)     # shape (batch_size, n_views, feature_dim)
        x = x.permute(0, 2, 1)                              # shape (batch_size, feature_dim, n_views)

        match self.aggregation:
            case "mean":
                x = torch.mean(x, dim=-1)
            case "sum":
                x = torch.sum(x, dim=-1)
            case "max":
                x = torch.max(x, dim=-1)[0]
            case "MLP":
                x = self.aggregate(x)
                x = x.squeeze(-1)
            
        x = self.linear(x)
        return x        
    