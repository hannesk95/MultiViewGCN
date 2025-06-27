import torch
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F

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