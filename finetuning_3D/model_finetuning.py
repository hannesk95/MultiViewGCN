import sys
sys.path.append('/home/johannes/Data/SSD_1.9TB/MultiViewGCN/feature_extractors_3D/FMCIB/foundation-cancer-image-biomarker')
from fmcib.models import fmcib_model
import torch
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F


class FMCIB_Classifier(torch.nn.Module):
    def __init__(self):
        super(FMCIB_Classifier, self).__init__()
        self.fmcib = fmcib_model(eval_mode=False)
        
        self.linear_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()        

        for i in range(1):
            if i == 0:
                linear = Linear(4096, 24)
            else:
                linear = Linear(24, 24)
            
            self.linear_layers.append(linear)
            self.bn_layers.append(BatchNorm1d(24))

        self.linear = torch.nn.Linear(24, 2)

    def forward(self, x):
        x = self.fmcib(x)

        features = []
        for linear, batch_norm in zip(self.linear_layers, self.bn_layers):

            x = linear(x)
            x = F.relu(batch_norm(x))      
            features.append(x)                        

        x = features[-1]  
        x = self.linear(x)

        return x