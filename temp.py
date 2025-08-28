import torch
from torch.nn import Linear, Sequential, ReLU, Dropout
from torch_geometric.nn.conv import GraphConv, GCNConv, ChebConv, SAGEConv

data = torch.load("/home/johannes/Data/SSD_1.9TB/MultiViewGCN/data/ucsf/glioma_four_sequences/UCSF-PDGM-0004_FLAIR_bias_DINOv2_24views_spherical_features.pt", weights_only=False)

model = Sequential(SAGEConv(in_channels=384, out_channels=86, aggr=["mean", "max"]),
                   Linear(in_features=16, out_features=2))

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {pytorch_total_params}")

model = SAGEConv(in_channels=384, out_channels=86, aggr=["mean", "max"]).cuda()
data = data.cuda()

out = model(data.x, data.edge_index)
out.shape