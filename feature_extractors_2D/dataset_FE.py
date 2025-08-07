import torch
from torch.utils.data import Dataset
from torch_geometric.utils import to_undirected
from torch_geometric.utils import to_networkx
import networkx as nx
import torch_geometric

class PlanarDatasetMLP(Dataset):
    def __init__(self, data: list, labels: list):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]        
        sample = torch.load(sample, weights_only=False)      
        sample = sample.view(-1).view(1, -1)  # Flatten the sample to a 1D tensor

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return sample, label
    

class SphericalDatasetGNN(Dataset):
    def __init__(self, data: list, labels: list, topology: str):
        self.data = data
        self.labels = labels
        self.topology = topology

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = torch.load(sample, weights_only=False)

        match self.topology:
            case "local":
                pass
            case "complete":                
                edge_index = self.fully_connected_edge_index(sample)                
                sample.edge_index = edge_index
            case "weighted":                
                edge_weight, edge_index = self.compute_edge_weights(sample)
                sample.edge_weight = edge_weight
                sample.edge_index = edge_index
            case _:
                raise ValueError(f"Unknown topology: {self.topology}")        

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return sample, label

    def fully_connected_edge_index(self, data: torch_geometric.data.Data, self_loops: bool = False) -> torch.Tensor:

        N = data.x.size(0)

        row = torch.arange(N).repeat_interleave(N)
        col = torch.arange(N).repeat(N)
        
        edge_index = torch.stack([row, col], dim=0)

        if not self_loops:
            # Remove self-loops (i.e., i → i)
            mask = row != col
            edge_index = edge_index[:, mask]

        return edge_index    

    def compute_edge_weights(self, data: torch_geometric.data.Data, self_loops: bool = False) -> torch.Tensor:

        N = data.x.size(0)  # Number of nodes in the graph

        row = torch.arange(N).repeat_interleave(N)
        col = torch.arange(N).repeat(N)
        
        edge_index = torch.stack([row, col], dim=0)

        if not self_loops:
            # Remove self-loops (i.e., i → i)
            mask = row != col
            edge_index = edge_index[:, mask]

        # Assume you have a Data object (undirected)
        G_nx = to_networkx(data, to_undirected=True)

        weights = []
        for i in range(edge_index.size(1)):
            source, target = edge_index[:, i]
            source = source.item()
            target = target.item()
            path = nx.shortest_path(G_nx, source=source, target=target)
            hop_count = len(path) - 1
            weights.append(1/hop_count**2)

        return torch.tensor(weights, dtype=torch.float), edge_index     

class SphericalDatasetMLP(Dataset):
    def __init__(self, data: list, labels: list):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = torch.load(sample, weights_only=False)
        sample = sample.x.view(1, -1)  # Flatten the sample to a 1D tensor

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return sample, label