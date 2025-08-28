import torch
from torch.utils.data import Dataset
from torch_geometric.utils import to_undirected
from torch_geometric.utils import to_networkx
import networkx as nx
import torch_geometric
import numpy as np

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
    def __init__(self, data: list, labels: list, topology: str, views: int):
        self.data = data
        self.labels = labels
        self.topology = topology
        self.views = views

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = torch.load(sample, weights_only=False)

        match self.topology:
            case "local":
                pass
            case "complete_uniform":                
                edge_index = self.fully_connected_edge_index(sample)                
                sample.edge_index = edge_index
            case "complete_uniform_wo_local":                
                edge_index = self.fully_connected_edge_index(sample)
                edges_without_direct_neighbors = self.remove_entries(edge_index, sample.edge_index)
                sample.edge_index = edges_without_direct_neighbors
            case "complete_weighted_hops_inverse_square":
                edge_weight, edge_index = self.compute_edge_weights(sample, decay="inverse_square")
                sample.edge_weight = edge_weight
                sample.edge_index = edge_index
            case "complete_weighted_hops_inverse":                
                edge_weight, edge_index = self.compute_edge_weights(sample, decay="inverse")
                sample.edge_weight = edge_weight
                sample.edge_index = edge_index
            case "complete_weighted_hops_linear":                
                edge_weight, edge_index = self.compute_edge_weights(sample, decay="linear")
                sample.edge_weight = edge_weight
                sample.edge_index = edge_index
            case "complete_weighted_geodesic":
                pass
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

    def compute_edge_weights(self, data: torch_geometric.data.Data, self_loops: bool = False, decay: str = "inverse") -> torch.Tensor:

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

        match decay:
            case "inverse_square":
                decay_factor = 2
            case "inverse":
                decay_factor = 1
            case "linear":
                pass
            case _:
                raise ValueError(f"Unknown decay type: {decay}")

        weights = []
        for i in range(edge_index.size(1)):
            source, target = edge_index[:, i]
            source = source.item()
            target = target.item()
            path = nx.shortest_path(G_nx, source=source, target=target)
            hop_count = len(path) - 1
            # weights.append(1/hop_count**2)

            if "inverse" in decay:
                weights.append(1/hop_count**decay_factor)
            elif "linear" in decay:
                weights.append(-0.2*hop_count+1.2) 
            else:
                raise ValueError(f"Unknown decay type: {decay}")

        return torch.tensor(weights, dtype=torch.float), edge_index

    def compute_geodesic_edge_weights(self, data: torch_geometric.data.Data) -> torch.Tensor:

        def geodesic_distances(points):
            """
            Compute geodesic (great-circle) distances between all pairs of points on a unit sphere.

            Parameters
            ----------
            points : ndarray of shape (N, 3)
                Each row is (x, y, z) coordinates of a point on the unit sphere.

            Returns
            -------
            dist_matrix : ndarray of shape (N, N)
                Symmetric matrix of geodesic distances (radians).
            """
            # Normalize points to be safe
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            P = points / norms

            # Dot products between all pairs (N x N)
            dot = P @ P.T

            # Cross products (N x N x 3) → norms (N x N)
            # Efficient way: use broadcasting
            cross = np.linalg.norm(P[:, None, :] * [1, 1, 1] - P[None, :, :], axis=-1)  # too naive

            # Better: explicit cross product using broadcasting
            cross = np.cross(P[:, None, :], P[None, :, :])
            cross_norm = np.linalg.norm(cross, axis=-1)

            # Geodesic distances
            dist = np.arctan2(cross_norm, dot)

            return dist

        # print(f"Loading Thomson sphere with {self.views} views...")
        vertices_faces = torch.load(f"thomson_graph_{self.views}_views.pt", weights_only=False)
        vertices = vertices_faces["vertices"]

        distances = geodesic_distances(vertices)
    
    def remove_entries(self, A, B):
        """
        Removes all rows of tensor B from tensor A.
        
        Args:
            A (torch.Tensor): A tensor of shape (2, N).
            B (torch.Tensor): A tensor of shape (2, M).
            
        Returns:
            torch.Tensor: A new tensor with rows of B removed from A.
        """
        # Check for empty tensors to avoid errors
        if A.numel() == 0 or B.numel() == 0:
            return A
        
        # Transpose the tensors to make each column a separate row for easier comparison
        A_t = A.t()
        B_t = B.t()
        
        # Reshape the tensors to a single dimension for comparison using torch.isin
        # This creates a unique identifier for each column (or point)
        # The columns are of shape (2, 1), so they are stacked and flattened
        A_flat = A_t.reshape(-1)
        B_flat = B_t.reshape(-1)
        
        # Check which elements of A_t are present in B_t
        # torch.isin returns a boolean tensor indicating membership
        is_in = torch.isin(A_flat, B_flat)
        
        # Reshape the boolean tensor back to match the original shape of A_t
        is_in = is_in.view(A.shape[1], -1).all(dim=1)
        
        # Filter out the rows of A that are not in B
        filtered_A = A_t[~is_in]
        
        # Transpose back to the original shape (2, K) where K <= N
        return filtered_A.t()



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