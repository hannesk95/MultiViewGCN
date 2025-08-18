import torch
from torch.utils.data import Dataset
import torchio as tio
import SimpleITK as sitk
import torchvision.transforms as T

# class MedicalSliceTransformerDataset(Dataset):
#     def __init__(self, data: list, labels: list):
#         self.data = data
#         self.labels = labels
#         self.imagenet_norm = T.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#             )

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):

#         # (16, 1, 32, 224, 224)

#         sample = self.data[idx]        
#         sample = tio.ScalarImage(sample)
#         sample = tio.Resample((1, 1, 1))(sample)
#         sample = tio.ToCanonical()(sample)
#         sample = tio.Resize((224, 224, -1))(sample)
#         sample = sample.tensor
#         sample = sample.permute(0, 3, 1, 2)    # (C, D, H, W)
#         sample = self.zscore_normalize_depthwise(sample)
#         label = torch.tensor(self.labels[idx], dtype=torch.long)

#         return sample, label

#     def zscore_normalize_depthwise(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
#         """
#         Apply z-score normalization to each depth slice of a 4D tensor (C, D, H, W).
        
#         Args:
#             x (torch.Tensor): Input tensor of shape (C, D, H, W).
#             eps (float): Small value to avoid division by zero.
            
#         Returns:
#             torch.Tensor: Normalized tensor of shape (C, D, H, W).
#         """
#         if x.ndim != 4:
#             raise ValueError(f"Expected input of shape (C, D, H, W), but got {x.shape}")

#         C, D, H, W = x.shape

#         # Compute mean and std per (C, D) slice over (H, W)
#         mean = x.mean(dim=(2, 3), keepdim=True)  # shape (C, D, 1, 1)
#         std = x.std(dim=(2, 3), keepdim=True)    # shape (C, D, 1, 1)

#         # Normalize
#         out = (x - mean) / (std + eps)

#         assert out.shape == (C, D, H, W), f"Shape mismatch: expected {(C, D, H, W)}, got {out.shape}"
#         return out
    
class MedicalSliceTransformerDataset(Dataset):
    def __init__(self, data: list, labels: list):
        self.data = data
        self.labels = labels        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]        
        sample = torch.load(sample, weights_only=False)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return sample, label

