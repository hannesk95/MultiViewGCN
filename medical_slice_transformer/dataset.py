import torch
from torch.utils.data import Dataset
import torchio as tio
import SimpleITK as sitk
import torchvision.transforms as T

class MedicalSliceTransformerDataset(Dataset):
    def __init__(self, data: list, labels: list):
        self.data = data
        self.labels = labels
        self.imagenet_norm = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # (16, 1, 32, 224, 224)

        sample = self.data[idx]        
        sample = tio.ScalarImage(sample)
        sample = tio.ToCanonical()(sample)
        sample = tio.Resize((224, 224, -1))(sample)
        sample = sample.tensor
        sample = sample.permute(0, 3, 1, 2)    # (C, D, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return sample, label